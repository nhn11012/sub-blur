# -*- coding: utf-8 -*-
"""
Hard-Sub / Logo Remover GUI (Blur)
- Preview video, Play/Pause, seek.
- Vẽ nhiều vùng thủ công (click-kéo trái). Right-click để xóa vùng dưới con trỏ. Clear để xóa tất cả.
- Tự động tìm văn bản trong các vùng đã chọn và làm mờ bằng Gaussian blur.
- Ghi video:
    * Nếu có FFmpeg (PATH) và bật “Use FFmpeg” -> x264 + copy audio gốc.
    * Nếu không -> VideoWriter (mp4v, video-only).
- Lưu cạnh file gốc: <name>_clean.mp4
- Tooltip cho từng điều khiển (đã sửa, không còn dùng IntVar._root()).
- Thông báo khi xong và hỏi mở thư mục.

Python 3.8. Yêu cầu: opencv-python, Pillow
"""
from __future__ import annotations

import os, sys, time, threading, shutil, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

APP_TITLE = "Hard-Sub / Logo Remover (GUI) — Auto + Manual"
MAX_PREVIEW_W, MAX_PREVIEW_H = 900, 720

# ------------------------ tooltip (đã sửa) ------------------------
class ToolTip:
    """Tooltip đơn giản cho widget Tk/ttk."""
    def __init__(self, widget, text, wraplength=320):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        # đặt tooltip ngay dưới widget
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("Segoe UI", 9), wraplength=self.wraplength
        )
        label.pack(ipadx=6, ipady=4)

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None

# ------------------------ helpers ------------------------
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def safe_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        return 30.0
    return float(fps)

def open_ffmpeg_writer_with_audio(out_path: str, in_path: str, W: int, H: int, fps: float,
                                  crf=18, preset="veryfast", pix_fmt="yuv420p") -> subprocess.Popen:
    cmd = [
        "ffmpeg","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",f"{fps:.6f}","-i","pipe:0",
        "-i", in_path,
        "-map","0:v:0","-map","1:a:0?",
        "-c:v","libx264","-preset",preset,"-crf",str(crf),
        "-pix_fmt", pix_fmt,
        "-c:a","copy",
        "-shortest",
        out_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def open_ffmpeg_writer_video_only(out_path: str, W: int, H: int, fps: float,
                                  crf=18, preset="veryfast", pix_fmt="yuv420p") -> subprocess.Popen:
    cmd = [
        "ffmpeg","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",f"{fps:.6f}","-i","pipe:0",
        "-c:v","libx264","-preset",preset,"-crf",str(crf),
        "-pix_fmt", pix_fmt,
        out_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def mux_audio_to_video(video_no_audio: str, input_audio_src: str, output_path: str):
    cmd = ["ffmpeg","-y","-i",video_no_audio,"-i",input_audio_src,
           "-map","0:v:0","-map","1:a:0?","-c:v","copy","-c:a","copy","-shortest", output_path]
    subprocess.run(cmd, check=True)

def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = frame_bgr.copy()
    red = np.zeros_like(out); red[:,:,2] = 255
    m3 = cv2.merge([mask, mask, mask])
    out = np.where(m3>0, (alpha*red+(1-alpha)*out).astype(np.uint8), out)
    return out

# ------------------------ auto mask core ------------------------
def detect_sub_pixels(frame: np.ndarray, roi: Tuple[int,int,int,int],
                      thresh_block=31, thresh_C=-10,
                      morph_w=13, morph_h=5,
                      min_area=600, min_aspect=2.5,
                      detect_yellow=True, detect_blue=False) -> np.ndarray:
    H, W = frame.shape[:2]
    x, y, w, h = roi
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x));  h = max(1, min(h, H-y))

    roi_img = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    tb = max(3, (thresh_block|1))
    bin_ad = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, tb, thresh_C)

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros((h,w), dtype=np.uint8)
    if detect_yellow:
        m = cv2.inRange(hsv, np.array([15,80,160], np.uint8), np.array([40,255,255], np.uint8))
        color_mask = cv2.bitwise_or(color_mask, m)
    if detect_blue:
        m = cv2.inRange(hsv, np.array([90,60,60], np.uint8), np.array([130,255,255], np.uint8))
        color_mask = cv2.bitwise_or(color_mask, m)

    raw = cv2.bitwise_or(bin_ad, color_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1,morph_w), max(1,morph_h)))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=1)

    nb, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered = np.zeros_like(closed)
    for i in range(1, nb):
        area = stats[i, cv2.CC_STAT_AREA]
        w_i  = stats[i, cv2.CC_STAT_WIDTH]
        h_i  = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w_i / max(1.0, float(h_i))
        if area >= min_area and aspect >= min_aspect:
            filtered[labels==i] = 255

    mask_full = np.zeros((H,W), dtype=np.uint8)
    mask_full[y:y+h, x:x+w] = filtered
    return mask_full

def build_auto_mask(video_path: str, mode: str, band_ratio: float,
                    max_samples: int, vote_ratio: float, stride_seconds: float,
                    dilate_iter: int, mask_vert_pad: int,
                    detect_yellow: bool, detect_blue: bool) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video để auto-detect")

    fps = safe_fps(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if mode == "band":
        band_h = int(round(H * band_ratio))
        roi = (0, H - band_h, W, band_h)
    else:
        roi = (0, 0, W, H)

    stride = max(1, int(round(stride_seconds * fps)))
    idxs = list(range(0, total, stride))
    if len(idxs) > max_samples:
        step = len(idxs) / max_samples
        idxs = [idxs[int(round(i*step))] for i in range(max_samples)]
    ns = max(1, len(idxs))

    heat = np.zeros((H, W), dtype=np.uint16)
    for fidx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        sm = detect_sub_pixels(frame, roi,
                               detect_yellow=detect_yellow, detect_blue=detect_blue)
        heat += (sm>0).astype(np.uint16)

    cap.release()

    thr = max(1, int(np.ceil(vote_ratio * ns)))
    mask = (heat >= thr).astype(np.uint8) * 255

    if mask_vert_pad > 0:
        kernel_pad = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2*mask_vert_pad+1))
        mask = cv2.dilate(mask, kernel_pad, iterations=1)

    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask, roi

# ------------------------ data ------------------------
@dataclass
class Rect:
    x:int; y:int; w:int; h:int
    def contains(self, px:int, py:int) -> bool:
        return self.x <= px <= self.x+self.w and self.y <= py <= self.y+self.h

# ------------------------ GUI ------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x840")

        # video state
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.W = self.H = 0
        self.fps = 30.0
        self.total_frames = 0
        self.cur_frame_idx = 0
        self.playing = False
        self.preview_tk = None
        self.scale_x = self.scale_y = 1.0

        # manual rects + auto mask
        self.rects: List[Rect] = []
        self.temp_rect_id = None
        self.dragging = False
        self.start_px = self.start_py = 0
        self.auto_mask: Optional[np.ndarray] = None
        self.auto_roi = (0,0,0,0)

        # processing
        self.proc_thread: Optional[threading.Thread] = None
        self.stop_flag = False

        self._build_ui()
        self._update_buttons()
        self.after(100, self._ui_loop)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8); top.pack(side=tk.TOP, fill=tk.X)
        self.btn_open = ttk.Button(top, text="Chọn video…", command=self._choose_video)
        self.btn_open.pack(side=tk.LEFT)
        ToolTip(self.btn_open, "Chọn file video để xem trước và xử lý.")

        self.lbl_path = ttk.Label(top, text="(chưa chọn)", width=90)
        self.lbl_path.pack(side=tk.LEFT, padx=8)

        self.btn_play = ttk.Button(top, text="▶ Play", command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4); ToolTip(self.btn_play, "Play/Pause preview video.")

        self.btn_prev = ttk.Button(top, text="⟸", width=3, command=self._step_back)
        self.btn_prev.pack(side=tk.LEFT); ToolTip(self.btn_prev, "Lùi 1 frame.")
        self.btn_next = ttk.Button(top, text="⟹", width=3, command=self._step_forward)
        self.btn_next.pack(side=tk.LEFT); ToolTip(self.btn_next, "Tới 1 frame.")

        mid = ttk.Frame(self, padding=8); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: canvas + timeline
        left = ttk.Frame(mid); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(left, bg="#202020", width=900, height=700, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<Button-3>", self._on_right_click)

        self.scale_pos = ttk.Scale(left, from_=0, to=100, orient=tk.HORIZONTAL, command=self._on_seek)
        self.scale_pos.pack(side=tk.TOP, fill=tk.X, pady=6)
        ToolTip(self.scale_pos, "Kéo để tua nhanh đến vị trí mong muốn.")

        # Right: controls
        right = ttk.LabelFrame(mid, text="Cài đặt & Preset", padding=8)
        right.pack(side=tk.LEFT, fill=tk.Y)

        # Radius / Blur kernel
        ttk.Label(right, text="Radius / Blur kernel (bán kính)").pack(anchor="w")
        self.var_radius = tk.IntVar(value=3)
        self.sb_radius = ttk.Spinbox(right, from_=1, to=12, textvariable=self.var_radius, width=6)
        self.sb_radius.pack(anchor="w", pady=(0,8))
        ToolTip(self.sb_radius, "Bán kính lấp pixel. Chữ đậm/viền dày → tăng 4–6.")

        # Padding & Dilate
        ttk.Label(right, text="Mask padding (px)").pack(anchor="w")
        self.var_pad = tk.IntVar(value=2)
        self.sb_pad = ttk.Spinbox(right, from_=0, to=12, textvariable=self.var_pad, width=6)
        self.sb_pad.pack(anchor="w", pady=(0,8))
        ToolTip(self.sb_pad, "Nới vùng vẽ thêm vài px để bắt viền chữ.")

        ttk.Label(right, text="Dilate iterations").pack(anchor="w")
        self.var_dilate = tk.IntVar(value=1)
        self.sb_dilate = ttk.Spinbox(right, from_=0, to=6, textvariable=self.var_dilate, width=6)
        self.sb_dilate.pack(anchor="w", pady=(0,8))
        ToolTip(self.sb_dilate, "Giãn mask để bắt stroke dày / răng cưa.")

        # Show mask overlay
        self.var_show_mask = tk.BooleanVar(value=True)
        self.chk_show_mask = ttk.Checkbutton(right, text="Show mask overlay", variable=self.var_show_mask, command=self._redraw)
        self.chk_show_mask.pack(anchor="w", pady=(2,8))
        ToolTip(self.chk_show_mask, "Hiển thị lớp mask đỏ đè lên preview để dễ canh.")

        # FFmpeg
        self.var_ffmpeg = tk.BooleanVar(value=has_ffmpeg())
        self.chk_ffmpeg = ttk.Checkbutton(right, text="Use FFmpeg (x264 + copy audio)", variable=self.var_ffmpeg)
        self.chk_ffmpeg.pack(anchor="w", pady=(2,2))
        ttk.Label(right, text="Khuyên dùng nếu đã cài FFmpeg vào PATH.").pack(anchor="w")
        ToolTip(self.chk_ffmpeg, "Bật: giữ âm thanh gốc, chất lượng nén tốt hơn.\nTắt: fallback VideoWriter (video-only).")

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        # Auto Preset group
        auto = ttk.LabelFrame(right, text="Preset TỰ ĐỘNG (tùy chọn)", padding=8)
        auto.pack(fill=tk.X, pady=(0,8))

        ttk.Label(auto, text="Chế độ Auto").pack(anchor="w")
        self.var_auto_mode = tk.StringVar(value="off")  # off|band|full
        self.rb_off  = ttk.Radiobutton(auto, text="Off (chỉ vùng vẽ thủ công)", variable=self.var_auto_mode, value="off")
        self.rb_band = ttk.Radiobutton(auto, text="Bottom band (dải đáy)", variable=self.var_auto_mode, value="band")
        self.rb_full = ttk.Radiobutton(auto, text="Full frame (toàn khung)", variable=self.var_auto_mode, value="full")
        self.rb_off.pack(anchor="w"); self.rb_band.pack(anchor="w"); self.rb_full.pack(anchor="w")
        ToolTip(self.rb_band, "Tự phát hiện chữ ở dải đáy — hợp với video 9:16/16:9 có sub ở dưới.")
        ToolTip(self.rb_full, "Tự phát hiện trên toàn khung — dùng khi logo/chữ ở nơi khác.")

        ttk.Label(auto, text="Band height ratio (độ cao dải đáy)").pack(anchor="w")
        self.var_band_ratio = tk.DoubleVar(value=0.28)
        self.sb_band_ratio = ttk.Spinbox(auto, from_=0.05, to=0.8, increment=0.01, textvariable=self.var_band_ratio, width=6)
        self.sb_band_ratio.pack(anchor="w", pady=(0,6))
        ToolTip(self.sb_band_ratio, "Tỷ lệ chiều cao dải đáy so với toàn khung (chỉ dùng khi chọn Bottom band).")

        ttk.Label(auto, text="Sampling stride (giây) / Max samples").pack(anchor="w")
        f_row = ttk.Frame(auto); f_row.pack(anchor="w")
        self.var_stride = tk.DoubleVar(value=0.5)
        self.var_maxs  = tk.IntVar(value=80)
        self.sb_stride = ttk.Spinbox(f_row, from_=0.1, to=3.0, increment=0.1, textvariable=self.var_stride, width=6)
        self.sb_maxs   = ttk.Spinbox(f_row, from_=10, to=200, textvariable=self.var_maxs, width=6)
        self.sb_stride.pack(side=tk.LEFT)
        self.sb_maxs.pack(side=tk.LEFT, padx=6)
        ToolTip(f_row, "Mỗi stride giây sẽ lấy 1 khung để vote; tối đa max samples.")

        ttk.Label(auto, text="Vote ratio (0–1)").pack(anchor="w")
        self.var_vote = tk.DoubleVar(value=0.30)
        self.sb_vote = ttk.Spinbox(auto, from_=0.10, to=0.80, increment=0.01, textvariable=self.var_vote, width=6)
        self.sb_vote.pack(anchor="w", pady=(0,6))
        ToolTip(self.sb_vote, "Pixel xuất hiện ≥ ratio * số mẫu sẽ vào auto mask.")

        self.var_mask_pad = tk.IntVar(value=4)
        ttk.Label(auto, text="Mask vertical pad (px)").pack(anchor="w")
        self.sb_mask_pad = ttk.Spinbox(auto, from_=0, to=16, textvariable=self.var_mask_pad, width=6)
        self.sb_mask_pad.pack(anchor="w", pady=(0,6))
        ToolTip(self.sb_mask_pad, "Nới mask theo chiều dọc để bắt stroke viền.")

        self.var_auto_dilate = tk.IntVar(value=1)
        ttk.Label(auto, text="Auto dilate iterations").pack(anchor="w")
        self.sb_auto_dilate = ttk.Spinbox(auto, from_=0, to=4, textvariable=self.var_auto_dilate, width=6)
        self.sb_auto_dilate.pack(anchor="w", pady=(0,6))

        self.var_yellow = tk.BooleanVar(value=True)
        self.var_blue   = tk.BooleanVar(value=False)
        self.chk_yellow = ttk.Checkbutton(auto, text="Detect yellow", variable=self.var_yellow)
        self.chk_blue   = ttk.Checkbutton(auto, text="Detect blue",   variable=self.var_blue)
        self.chk_yellow.pack(anchor="w"); self.chk_blue.pack(anchor="w")
        ToolTip(self.chk_yellow, "Bắt chữ phụ đề màu vàng (thường gặp).")
        ToolTip(self.chk_blue, "Bắt chữ/karaoke màu xanh lam (tuỳ video).")

        btns = ttk.Frame(auto); btns.pack(fill=tk.X, pady=(6,0))
        self.btn_preview_auto = ttk.Button(btns, text="Preview Auto Mask", command=self._preview_auto)
        self.btn_preview_auto.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ToolTip(self.btn_preview_auto, "Chạy dò tự động theo cài đặt và hiển thị lớp mask đỏ để kiểm tra.")
        self.lbl_auto = ttk.Label(auto, text="Auto mask: (chưa tạo)")
        self.lbl_auto.pack(anchor="w", pady=(4,0))

        ttk.Separator(right).pack(fill=tk.X, pady=10)

        # Run + folder
        self.btn_clear = ttk.Button(right, text="Clear regions", command=self._clear_rects)
        self.btn_clear.pack(fill=tk.X, pady=(0,4))
        ToolTip(self.btn_clear, "Xoá toàn bộ vùng thủ công đã vẽ.")

        self.btn_run = ttk.Button(right, text="Remove Now", command=self._start_process)
        self.btn_run.pack(fill=tk.X, pady=4)
        ToolTip(self.btn_run, "Bắt đầu xử lý: auto mask (nếu bật) + vùng thủ công -> inpaint -> xuất video.")

        self.btn_open_folder = ttk.Button(right, text="Open Output Folder", command=self._open_out_dir, state="disabled")
        self.btn_open_folder.pack(fill=tk.X, pady=2)
        ToolTip(self.btn_open_folder, "Mở thư mục chứa video đã xử lý.")

        # Status bar
        bottom = ttk.Frame(self, padding=(8,0,8,8)); bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self.var_status).pack(side=tk.LEFT)
        self.prog = ttk.Progressbar(bottom, orient=tk.HORIZONTAL, mode="determinate", length=280)
        self.prog.pack(side=tk.RIGHT)

    # ---------- video ops ----------
    def _choose_video(self):
        path = filedialog.askopenfilename(title="Chọn video",
            filetypes=[("Video","*.mp4;*.mov;*.mkv;*.avi;*.webm"),("All files","*.*")])
        if not path: return
        self._load_video(path)

    def _load_video(self, path: str):
        if self.cap: self.cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", f"Không mở được video:\n{path}")
            return
        self.cap = cap
        self.video_path = path
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = safe_fps(cap)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cur_frame_idx = 0
        self.playing = False
        self.rects.clear()
        self.auto_mask = None
        self.lbl_path.config(text=path)
        self.scale_pos.config(from_=0, to=max(1, self.total_frames-1))
        self._seek_to(0)
        self._update_buttons()
        self._status(f"Loaded: {os.path.basename(path)} | {self.W}x{self.H} @ {self.fps:.2f} | frames={self.total_frames}")
        self.btn_open_folder.config(state="disabled")
        self.lbl_auto.config(text="Auto mask: (chưa tạo)")

    def _read_frame(self, idx:int) -> Optional[np.ndarray]:
        if not self.cap: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        return f if ok else None

    def _seek_to(self, idx:int):
        idx = max(0, min(idx, self.total_frames-1))
        self.cur_frame_idx = idx
        fr = self._read_frame(idx)
        if fr is not None: self._draw(fr)

    def _toggle_play(self):
        if not self.cap: return
        self.playing = not self.playing
        self.btn_play.config(text="❚❚ Pause" if self.playing else "▶ Play")

    def _step_back(self):
        if self.cap:
            self._seek_to(self.cur_frame_idx-1); self.scale_pos.set(self.cur_frame_idx)

    def _step_forward(self):
        if self.cap:
            self._seek_to(self.cur_frame_idx+1); self.scale_pos.set(self.cur_frame_idx)

    def _on_seek(self, s: str):
        if self.cap:
            self._seek_to(int(float(s)))

    def _ui_loop(self):
        if self.cap and self.playing:
            ni = self.cur_frame_idx + 1
            if ni >= self.total_frames:
                self.playing = False; self.btn_play.config(text="▶ Play")
            else:
                self._seek_to(ni); self.scale_pos.set(ni)
        self.after(int(1000/30), self._ui_loop)

    # ---------- draw / mouse ----------
    def _draw(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        scale = min(MAX_PREVIEW_W/W, MAX_PREVIEW_H/H, 1.0)
        dw, dh = int(W*scale), int(H*scale)
        self.scale_x, self.scale_y = W/dw, H/dh

        show = frame_bgr.copy()
        if self.var_show_mask.get() and self.auto_mask is not None:
            show = overlay_mask(show, self.auto_mask)

        rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        if scale != 1.0: rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(rgb)
        self.preview_tk = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.config(width=dw, height=dh)
        self.canvas.create_image(0,0, anchor=tk.NW, image=self.preview_tk)

        for i, r in enumerate(self.rects, 1):
            x0 = int(r.x/self.scale_x); y0 = int(r.y/self.scale_y)
            x1 = int((r.x+r.w)/self.scale_x); y1 = int((r.y+r.h)/self.scale_y)
            self.canvas.create_rectangle(x0,y0,x1,y1, outline="#00ff88", width=2)
            self.canvas.create_text(x0+6, y0+12, anchor="w", text=f"ROI {i}", fill="#00ff88")

    def _redraw(self):
        fr = self._read_frame(self.cur_frame_idx)
        if fr is not None: self._draw(fr)

    def _on_down(self, e):
        if not self.cap: return
        self.dragging = True
        self.start_px, self.start_py = e.x, e.y
        self.temp_rect_id = self.canvas.create_rectangle(self.start_px, self.start_py, e.x, e.y,
                                                         outline="#ff4444", width=2, dash=(4,2))

    def _on_drag(self, e):
        if self.dragging and self.temp_rect_id:
            self.canvas.coords(self.temp_rect_id, self.start_px, self.start_py, e.x, e.y)

    def _on_up(self, e):
        if not self.dragging: return
        self.dragging = False
        x0,y0,x1,y1 = self.start_px, self.start_py, e.x, e.y
        if abs(x1-x0)<4 or abs(y1-y0)<4:
            self.canvas.delete(self.temp_rect_id); self.temp_rect_id=None; return
        sx0,sy0 = min(x0,x1), min(y0,y1)
        sx1,sy1 = max(x0,x1), max(y0,y1)
        fx = int(sx0*self.scale_x); fy = int(sy0*self.scale_y)
        fw = int((sx1-sx0)*self.scale_x); fh = int((sy1-sy0)*self.scale_y)
        self.rects.append(Rect(fx,fy,fw,fh))
        self.canvas.delete(self.temp_rect_id); self.temp_rect_id=None
        self._redraw()

    def _on_right_click(self, e):
        if not self.rects: return
        fx,fy = int(e.x*self.scale_x), int(e.y*self.scale_y)
        for i, r in enumerate(self.rects):
            if r.contains(fx,fy):
                del self.rects[i]; break
        self._redraw()

    def _clear_rects(self):
        self.rects.clear(); self._redraw()

    # ---------- auto preset ----------
    def _preview_auto(self):
        if not self.video_path:
            messagebox.showwarning("Chưa có video", "Hãy chọn video trước.")
            return
        mode = self.var_auto_mode.get()
        if mode == "off":
            self.auto_mask = None
            self.lbl_auto.config(text="Auto mask: OFF")
            self._redraw()
            return
        try:
            mask, roi = build_auto_mask(
                self.video_path,
                mode=("band" if mode=="band" else "full"),
                band_ratio=float(self.var_band_ratio.get()),
                max_samples=int(self.var_maxs.get()),
                vote_ratio=float(self.var_vote.get()),
                stride_seconds=float(self.var_stride.get()),
                dilate_iter=int(self.var_auto_dilate.get()),
                mask_vert_pad=int(self.var_mask_pad.get()),
                detect_yellow=bool(self.var_yellow.get()),
                detect_blue=bool(self.var_blue.get())
            )
            self.auto_mask, self.auto_roi = mask, roi
            self.lbl_auto.config(text=f"Auto mask: {mode.upper()} ✓  (roi={roi})")
            self._redraw()
        except Exception as e:
            messagebox.showerror("Auto detect lỗi", str(e))

    # ---------- processing ----------
    def _start_process(self):
        if not self.video_path:
            messagebox.showwarning("Thiếu video", "Hãy chọn video trước.")
            return

        base, _ = os.path.splitext(self.video_path)
        out_path = base + "_clean.mp4"

        for b in (self.btn_run, self.btn_open, self.btn_play, self.btn_prev, self.btn_next, self.btn_preview_auto):
            b.config(state=tk.DISABLED)
        self.btn_open_folder.config(state=tk.DISABLED)
        self.stop_flag = False
        self.prog.config(maximum=max(1, self.total_frames), value=0)
        self._status("Processing...")

        radius = int(self.var_radius.get())
        pad = int(self.var_pad.get())
        dilate_it = int(self.var_dilate.get())
        use_ff = bool(self.var_ffmpeg.get())
        detect_yellow = bool(self.var_yellow.get())
        detect_blue = bool(self.var_blue.get())

        t = threading.Thread(
            target=self._worker,
            args=(self.video_path, out_path, self.rects[:], radius, pad, dilate_it, use_ff, detect_yellow, detect_blue),
            daemon=True
        )
        t.start(); self.proc_thread = t

    def _worker(self, in_path: str, out_path: str, rects: List[Rect],
                radius: int, pad_px: int, dilate_it: int, use_ffmpeg: bool,
                detect_yellow: bool, detect_blue: bool):
        try:
            cap = cv2.VideoCapture(in_path)
            if not cap.isOpened():
                raise RuntimeError("Không mở được video input.")
            fps = safe_fps(cap)
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            tmp_no_audio = None
            if use_ffmpeg and has_ffmpeg():
                try:
                    writer = open_ffmpeg_writer_with_audio(out_path, in_path, W, H, fps)
                except Exception:
                    writer = open_ffmpeg_writer_video_only(out_path, W, H, fps)
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                tmp_no_audio = out_path
                writer = cv2.VideoWriter(tmp_no_audio, fourcc, fps, (W,H))
                if not writer:
                    raise RuntimeError("Không khởi tạo VideoWriter.")

            processed = 0; last_upd = time.time()
            while True:
                ok, frame = cap.read()
                if not ok: break

                if isinstance(writer, cv2.VideoWriter):
                    writer.write(out)
                else:
                    writer.stdin.write(out.tobytes())

                processed += 1
                now = time.time()
                if now-last_upd > 0.05:
                    self.after(0, self.prog.config, {'value': processed})
                    last_upd = now

            cap.release()
            if isinstance(writer, cv2.VideoWriter):
                writer.release()
            else:
                writer.stdin.close(); writer.wait()

            if tmp_no_audio and has_ffmpeg():
                try:
                    mux_audio_to_video(tmp_no_audio, in_path, out_path)
                except Exception:
                    pass

            self.after(0, self._on_done, True, out_path, "")
        except Exception as e:
            self.after(0, self._on_done, False, None, str(e))

    def _on_done(self, ok: bool, out_path: Optional[str], msg: str):
        for b in (self.btn_run, self.btn_open, self.btn_play, self.btn_prev, self.btn_next, self.btn_preview_auto):
            b.config(state=tk.NORMAL)
        if ok and out_path:
            self._status(f"Done: {out_path}")
            self.btn_open_folder.config(state=tk.NORMAL)
            if messagebox.askyesno("Hoàn tất", "Xử lý xong video.\nBạn có muốn mở thư mục chứa file xuất không?"):
                self._open_out_dir(os.path.dirname(out_path))
        else:
            self._status("ERROR: " + msg)
            messagebox.showerror("Lỗi", msg)
            self.btn_open_folder.config(state=tk.DISABLED)

    def _open_out_dir(self, path: Optional[str]=None):
        d = path or (os.path.dirname(self.video_path) if self.video_path else None)
        if not d: return
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)  # type: ignore
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception:
            messagebox.showinfo("Folder", d)

    def _update_buttons(self):
        st = tk.NORMAL if self.cap else tk.DISABLED
        for w in (self.btn_play, self.btn_prev, self.btn_next, self.btn_run, self.btn_preview_auto):
            w.config(state=st)

    def _status(self, txt:str):
        self.var_status.set(txt)

# ------------------------ run ------------------------
def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
