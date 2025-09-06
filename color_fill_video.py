import random
import math
from typing import Tuple, List
from pathlib import Path
import os
from uuid import uuid4
import threading

import cv2
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    AudioFileClip,
    ColorClip,
    vfx,
    afx,
)
import tkinter as tk
from tkinter import ttk, messagebox
import itertools


def _palette_bgr() -> List[Tuple[int, int, int]]:
    """Fixed allowed palette in BGR order for OpenCV.

    Colors mirror the provided set:
    RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, DKGRAY, LTGRAY,
    orange(255,165,0), brown(139,69,19), pink(255,192,203), purple(128,0,128),
    teal(0,128,128), gold(255,215,0), dark green(0,128,0), maroon(128,0,0),
    steel blue(70,130,180), tomato(255,99,71), turquoise(64,224,208)
    """
    rgb = [
        (255, 0, 0),      # RED
        (0, 255, 0),      # GREEN
        (0, 0, 255),      # BLUE
        (255, 255, 0),    # YELLOW
        (0, 255, 255),    # CYAN
        (255, 0, 255),    # MAGENTA
        (68, 68, 68),     # DKGRAY (Android Color.DKGRAY)
        (204, 204, 204),  # LTGRAY (Android Color.LTGRAY)
        (255, 165, 0),    # ORANGE
        (139, 69, 19),    # BROWN
        (255, 192, 203),  # PINK
        (128, 0, 128),    # PURPLE
        (0, 128, 128),    # TEAL
        (255, 215, 0),    # GOLD
        (0, 128, 0),      # DARK GREEN
        (128, 0, 0),      # MAROON
        (70, 130, 180),   # STEEL BLUE
        (255, 99, 71),    # TOMATO
        (64, 224, 208),   # TURQUOISE
    ]
    return [(b, g, r) for (r, g, b) in rgb]


PALETTE_BGR = _palette_bgr()

# Sub-palettes for human-like mode (avoid neutrals to keep image lively)
HUMAN_VIVID_BGR: List[Tuple[int, int, int]] = [
    c for c in PALETTE_BGR
    if c not in [
        (255, 255, 255),           # white in BGR
        (68, 68, 68),              # dkgray
        (204, 204, 204),           # ltgray
    ]
]

# Warm/cool grouping from allowed palette
HUMAN_WARM: List[Tuple[int, int, int]] = [
    (0, 0, 255),      # RED
    (0, 165, 255),    # ORANGE
    (0, 215, 255),    # GOLD/YELLOW-ish
    (71, 99, 255),    # TOMATO
    (203, 192, 255),  # PINK
    (0, 0, 128),      # PURPLE (BGR)
    (19, 69, 139),    # BROWN (BGR from RGB(139,69,19))
    (0, 0, 128),      # MAROON approximated already included, keep once
]
HUMAN_COOL: List[Tuple[int, int, int]] = [
    (255, 0, 0),      # BLUE
    (255, 255, 0),    # CYAN
    (128, 128, 0),    # TEAL (BGR from RGB(0,128,128))
    (208, 224, 64),   # TURQUOISE (BGR from RGB(64,224,208))
    (180, 130, 70),   # STEEL BLUE (BGR from RGB(70,130,180))
    (0, 128, 0),      # GREEN
    (0, 128, 0),      # DARK GREEN (same BGR)
]


def random_vivid_bgr(rng: random.Random) -> Tuple[int, int, int]:
    """Pick a random color from the fixed allowed palette (BGR)."""
    return rng.choice(PALETTE_BGR)


def generate_human_palette(rng: random.Random, n: int = 16) -> List[Tuple[int, int, int]]:
    """Return a pleasant palette order using only allowed colors.

    Mixes warm and cool lists to keep harmony, avoiding white/greys.
    """
    warm = HUMAN_WARM.copy()
    cool = HUMAN_COOL.copy()
    rng.shuffle(warm)
    rng.shuffle(cool)
    merged = list(itertools.chain.from_iterable(zip(warm, cool)))
    # If uneven lengths, append remainder
    if len(warm) != len(cool):
        tail = warm[len(cool):] if len(warm) > len(cool) else cool[len(warm):]
        merged.extend(tail)
    # Ensure we have enough colors
    while len(merged) < n:
        merged.extend(merged)
    return merged[:n]


def compute_masks(
    img_bgr: np.ndarray,
    white_thresh: int = 240,
    black_thresh: int = 50,
    exclude_top_px: int = 0,
    exclude_bottom_px: int = 0,
):
    """Compute masks for white fillable regions and linework (black).

    Returns (white_mask, line_mask)
    - white_mask: binary uint8 where white fillable pixels are 255
    - line_mask: binary uint8 where black linework pixels are 255
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Black lines (and black shapes) mask
    _, line_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)

    # White areas mask – accept "near white" as fillable
    _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

    # Remove line pixels from fillable mask
    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(line_mask))

    # Close tiny gaps and hug outlines to avoid unfilled fringes near lines
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Ensure we didn't grow into lines
    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(line_mask))

    # Exclude top/bottom strips from being colored
    h = img_bgr.shape[0]
    top_end = min(exclude_top_px, h)
    bottom_start = max(0, h - exclude_bottom_px)
    if top_end > 0:
        white_mask[:top_end, :] = 0
    if bottom_start < h:
        white_mask[bottom_start:, :] = 0
    return white_mask, line_mask


def fill_regions_progressively(
    img_bgr: np.ndarray,
    duration: float = 20.0,
    fps: int = 30,
    seed: int | None = None,
    white_thresh: int = 240,
    black_thresh: int = 50,
    exclude_top_px: int = 500,
    exclude_bottom_px: int = 500,
    codec: str = "mp4v",
    output_path: str = "output.mp4",
    human_like: bool = False,
    min_region_area: int = 100,
    tail_secs: float = 7.0,
):
    """Fill white regions with random colors and write an animation video.

    The function writes exactly duration*fps frames to `output_path`.
    """
    rng = random.Random(seed)

    h, w = img_bgr.shape[:2]
    total_frames_target = max(1, int(round(duration * fps)))
    white_mask, line_mask = compute_masks(
        img_bgr,
        white_thresh=white_thresh,
        black_thresh=black_thresh,
        exclude_top_px=exclude_top_px,
        exclude_bottom_px=exclude_bottom_px,
    )

    # Connected components with stats/centroids for white regions
    # Use 4-connectivity to respect thin line boundaries
    num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(white_mask, connectivity=4)

    # Exclude any white region that touches the image border (likely background)
    h_idx = [0, img_bgr.shape[0]-1]
    w_idx = [0, img_bgr.shape[1]-1]
    border_labels = set()
    for r in h_idx:
        row = labels[r, :]
        border_labels.update(np.unique(row[row > 0]).tolist())
    for c in w_idx:
        col = labels[:, c]
        border_labels.update(np.unique(col[col > 0]).tolist())

    region_ids = [
        rid
        for rid in range(1, num_labels)
        if rid not in border_labels and stats[rid, cv2.CC_STAT_AREA] >= max(1, int(min_region_area))
    ]

    # Optionally order regions like a human would color (spatially coherent)
    if human_like and region_ids:
        h_img, w_img = img_bgr.shape[:2]
        center = np.array([h_img / 2.0, w_img / 2.0])
        centroids = {rid: np.array([cents[rid][1], cents[rid][0]]) for rid in region_ids}  # y,x -> row,col
        # Start near center
        start = min(region_ids, key=lambda rid: np.linalg.norm(centroids[rid] - center))
        ordered = [start]
        remaining = set(region_ids)
        remaining.remove(start)
        current = start
        while remaining:
            next_rid = min(remaining, key=lambda rid: np.linalg.norm(centroids[rid] - centroids[current]))
            ordered.append(next_rid)
            remaining.remove(next_rid)
            current = next_rid
        region_ids = ordered

    if len(region_ids) == 0:
        raise ValueError("No white regions detected to fill.")

    if not human_like:
        rng.shuffle(region_ids)

    # Decide how many regions to color per emitted frame
    # For human-like mode, emit a frame for every colored region to reduce pauses.
    regions_per_frame = 1 if human_like else max(1, math.ceil(len(region_ids) / total_frames_target))

    # Prepare the writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open video writer. Try a different codec (e.g., 'avc1', 'MJPG')."
        )

    try:
        current = img_bgr.copy()
        frames_written = 0
        colored = np.zeros_like(white_mask, dtype=bool)

        kernel_dilate = np.ones((2, 2), np.uint8)
        palette = generate_human_palette(rng, n=16) if human_like else None
        last_color = None
        # In human-like mode, keep using the same color for a run of regions
        run_remaining = 0

        # Build a simple neighbor map to reduce same-color touching
        neighbor_map: dict[int, set[int]] = {}
        if human_like:
            lab = labels
            k3 = np.ones((3, 3), np.uint8)
            for rid in region_ids:
                x, y, w, h = stats[rid, cv2.CC_STAT_LEFT], stats[rid, cv2.CC_STAT_TOP], stats[rid, cv2.CC_STAT_WIDTH], stats[rid, cv2.CC_STAT_HEIGHT]
                y0 = max(0, y - 1)
                x0 = max(0, x - 1)
                y1 = min(lab.shape[0], y + h + 1)
                x1 = min(lab.shape[1], x + w + 1)
                window = lab[y0:y1, x0:x1]
                m = (window == rid).astype(np.uint8) * 255
                md = cv2.dilate(m, k3, iterations=1)
                ring = (md > 0) & (m == 0)
                neigh = set(np.unique(window[ring]).tolist())
                neigh.discard(0)
                neigh.discard(rid)
                neighbor_map[rid] = neigh

        assigned_color: dict[int, Tuple[int, int, int]] = {}
        for i, region_id in enumerate(region_ids, start=1):
            # Assign a color for this region
            if human_like:
                if run_remaining <= 0 or last_color is None:
                    # Start a new color run (longer runs -> more same color usage)
                    last_color = palette[(i // 3) % len(palette)]  # slowly rotate palette
                    run_remaining = rng.randint(4, 10)  # use same color across 4–10 regions
                # Try to avoid matching already-colored neighbors
                neighbor_colors = {assigned_color.get(n) for n in neighbor_map.get(region_id, set())}
                neighbor_colors.discard(None)
                color = last_color
                if color in neighbor_colors:
                    # pick an alternative from the same palette
                    for c_try in palette:
                        if c_try not in neighbor_colors:
                            color = c_try
                            break
                run_remaining -= 1
            else:
                color = random_vivid_bgr(rng)
            mask = labels == region_id
            # Slightly dilate region mask to cover tiny gaps, but never over lines
            m8 = (mask.astype(np.uint8) * 255)
            m8 = cv2.dilate(m8, kernel_dilate, iterations=1)
            mask = (m8 > 0) & (~line_mask.astype(bool))
            colored |= mask
            current[mask] = color
            assigned_color[region_id] = color

            # Emit a frame every `regions_per_frame` colored regions
            if i % regions_per_frame == 0:
                frame = current.copy()
                # Reimpose linework to keep outlines crisp
                frame[line_mask.astype(bool)] = (0, 0, 0)
                writer.write(frame)
                frames_written += 1

        # After finishing all regions, hold the final image for tail_secs seconds
        # (composition may place splash over a portion of this tail)
        final_frame = current.copy()
        final_frame[line_mask.astype(bool)] = (0, 0, 0)
        tail_frames = int(round(tail_secs * fps))
        for _ in range(tail_frames):
            writer.write(final_frame)
            frames_written += 1
    finally:
        writer.release()
    return frames_written, fps, final_frame


def pick_random_png(pics_dir: Path) -> Path:
    candidates = [p for p in pics_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if not candidates:
        raise FileNotFoundError(f"No .png files found in {pics_dir}")
    return random.choice(candidates)


def next_unique_path(folder: Path, stem: str, suffix: str) -> Path:
    """Return a path in `folder` named `stem+suffix` or with _NNN appended.

    Example: stem='image', suffix='.mp4' -> image.mp4, image_001.mp4, ...
    """
    base = folder / f"{stem}{suffix}"
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = folder / f"{stem}_{n:03d}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def next_unique_image_path(folder: Path, stem: str, suffix: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return next_unique_path(folder, stem, suffix)


def pick_random_music(music_dir: Path) -> Path | None:
    if not music_dir.exists():
        return None
    exts = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
    candidates = [p for p in music_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        return None
    return random.choice(candidates)


def process_single_image(
    input_path: Path,
    base_dir: Path,
    finished_dir: Path,
    DURATION: float,
    FPS: int,
    WHITE_THRESH: int,
    BLACK_THRESH: int,
    EXCLUDE_TOP: int,
    EXCLUDE_BOTTOM: int,
    CODEC: str,
    HUMAN_LIKE: bool,
):
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    temp_path = finished_dir / f"_tmp_{input_path.stem}_{uuid4().hex}.mp4"

    # Timing for end-card sequence
    PRE_HOLD = 1.5    # seconds of finished frame before splash appears
    FADE_IN = 2.0     # slower fade-in for smoother transition
    POST_HOLD = 3.0   # seconds splash remains fully visible
    TAIL = PRE_HOLD + FADE_IN + POST_HOLD

    frames_written, out_fps, final_frame = fill_regions_progressively(
        img,
        duration=DURATION,
        fps=FPS,
        seed=None,
        white_thresh=WHITE_THRESH,
        black_thresh=BLACK_THRESH,
        exclude_top_px=EXCLUDE_TOP,
        exclude_bottom_px=EXCLUDE_BOTTOM,
        codec=CODEC,
        output_path=str(temp_path),
        human_like=HUMAN_LIKE,
        min_region_area=100,
        tail_secs=TAIL,
    )

    # Save last frame
    colored_pics_dir = finished_dir / "colored_pics"
    final_img_path = next_unique_image_path(colored_pics_dir, input_path.stem + "_final", ".png")
    cv2.imwrite(str(final_img_path), final_frame)

    # Compose with logo and music
    base_clip = VideoFileClip(str(temp_path))
    video_duration = base_clip.duration
    # Find logo in common formats (png/jpg/jpeg)
    def find_logo_image(folder: Path) -> Path | None:
        for name in (
            "splash-1080x2400.png",  # preferred new name
            # Backwards-compatibility fallbacks
            "logo_with_name_full_small.png",
            "logo_with_name_full_small.jpg",
            "logo_with_name_full_small.jpeg",
            "logo_with_name_full_small.JPG",
            "logo_with_name_full_small.JPEG",
            "logo_with_name_full_small.PNG",
        ):
            p = folder / name
            if p.exists():
                return p
        return None

    logo_path = find_logo_image(base_dir)
    overlays = []
    if logo_path is not None and logo_path.exists():
        # Full-screen overlay: match the video frame size
        logo_clip = ImageClip(str(logo_path)).resize((base_clip.w, base_clip.h))
        # Start so that we keep PRE_HOLD seconds with no overlay, then FADE_IN + POST_HOLD
        start_t = max(0, video_duration - (FADE_IN + POST_HOLD))
        logo_duration = min(FADE_IN + POST_HOLD, video_duration - start_t)
        # Create a mask that fades from transparent (0) to opaque (1)
        mask_clip = (
            ColorClip(size=base_clip.size, color=1, ismask=True)
            .set_start(start_t)
            .set_duration(logo_duration)
            .fx(vfx.fadein, FADE_IN)
        )
        logo_clip = (
            logo_clip.set_start(start_t)
            .set_duration(logo_duration)
            .set_mask(mask_clip)
            .set_position((0, 0))
        )
        overlays.append(logo_clip)

    final_clip = CompositeVideoClip([base_clip] + overlays, size=base_clip.size)

    music_path = pick_random_music(base_dir / "music")
    if music_path is not None:
        try:
            audio = AudioFileClip(str(music_path))
            if audio.duration > final_clip.duration:
                audio = audio.subclip(0, final_clip.duration)
            # Make the background music subtle
            audio = audio.fx(afx.volumex, 0.2)
            final_clip = final_clip.set_audio(audio)
        except Exception:
            pass

    output_path = next_unique_path(finished_dir, input_path.stem, ".mp4")
    final_clip.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(finished_dir / f"_tmp_{uuid4().hex}.m4a"),
        remove_temp=True,
        threads=os.cpu_count() or 2,
        verbose=False,
        logger=None,
    )

    base_clip.close()
    final_clip.close()
    try:
        os.remove(temp_path)
    except OSError:
        pass


def main():
    # Configuration constants (can later be added to the GUI)
    DURATION = 20.0  # seconds
    FPS = 30
    WHITE_THRESH = 240  # accept slightly more near-white as white
    BLACK_THRESH = 50
    EXCLUDE_TOP = 500
    EXCLUDE_BOTTOM = 500
    CODEC = "mp4v"  # try "MJPG" if mp4v doesn't work on your system
    MIN_REGION_AREA = 100  # ignore regions smaller than 10x10 pixels

    base_dir = Path(__file__).resolve().parent
    pics_dir = base_dir / "pics"
    finished_dir = base_dir / "finished"
    finished_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(pics_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".png"]

    # Build GUI --------------------------------------------------------------
    root = tk.Tk()
    root.title("Color Fill Video Generator")
    root.geometry("460x260")

    frm = ttk.Frame(root, padding=12)
    frm.pack(expand=True, fill=tk.BOTH)

    lbl_info = ttk.Label(frm, text=f"PNG images found in pics/: {len(images)}")
    lbl_info.pack(anchor="w")

    if not images:
        ttk.Label(frm, text="No PNG images found. Add images to pics/ and restart.").pack(anchor="w", pady=(8, 0))

    # How many videos to make
    row = ttk.Frame(frm)
    row.pack(fill=tk.X, pady=(12, 0))
    ttk.Label(row, text="How many videos to make:").pack(side=tk.LEFT)
    num_var = tk.IntVar(value=max(1, len(images) or 1))
    spn = ttk.Spinbox(row, from_=1, to=9999, textvariable=num_var, width=8)
    spn.pack(side=tk.LEFT, padx=(8, 0))

    # True random off (human-like coloring)
    rand_frame = ttk.Frame(frm)
    rand_frame.pack(fill=tk.X, pady=(12, 0))
    human_var = tk.BooleanVar(value=False)
    chk = ttk.Checkbutton(rand_frame, text="True random off (human-like)", variable=human_var)
    chk.pack(anchor="w")

    # Progress
    pbar = ttk.Progressbar(frm, mode="determinate", maximum=100)
    pbar.pack(fill=tk.X, pady=(16, 0))
    status_var = tk.StringVar(value="Idle")
    lbl_status = ttk.Label(frm, textvariable=status_var)
    lbl_status.pack(anchor="w", pady=(6, 0))

    btn_frame = ttk.Frame(frm)
    btn_frame.pack(fill=tk.X, pady=(12, 0))
    start_btn = ttk.Button(btn_frame, text="Start")
    start_btn.pack(side=tk.LEFT)
    close_btn = ttk.Button(btn_frame, text="Close")
    close_btn.pack(side=tk.RIGHT)
    cancel_flag = {"stop": False}

    def on_start():
        if not images:
            messagebox.showerror("No images", "No PNG images found in pics/.")
            return
        n = int(num_var.get())
        if n <= 0:
            messagebox.showerror("Invalid input", "Please enter a positive number.")
            return
        start_btn.config(state=tk.DISABLED)
        cancel_flag["stop"] = False
        pbar.config(value=0)
        status_var.set("Starting…")

        def worker():
            try:
                imgs_cycle = list(images)
                total = n
                for i in range(n):
                    if cancel_flag["stop"]:
                        break
                    img_path = imgs_cycle[i % len(imgs_cycle)]
                    status_var.set(f"Processing {i+1}/{total}: {img_path.name}")
                    root.update_idletasks()
                    process_single_image(
                        img_path,
                        base_dir,
                        finished_dir,
                        DURATION,
                        FPS,
                        WHITE_THRESH,
                        BLACK_THRESH,
                        EXCLUDE_TOP,
                        EXCLUDE_BOTTOM,
                        CODEC,
                        human_var.get(),
                    )
                    pbar.config(value=((i + 1) / total) * 100)
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                start_btn.config(state=tk.NORMAL)
                status_var.set("Done" if not cancel_flag["stop"] else "Cancelled")

        threading.Thread(target=worker, daemon=True).start()

    start_btn.config(command=on_start)
    close_btn.config(command=lambda: on_close())

    def on_close():
        cancel_flag["stop"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
