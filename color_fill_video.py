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
from dataclasses import dataclass


# Optional OCR support: if pytesseract is available, we'll leverage it to
# locate the "current color" label. Otherwise, we fall back to circle search.
try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore


@dataclass
class Circle:
    x: int
    y: int
    r: int


def _clip_rect(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _find_circles_hough(img_bgr: np.ndarray, min_r: int = 6, max_r: int | None = None) -> list[Circle]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    if max_r is None:
        max_r = max(10, min(h, w) // 6)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min(h, w) // 20),
        param1=120,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    found: list[Circle] = []
    if circles is not None:
        for c in circles[0, :]:
            cx, cy, r = int(round(c[0])), int(round(c[1])), int(round(c[2]))
            found.append(Circle(cx, cy, r))
    return found


def _score_circle_by_saturation(img_bgr: np.ndarray, circ: Circle) -> float:
    # Mean saturation inside the circle; higher favors vivid swatches
    x, y, r = circ.x, circ.y, circ.r
    h, w = img_bgr.shape[:2]
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(w, x + r), min(h, y + r)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return -1.0
    yy, xx = np.ogrid[0:roi.shape[0], 0:roi.shape[1]]
    mask = (xx - (x - x0)) ** 2 + (yy - (y - y0)) ** 2 <= r * r
    if not np.any(mask):
        return -1.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    return float(np.mean(sat[mask]))


def _detect_circle_near_text_with_ocr(img_bgr: np.ndarray) -> Circle | None:
    if pytesseract is None:
        return None
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, output_type='dict')  # type: ignore
    except Exception:
        return None
    n = len(data.get('text', []))
    # Build words with positions
    words = []
    for i in range(n):
        t = data['text'][i]
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1.0
        if not t or t.isspace() or conf < 0:
            continue
        x, y, w, h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
        words.append((t, x, y, w, h))
    # Search for sequence "current" + "color" (case-insensitive)
    target_rect = None
    for i in range(len(words) - 1):
        w1, x1, y1, ww1, hh1 = words[i]
        w2, x2, y2, ww2, hh2 = words[i + 1]
        if w1.lower() == 'current' and w2.lower() == 'color':
            x0 = min(x1, x2)
            y0 = min(y1, y2)
            x1b = max(x1 + ww1, x2 + ww2)
            y1b = max(y1 + hh1, y2 + hh2)
            target_rect = (x0, y0, x1b, y1b)
            break
    if target_rect is None:
        # Also consider the single token "currentcolor" if OCR fused it
        for t, x, y, w, h in words:
            if t.lower().replace(" ", "") == "currentcolor":
                target_rect = (x, y, x + w, y + h)
                break
    if target_rect is None:
        return None
    h_img, w_img = img_bgr.shape[:2]
    x0, y0, x1, y1 = _clip_rect(*target_rect, w_img, h_img)
    rect_h = max(1, y1 - y0)
    # Probe to the right first
    search_w = rect_h * 6
    rx0, ry0, rx1, ry1 = _clip_rect(x1, y0 - rect_h, x1 + search_w, y1 + rect_h * 2, w_img, h_img)
    probe_right = img_bgr[ry0:ry1, rx0:rx1]
    candidates = _find_circles_hough(probe_right)
    if candidates:
        # Adjust to full-image coords and pick max-saturation
        adjusted = [Circle(c.x + rx0, c.y + ry0, c.r) for c in candidates]
        best = max(adjusted, key=lambda c: _score_circle_by_saturation(img_bgr, c))
        return best
    # Probe to the left as fallback
    lx0, ly0, lx1, ly1 = _clip_rect(x0 - search_w, y0 - rect_h, x0, y1 + rect_h * 2, w_img, h_img)
    probe_left = img_bgr[ly0:ly1, lx0:lx1]
    candidates = _find_circles_hough(probe_left)
    if candidates:
        adjusted = [Circle(c.x + lx0, c.y + ly0, c.r) for c in candidates]
        best = max(adjusted, key=lambda c: _score_circle_by_saturation(img_bgr, c))
        return best
    return None


def _detect_ui_color_circle(
    img_bgr: np.ndarray,
    exclude_top_px: int = 0,
    exclude_bottom_px: int = 0,
) -> Circle | None:
    # Prefer OCR-based detection
    circ = _detect_circle_near_text_with_ocr(img_bgr)
    if circ is not None:
        return circ
    # Fallback: search top/bottom excluded bands for vivid circles
    h, w = img_bgr.shape[:2]
    bands: list[tuple[int, int, int, int]] = []
    if exclude_top_px > 0:
        bands.append((0, 0, w, min(h, exclude_top_px)))
    if exclude_bottom_px > 0:
        bands.append((0, max(0, h - exclude_bottom_px), w, h))
    best: Circle | None = None
    best_score = -1.0
    for (x0, y0, x1, y1) in bands:
        roi = img_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        for c in _find_circles_hough(roi):
            adj = Circle(c.x + x0, c.y + y0, c.r)
            score = _score_circle_by_saturation(img_bgr, adj)
            if score > best_score:
                best_score = score
                best = adj
    return best


def _draw_ui_circle(
    frame_bgr: np.ndarray,
    circ: Circle,
    color_bgr: tuple[int, int, int],
    inflate_px: int = 3,
    inflate_scale: float = 1.12,
) -> None:
    # Fill the circle with the provided color, slightly inflated to hide any background fringe.
    if circ is None:
        return
    h, w = frame_bgr.shape[:2]
    if circ.x < 0 or circ.y < 0 or circ.x >= w or circ.y >= h or circ.r <= 0:
        return
    r = int(round(max(circ.r * float(inflate_scale), circ.r + float(inflate_px))))
    r = max(1, r)
    cv2.circle(
        frame_bgr,
        (int(circ.x), int(circ.y)),
        r,
        color_bgr,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


def _find_circle_near_point(
    img_bgr: np.ndarray,
    pt: tuple[int, int],
    search_radius: int = 120,
) -> Circle | None:
    x, y = int(pt[0]), int(pt[1])
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = _clip_rect(x - search_radius, y - search_radius, x + search_radius, y + search_radius, w, h)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    cands = _find_circles_hough(roi)
    if not cands:
        return None
    # Choose the candidate whose center is nearest to the requested point
    best = min(cands, key=lambda c: (c.x + x0 - x) ** 2 + (c.y + y0 - y) ** 2)
    return Circle(best.x + x0, best.y + y0, best.r)


def _resolve_ui_circle(
    img_bgr: np.ndarray,
    exclude_top_px: int = 0,
    exclude_bottom_px: int = 0,
    override_pos: tuple[int, int] | tuple[float, float] | None = None,
    pos_is_normalized: bool = False,
    search_radius: int | None = None,
) -> Circle | None:
    h, w = img_bgr.shape[:2]
    if override_pos is not None:
        if pos_is_normalized:
            x = int(round(float(override_pos[0]) * w))
            y = int(round(float(override_pos[1]) * h))
        else:
            x = int(round(float(override_pos[0])))
            y = int(round(float(override_pos[1])))
        sr = int(search_radius if search_radius is not None else max(40, min(h, w) // 8))
        circ = _find_circle_near_point(img_bgr, (x, y), search_radius=sr)
        if circ is not None:
            return circ
        # Fallback: assume a sensible radius if Hough misses
        r_guess = max(8, min(h, w) // 30)
        return Circle(x, y, r_guess)
    # No override -> use automatic detection
    return _detect_ui_color_circle(img_bgr, exclude_top_px=exclude_top_px, exclude_bottom_px=exclude_bottom_px)


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
    exclude_rect: tuple[int, int, int, int] | None = None,
):
    """Compute masks for white fillable regions and linework (black).

    Returns (white_mask, line_mask)
    - white_mask: binary uint8 where white fillable pixels are 255
    - line_mask: binary uint8 where black linework pixels are 255
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Black lines (and black shapes) mask
    # Treat more near-black as line to protect borders
    _, line_mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    # Slightly dilate line mask so coloring never touches the outlines
    kernel_line = np.ones((3, 3), np.uint8)
    line_mask = cv2.dilate(line_mask, kernel_line, iterations=1)

    # White areas mask – accept "near white" as fillable
    _, white_mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)

    # Remove line pixels from fillable mask
    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(line_mask))

    # Close tiny gaps and hug outlines to avoid unfilled fringes near lines
    kernel = np.ones((3, 3), np.uint8)
    # Close small gaps so region masks hug outlines better (but keep off lines)
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
    # Exclude an arbitrary rectangle if provided (x0,y0,x1,y1)
    if exclude_rect is not None:
        x0, y0, x1, y1 = exclude_rect
        H, W = white_mask.shape[:2]
        x0 = max(0, min(int(x0), W))
        x1 = max(0, min(int(x1), W))
        y0 = max(0, min(int(y0), H))
        y1 = max(0, min(int(y1), H))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 > x0 and y1 > y0:
            white_mask[y0:y1, x0:x1] = 0
    return white_mask, line_mask


def fill_regions_progressively(
    img_bgr: np.ndarray,
    duration: float = 20.0,
    fps: int = 30,
    seed: int | None = None,
    white_thresh: int = 240,
    black_thresh: int = 50,
    exclude_top_px: int = 500,
    exclude_bottom_px: int = 400,
    codec: str = "mp4v",
    output_path: str = "output.mp4",
    human_like: bool = False,
    min_region_area: int = 100,
    tail_secs: float = 7.0,
    max_colors: int | None = None,
    accelerate: bool = False,
    ui_circle_pos: tuple[int, int] | tuple[float, float] | None = None,
    ui_circle_pos_is_normalized: bool = False,
    ui_circle_search_radius: int | None = None,
    exclude_rect: tuple[int, int, int, int] | None = None,
    pattern: str | None = None,
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
        exclude_rect=exclude_rect,
    )

    # Try to detect the UI "current color" circle once so we can update it per frame
    ui_circle = _resolve_ui_circle(
        img_bgr,
        exclude_top_px=exclude_top_px,
        exclude_bottom_px=exclude_bottom_px,
        override_pos=ui_circle_pos,
        pos_is_normalized=ui_circle_pos_is_normalized,
        search_radius=ui_circle_search_radius,
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
    if human_like and region_ids and (pattern is None or str(pattern).lower() in ("", "auto")):
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

    # Apply explicit pattern ordering if requested (overrides the above ordering)
    if pattern is not None and str(pattern).lower() not in ("", "auto") and region_ids:
        mode = str(pattern).lower().strip()
        h_img, w_img = img_bgr.shape[:2]
        cy, cx = h_img / 2.0, w_img / 2.0
        # Use provided component centroids
        cents_map = {rid: (float(cents[rid][1]), float(cents[rid][0])) for rid in region_ids}  # (y,x)
        def key_top_to_bottom(rid: int):
            y, x = cents_map[rid]
            return (y, x)
        def key_left_to_right(rid: int):
            y, x = cents_map[rid]
            return (x, y)
        def key_center_out(rid: int):
            y, x = cents_map[rid]
            return (math.hypot(y - cy, x - cx), y, x)
        def key_angle_sweep(rid: int):
            y, x = cents_map[rid]
            ang = math.atan2(y - cy, x - cx)
            return (ang, math.hypot(y - cy, x - cx))
        if mode in ("top-to-bottom", "top_to_bottom", "vertical"):
            region_ids = sorted(region_ids, key=key_top_to_bottom)
        elif mode in ("left-to-right", "left_to_right", "horizontal"):
            region_ids = sorted(region_ids, key=key_left_to_right)
        elif mode in ("center-out", "center_out", "radial"):
            region_ids = sorted(region_ids, key=key_center_out)
        elif mode in ("angle", "angle-sweep", "spiral", "cw"):
            region_ids = sorted(region_ids, key=key_angle_sweep)
        elif mode in ("special",):
            # Alternate near/far from center for a dynamic visual
            ordered = sorted(region_ids, key=key_center_out)
            i, j = 0, len(ordered) - 1
            woven = []
            while i <= j:
                if i == j:
                    woven.append(ordered[i])
                else:
                    woven.append(ordered[i])
                    woven.append(ordered[j])
                i += 1
                j -= 1
            region_ids = woven

    # Decide how many regions to color per emitted frame
    # Optionally accelerate: start with fewer regions per frame, then increase.
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
        latest_color_bgr: Tuple[int, int, int] | None = None
        # Preserve the original top/bottom bands exactly so UI chrome remains untouched
        if exclude_top_px > 0:
            top_preserve_h = min(exclude_top_px, h)
            original_top = img_bgr[:top_preserve_h, :].copy()
        else:
            top_preserve_h = 0
            original_top = None
        if exclude_bottom_px > 0:
            bottom_preserve_h = min(exclude_bottom_px, h)
            original_bottom = img_bgr[h - bottom_preserve_h :, :].copy()
        else:
            bottom_preserve_h = 0
            original_bottom = None
        last_emitted_frame: np.ndarray | None = None
        colored = np.zeros_like(white_mask, dtype=bool)

        # Conservative dilation to hug outlines without crossing
        kernel_dilate = np.ones((2, 2), np.uint8)
        dilate_iter = 1
        # Decide the set of colors allowed for this run (limit by max_colors)
        # For non-human mode, use a random subset of PALETTE_BGR.
        # For human-like mode, derive an ordered palette then reduce to unique up to the limit.
        limit = None
        if max_colors is None:
            limit = None
        else:
            try:
                limit = int(max_colors)
            except Exception:
                limit = None
        if limit is not None:
            limit = max(1, min(limit, len(PALETTE_BGR)))

        if human_like:
            # Build a pleasant sequence first, then keep unique colors up to the limit
            seq = generate_human_palette(rng, n=max(16, (limit or 16)))
            uniq: List[Tuple[int, int, int]] = []
            for c in seq:
                if c not in uniq:
                    uniq.append(c)
                if limit is not None and len(uniq) >= limit:
                    break
            if not uniq:
                uniq = [random_vivid_bgr(rng)]
            palette = uniq
            allowed_colors = palette
        else:
            if limit is None or limit >= len(PALETTE_BGR):
                allowed_colors = list(PALETTE_BGR)
            else:
                # Choose a stable random subset for this run
                allowed_colors = rng.sample(PALETTE_BGR, k=limit)
            palette = None
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
                if window.size == 0:
                    neighbor_map[rid] = set()
                    continue
                m = (window == rid).astype(np.uint8) * 255
                if m.size == 0:
                    neighbor_map[rid] = set()
                    continue
                md = cv2.dilate(m, k3, iterations=1)
                ring = (md > 0) & (m == 0)
                neigh = set(np.unique(window[ring]).tolist())
                neigh.discard(0)
                neigh.discard(rid)
                neighbor_map[rid] = neigh

        assigned_color: dict[int, Tuple[int, int, int]] = {}

        # Build an accelerating schedule if requested
        chunks: List[int] | None = None
        if accelerate:
            total_regions = len(region_ids)
            tail_frames_expected = int(round(tail_secs * fps))
            color_frames_target = max(1, total_frames_target - tail_frames_expected)
            # Ensure acceleration is visible even when regions < color frames:
            # compress the number of coloring frames to ~60% of regions (but cap by target frames).
            frames_for_coloring = max(1, min(color_frames_target, int(max(1, round(total_regions * 0.6)))))
            # Weight later frames heavier to accelerate; alpha controls curvature
            alpha = 2.0
            weights = [(k + 1) ** alpha for k in range(frames_for_coloring)]
            sw = sum(weights)
            raw = [total_regions * w / sw for w in weights]
            ints = [max(1, int(math.floor(x))) for x in raw]
            diff = total_regions - sum(ints)
            if diff > 0:
                # Distribute remaining regions to frames with largest fractional parts
                fracs = [x - math.floor(x) for x in raw]
                order = sorted(range(frames_for_coloring), key=lambda i: fracs[i], reverse=True)
                for idx in order:
                    if diff <= 0:
                        break
                    ints[idx] += 1
                    diff -= 1
            elif diff < 0:
                # Remove extras from earliest frames while keeping at least 1
                need = -diff
                for idx in range(frames_for_coloring):
                    if need <= 0:
                        break
                    take = min(need, max(0, ints[idx] - 1))
                    ints[idx] -= take
                    need -= take
                # If still need to trim, remove from the end
                idx = frames_for_coloring - 1
                while need > 0 and idx >= 0:
                    take = min(need, max(0, ints[idx] - 1))
                    ints[idx] -= take
                    need -= take
                    idx -= 1
            # Ensure non-decreasing sequence
            for k in range(1, frames_for_coloring):
                if ints[k] < ints[k - 1]:
                    inc = ints[k - 1] - ints[k]
                    ints[k] += inc
            # Fix sum if grew due to monotonic enforcement
            over = sum(ints) - total_regions
            idx = frames_for_coloring - 1
            while over > 0 and idx >= 0:
                reducible = max(0, ints[idx] - 1)
                take = min(over, reducible)
                ints[idx] -= take
                over -= take
                idx -= 1
            chunks = ints

        # Iterate regions and emit frames according to schedule
        colored_in_chunk = 0
        next_chunk_target = chunks[0] if accelerate and chunks else regions_per_frame
        chunk_index = 0

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
                # Pick from the limited allowed palette
                color = rng.choice(allowed_colors)
            # Build a robust per-region mask on a local ROI and fill tiny holes
            rx, ry, rw, rh = (
                int(stats[region_id, cv2.CC_STAT_LEFT]),
                int(stats[region_id, cv2.CC_STAT_TOP]),
                int(stats[region_id, cv2.CC_STAT_WIDTH]),
                int(stats[region_id, cv2.CC_STAT_HEIGHT]),
            )
            pad = 3
            y0 = max(0, ry - pad)
            x0 = max(0, rx - pad)
            y1 = min(h, ry + rh + pad)
            x1 = min(w, rx + rw + pad)
            if y1 <= y0 or x1 <= x0:
                # Fallback to whole-image path for safety
                mask = labels == region_id
                m8 = (mask.astype(np.uint8) * 255)
                if m8.size > 0:
                    m8 = cv2.dilate(m8, kernel_dilate, iterations=dilate_iter)
                mask = (m8 > 0) & (~line_mask.astype(bool))
                colored |= mask
                current[mask] = color
            else:
                roi_labels = labels[y0:y1, x0:x1]
                roi_line = line_mask[y0:y1, x0:x1]
                base = (roi_labels == region_id)
                if base.size == 0:
                    # Fallback
                    mask = labels == region_id
                    m8 = (mask.astype(np.uint8) * 255)
                    if m8.size > 0:
                        m8 = cv2.dilate(m8, kernel_dilate, iterations=dilate_iter)
                        mask = (m8 > 0) & (~line_mask.astype(bool))
                        colored |= mask
                        current[mask] = color
                    else:
                        # Nothing to do
                        pass
                else:
                    m8 = (base.astype(np.uint8) * 255)
                    if m8.size > 0:
                        # Slightly dilate to hug outlines
                        m8 = cv2.dilate(m8, kernel_dilate, iterations=dilate_iter)
                        # Fill small internal holes within the region ROI
                        ff = m8.copy()
                        ff_mask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
                        # Ensure seed is background by forcing ROI corners to 0
                        if ff.shape[0] > 0 and ff.shape[1] > 0:
                            ff[0, 0] = 0
                            ff[0, -1] = 0
                            ff[-1, 0] = 0
                            ff[-1, -1] = 0
                        cv2.floodFill(ff, ff_mask, (0, 0), 255)
                        inv = cv2.bitwise_not(ff)  # pixels not reachable from border -> holes
                        filled = cv2.bitwise_or(m8, inv)
                    else:
                        filled = m8
                    roi_mask = (filled > 0) & (~roi_line.astype(bool))
                    # Apply to current frame
                    sub = current[y0:y1, x0:x1]
                    sub[roi_mask] = color
                    current[y0:y1, x0:x1] = sub
                    # Track colored map (optional)
                    colored[y0:y1, x0:x1] |= roi_mask
            assigned_color[region_id] = color
            latest_color_bgr = color

            # Emit frames based on accelerating schedule or fixed cadence
            colored_in_chunk += 1
            if colored_in_chunk >= next_chunk_target:
                frame = current.copy()
                # Reimpose linework to keep outlines crisp
                frame[line_mask.astype(bool)] = (0, 0, 0)
                # Restore top/bottom bands exactly as original to avoid any changes there
                if top_preserve_h > 0 and original_top is not None:
                    top_h = min(top_preserve_h, frame.shape[0], original_top.shape[0])
                    if top_h > 0:
                        frame[:top_h, :] = original_top[:top_h, :]
                if bottom_preserve_h > 0 and original_bottom is not None:
                    bottom_h = min(bottom_preserve_h, frame.shape[0], original_bottom.shape[0])
                    if bottom_h > 0:
                        frame[-bottom_h:, :] = original_bottom[-bottom_h:, :]
                # Finally, draw the UI circle with the current color so it updates
                # even if it lives inside the preserved UI bands.
                if ui_circle is not None and latest_color_bgr is not None:
                    _draw_ui_circle(frame, ui_circle, latest_color_bgr)
                writer.write(frame)
                frames_written += 1
                last_emitted_frame = frame
                colored_in_chunk = 0
                if accelerate and chunks is not None:
                    chunk_index += 1
                    if chunk_index < len(chunks):
                        next_chunk_target = chunks[chunk_index]
                # else keep using regions_per_frame

        # After finishing all regions, hold the final image for tail_secs seconds
        # (composition may place splash over a portion of this tail)
        final_frame = current.copy()
        final_frame[line_mask.astype(bool)] = (0, 0, 0)
        tail_frames = int(round(tail_secs * fps))
        for _ in range(tail_frames):
            # Keep showing last used color in the UI circle during the tail
            frame_tail = final_frame.copy()
            # Restore top/bottom bands exactly as original
            if top_preserve_h > 0 and original_top is not None:
                top_h = min(top_preserve_h, frame_tail.shape[0], original_top.shape[0])
                if top_h > 0:
                    frame_tail[:top_h, :] = original_top[:top_h, :]
            if bottom_preserve_h > 0 and original_bottom is not None:
                bottom_h = min(bottom_preserve_h, frame_tail.shape[0], original_bottom.shape[0])
                if bottom_h > 0:
                    frame_tail[-bottom_h:, :] = original_bottom[-bottom_h:, :]
            # Then overlay the current-color circle so it reflects the last color
            if ui_circle is not None and latest_color_bgr is not None:
                _draw_ui_circle(frame_tail, ui_circle, latest_color_bgr)
            writer.write(frame_tail)
            frames_written += 1
            last_emitted_frame = frame_tail
    finally:
        writer.release()
    if last_emitted_frame is None:
        last_emitted_frame = final_frame
    return frames_written, fps, last_emitted_frame


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


def _sanitize_filename_component(s: str) -> str:
    """Return a version of `s` that is safe for Windows filenames.

    Replaces illegal characters with '-'. Ensures no trailing space/dot.
    Keeps readability (spaces, parentheses, dashes, equals are allowed).
    """
    # Windows disallows: < > : " / \ | ? * and control chars
    forbidden = set('<>:"/\\|?*')
    cleaned = []
    for ch in s:
        if ord(ch) < 32 or ch in forbidden:
            cleaned.append('-')
        else:
            cleaned.append(ch)
    out = ''.join(cleaned).strip()
    # Avoid trailing dot/space which can cause issues on Windows
    out = out.rstrip(' .')
    return out


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
    MAX_COLORS: int,
    ACCELERATE: bool,
    PATTERN: str,
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

    # Override UI circle location: user-provided absolute coordinates for the swatch
    UI_CIRCLE_POS_ABS: tuple[int, int] | None = (980, 590)

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
        max_colors=MAX_COLORS,
        accelerate=ACCELERATE,
        ui_circle_pos=UI_CIRCLE_POS_ABS,
        ui_circle_pos_is_normalized=False,
        # Exclude the top-left rectangle: x<925 and y<650
        exclude_rect=(0, 0, 925, 650),
        pattern=PATTERN,
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
            audio = audio.fx(afx.volumex, 0.1)
            final_clip = final_clip.set_audio(audio)
        except Exception:
            pass

    # Include mode details in the output filename, sanitized for Windows
    pretty_pattern = (
        str(PATTERN).strip() if PATTERN is not None and str(PATTERN).strip() else "Off"
    )
    mode_text = (
        f"(human-like | accel={'on' if ACCELERATE else 'off'} | pattern={pretty_pattern} | colors={MAX_COLORS})"
        if HUMAN_LIKE
        else f"(true-random | accel={'on' if ACCELERATE else 'off'} | pattern={pretty_pattern} | colors={MAX_COLORS})"
    )
    # Replace illegal characters (notably '|' on Windows) while keeping readability
    safe_mode = _sanitize_filename_component(mode_text)
    stem_with_mode = f"{input_path.stem} {safe_mode}" if safe_mode else input_path.stem
    output_path = next_unique_path(finished_dir, stem_with_mode, ".mp4")
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
    # Increase tolerance for near-white edge pixels and near-black anti-aliased lines
    WHITE_THRESH = 230
    BLACK_THRESH = 80
    EXCLUDE_TOP = 500
    EXCLUDE_BOTTOM = 400
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
    # Make the default window larger so resizing isn't needed each run
    root.geometry("800x640")
    try:
        root.minsize(720, 560)
    except Exception:
        pass

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

    # Pattern option
    pattern_frame = ttk.Frame(frm)
    pattern_frame.pack(fill=tk.X, pady=(12, 0))
    ttk.Label(pattern_frame, text="Coloring pattern:").pack(anchor="w")
    pattern_enable_var = tk.BooleanVar(value=False)
    pattern_enable_chk = ttk.Checkbutton(pattern_frame, text="Enable pattern", variable=pattern_enable_var)
    pattern_enable_chk.pack(anchor="w")
    pattern_var = tk.StringVar(value="Auto")
    cmb = ttk.Combobox(pattern_frame, textvariable=pattern_var, state="disabled")
    cmb['values'] = ("Auto", "Top-to-bottom", "Left-to-right", "Center-out", "Angle sweep", "Special")
    cmb.current(0)
    cmb.pack(fill=tk.X)

    def on_toggle_pattern_enable():
        cmb.config(state=("readonly" if pattern_enable_var.get() else "disabled"))
    pattern_enable_chk.config(command=on_toggle_pattern_enable)

    # Acceleration option
    accel_frame = ttk.Frame(frm)
    accel_frame.pack(fill=tk.X, pady=(8, 0))
    accelerate_var = tk.BooleanVar(value=False)
    accel_chk = ttk.Checkbutton(accel_frame, text="Start slow, then accelerate", variable=accelerate_var)
    accel_chk.pack(anchor="w")

    # Randomly choose per video (affects human-like, accelerate, pattern, max colors)
    random_per_video_var = tk.BooleanVar(value=False)
    def on_toggle_random_per_video():
        state = tk.DISABLED if random_per_video_var.get() else tk.NORMAL
        # Grey/disable inputs when random-per-video is enabled
        chk.config(state=state)
        accel_chk.config(state=state)
        pattern_enable_chk.config(state=state)
        cmb.config(state=("disabled" if random_per_video_var.get() else ("readonly" if pattern_enable_var.get() else "disabled")))
    rand_all_frame = ttk.Frame(frm)
    rand_all_frame.pack(fill=tk.X, pady=(12, 0))
    random_chk2 = ttk.Checkbutton(
        rand_all_frame,
        text="Randomly choose mode per video",
        variable=random_per_video_var,
        command=on_toggle_random_per_video,
    )
    random_chk2.pack(anchor="w")

    # Max distinct colors per video slider
    colors_frame = ttk.Frame(frm)
    colors_frame.pack(fill=tk.X, pady=(12, 0))
    ttk.Label(colors_frame, text="Max different colors per video:").pack(anchor="w")
    colors_row = ttk.Frame(colors_frame)
    colors_row.pack(fill=tk.X)
    max_colors_var = tk.IntVar(value=len(PALETTE_BGR))
    scale_var = tk.DoubleVar(value=float(len(PALETTE_BGR)))

    lbl_val = ttk.Label(colors_row, text=str(len(PALETTE_BGR)))

    def on_colors_scale(v):
        try:
            val = int(round(float(v)))
        except Exception:
            val = len(PALETTE_BGR)
        val = max(1, min(val, len(PALETTE_BGR)))
        max_colors_var.set(val)
        lbl_val.config(text=str(val))

    scl = ttk.Scale(
        colors_row,
        from_=1,
        to=len(PALETTE_BGR),
        orient=tk.HORIZONTAL,
        variable=scale_var,
        command=on_colors_scale,
    )
    scl.pack(side=tk.LEFT, expand=True, fill=tk.X)
    lbl_val.pack(side=tk.LEFT, padx=(8, 0))

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
                # Shuffle the images so selection is random per batch
                img_rng = random.SystemRandom()
                if len(imgs_cycle) > 1:
                    imgs_cycle = img_rng.sample(imgs_cycle, len(imgs_cycle))
                total = n
                # Use a cryptographically-strong RNG for random-per-video selections
                mode_rng = random.SystemRandom()
                for i in range(n):
                    if cancel_flag["stop"]:
                        break
                    img_path = imgs_cycle[i % len(imgs_cycle)]
                    # Decide options for this video
                    if random_per_video_var.get():
                        current_human_like = bool(mode_rng.getrandbits(1))
                        current_accelerate = bool(mode_rng.getrandbits(1))
                        # Pattern enabled?
                        use_pattern = bool(mode_rng.getrandbits(1))
                        # Choose a pattern if enabled
                        patterns = [
                            "Top-to-bottom",
                            "Left-to-right",
                            "Center-out",
                            "Angle sweep",
                            "Special",
                        ]
                        current_pattern = mode_rng.choice(patterns) if use_pattern else None
                        # Randomize max colors between 3 and full palette
                        current_max_colors = mode_rng.randrange(3, len(PALETTE_BGR) + 1)
                    else:
                        current_human_like = bool(human_var.get())
                        current_accelerate = bool(accelerate_var.get())
                        current_pattern = pattern_var.get() if bool(pattern_enable_var.get()) else None
                        current_max_colors = int(max_colors_var.get())

                    pretty_pattern = current_pattern if current_pattern is not None else ("Auto" if bool(pattern_enable_var.get()) else "Off")
                    mode_label = (
                        f"{'human-like' if current_human_like else 'true random'} | accel={'on' if current_accelerate else 'off'} | pattern={pretty_pattern} | colors={current_max_colors}"
                    )
                    status_var.set(f"Processing {i+1}/{total}: {img_path.name} ({mode_label})")
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
                        current_human_like,
                        int(current_max_colors),
                        bool(current_accelerate),
                        current_pattern if current_pattern is not None else (pattern_var.get() if bool(pattern_enable_var.get()) else None),
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
