Color fill video generator

What it does
- Finds all white regions in a black–outline coloring page and fills them with random, vivid colors.
- Produces a 20-second MP4 animation of the coloring process (configurable).

Quick start
1) Place your input image (PNG/JPG) in this folder. Prefer a black-and-white line drawing where fillable areas are pure white and outlines are black.
2) Install dependencies (Python 3.9+):
   pip install -r requirements.txt
3) Run:
   python color_fill_video.py --input your_image.png --output out.mp4 --duration 20 --fps 30

Notes
- If `mp4v` codec fails on your system, try a different one:
  python color_fill_video.py --input your_image.png --codec MJPG --output out.avi
- Tweak thresholds if needed:
  --white-thresh 245   (higher = only very white areas get colored)
  --black-thresh 50    (lower = more pixels treated as linework)

