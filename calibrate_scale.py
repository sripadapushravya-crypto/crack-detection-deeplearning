"""
Scale calibration helper for crack measurement validation.

For each "ruler-along-length" image, this script opens the image, lets you
click two points on the ruler that are a known distance apart in mm, and
prints the mm-per-pixel scale for that image.

USAGE:
  python calibrate_scale.py <path_to_ruler_image.jpg>

Then follow the prompts:
  1. The image opens in a matplotlib window
  2. Use the zoom tool (magnifying glass icon) to zoom into the ruler
  3. Press 'q' or close the toolbar to start picking points
  4. Click the FIRST ruler mark (e.g. the 0 cm mark)
  5. Click the SECOND ruler mark (e.g. the 10 cm = 100 mm mark)
  6. The script asks: how many millimetres between those two clicks?
  7. Enter the number (e.g. 100 for 10 cm), press Enter
  8. Script prints the mm-per-pixel scale. Record it next to that crack.

Repeat for all 9 ruler-along-length images.

Tip: pick two points as FAR APART as you can see clearly on the ruler.
The further apart, the more precise the scale.
"""

import sys
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install matplotlib pillow numpy")
    sys.exit(1)


def calibrate(image_path: str) -> float | None:
    """Open image, collect 2 clicks, compute mm/px from user-supplied mm distance."""
    p = Path(image_path)
    if not p.is_file():
        print(f"ERROR: file not found: {p}")
        return None

    img = Image.open(p).convert("RGB")
    arr = np.asarray(img)
    print(f"\nLoaded: {p.name}  size={img.size[0]}x{img.size[1]} px")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(arr)
    ax.set_title(
        f"{p.name}\n"
        "Zoom into the ruler (toolbar), then click TWO points along it.\n"
        "After 2 clicks the window closes automatically."
    )
    ax.axis("on")

    print("\nInstructions:")
    print("  1. Use the zoom/pan tools in the toolbar to zoom into the ruler.")
    print("  2. When ready to click points, make sure no toolbar tool is active.")
    print("  3. Click TWO points on the ruler at known mm positions.")
    print("     (e.g. click the 0 mm mark, then click the 100 mm mark.)")
    print("  4. The window closes automatically after 2 clicks.\n")

    pts = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close(fig)

    if len(pts) != 2:
        print("ERROR: need exactly 2 clicks; aborting.")
        return None

    (x1, y1), (x2, y2) = pts
    px_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    print(f"\nPicked points:")
    print(f"  Point 1: ({x1:.1f}, {y1:.1f})")
    print(f"  Point 2: ({x2:.1f}, {y2:.1f})")
    print(f"  Pixel distance: {px_dist:.2f} px")

    while True:
        raw = input("\nHow many millimetres between those two points? ").strip()
        try:
            mm_dist = float(raw)
            if mm_dist <= 0:
                print("Must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a number (e.g. 100 for 100mm = 10cm).")

    mm_per_px = mm_dist / px_dist
    print(f"\n{'='*55}")
    print(f"SCALE for {p.name}:")
    print(f"  {mm_dist} mm / {px_dist:.2f} px = {mm_per_px:.6f} mm/px")
    print(f"{'='*55}")
    print(f"\nRecord this number next to the crack in your spreadsheet.")
    return mm_per_px


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        try:
            calibrate(arg)
        except Exception as e:
            print(f"ERROR processing {arg}: {e}")
        if len(sys.argv) > 2:
            input("\nPress Enter to continue to next image, or Ctrl+C to stop...")


if __name__ == "__main__":
    main()