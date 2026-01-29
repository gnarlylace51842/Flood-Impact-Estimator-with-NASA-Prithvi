import os
import time
import argparse
import numpy as np
import cv2

def save_binary_mask(mask01, path):
    cv2.imwrite(path, (mask01.astype(np.uint8) * 255))

def load_binary_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (m > 127).astype(np.uint8)

def clean_mask(mask01, min_region_pixels=200, hole_fill_ksize=7):
    mask = (mask01 > 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_fill_ksize, hole_fill_ksize))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    cleaned = np.zeros_like(mask_closed, dtype=np.uint8)

    for comp_id in range(1, n):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area >= min_region_pixels:
            cleaned[labels == comp_id] = 1

    return cleaned

def region_stats(mask01):
    m = (mask01 > 0).astype(np.uint8)
    total = m.size
    fg_pixels = int(m.sum())
    coverage_pct = (fg_pixels / total) * 100.0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    region_count = max(0, n - 1)

    largest = 0
    if region_count > 0:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())

    return coverage_pct, region_count, largest

def coverage_tier(coverage_pct):
    if coverage_pct < 5.0:
        return "Low coverage"
    if coverage_pct <= 20.0:
        return "Medium coverage"
    return "High coverage"

def overlay_mask(image_bgr, mask01, alpha=0.45):
    color = np.array([255, 140, 0], dtype=np.float32)
    img = image_bgr.astype(np.float32).copy()
    m = mask01.astype(bool)
    img[m] = (1 - alpha) * img[m] + alpha * color
    return img.astype(np.uint8)

def make_report(image_bgr, mask_clean01, coverage_pct, region_count, largest_region, tier, out_path):
    overlay = overlay_mask(image_bgr, mask_clean01, alpha=0.45)

    H, W = image_bgr.shape[:2]
    top = np.hstack([image_bgr, overlay])

    text_h = 180
    canvas = np.ones((top.shape[0] + text_h, top.shape[1], 3), dtype=np.uint8) * 255
    canvas[:top.shape[0]] = top

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 16

    cv2.putText(canvas, "Segmentation Report", (pad, top.shape[0] + 32),
                font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    lines = [
        f"Foreground coverage: {coverage_pct:.2f}%",
        f"Region count (connected components): {region_count}",
        f"Largest region size (pixels): {largest_region}",
        f"Coverage tier: {tier}",
    ]
    y0 = top.shape[0] + 70
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (pad, y0 + i * 34),
                    font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Original", (pad, 40), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Overlay (clean mask)", (W + pad, 40), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image (png/jpg)")
    parser.add_argument("--mask", required=True, help="Path to model mask (png). Foreground=white, background=black.")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--min_region_pixels", type=int, default=200, help="Remove regions smaller than this")
    parser.add_argument("--hole_fill_ksize", type=int, default=7, help="Closing kernel size (odd)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)

    mask01 = load_binary_mask(args.mask)

    image_png = os.path.join(args.outdir, "image.png")
    mask_png = os.path.join(args.outdir, "mask.png")
    mask_clean_png = os.path.join(args.outdir, "mask_clean.png")
    report_png = os.path.join(args.outdir, "report.png")

    cv2.imwrite(image_png, image_bgr)
    save_binary_mask(mask01, mask_png)

    mask_clean01 = clean_mask(mask01, min_region_pixels=args.min_region_pixels, hole_fill_ksize=args.hole_fill_ksize)
    save_binary_mask(mask_clean01, mask_clean_png)

    coverage_pct, region_count, largest_region = region_stats(mask_clean01)
    tier = coverage_tier(coverage_pct)

    print(f"Foreground coverage %: {coverage_pct:.2f}")
    print(f"Region count: {region_count}")
    print(f"Largest region size (pixels): {largest_region}")
    print(f"Coverage tier: {tier}")

    make_report(image_bgr, mask_clean01, coverage_pct, region_count, largest_region, tier, report_png)

    print("\nSaved:")
    print(image_png)
    print(mask_png)
    print(mask_clean_png)
    print(report_png)

if __name__ == "__main__":
    main()
