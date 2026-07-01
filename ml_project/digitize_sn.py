#!/usr/bin/env python3
"""
Interactive S-N curve digitiser.

Usage:
    python3 digitize_sn.py path/to/page.png

Workflow (six stages, all in one matplotlib window):
    1. Crop  : click two opposite corners of the plot's axis box, press Enter
    2. X-cal : click two points on the x-axis whose data value you know
    3. Y-cal : click two points on the y-axis whose data value you know
    4. Color : click on one marker to sample its colour
    5. Auto  : algorithm finds blob-like regions matching the colour
                and rejects line-shaped components (the fitted curve)
    6. Edit  : left-click empty area to ADD a point, right-click an
                existing point to DELETE it. Press 's' to save.

For each axis you'll be prompted in the terminal for the data values
of your two clicks and whether the axis is linear or log.

Outputs:
    <image>.cal.json   - calibration record (re-runnable)
    <image>.csv        - extracted data points (x, y, x_pixel, y_pixel)
"""
import sys, os, json, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def prompt(msg, cast=str, default=None):
    while True:
        s = input(f"{msg}{' [' + str(default) + ']' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return cast(default) if cast is not bool else default
        try:
            if cast is bool:
                return s.lower() in ("y", "yes", "true", "1", "log")
            return cast(s)
        except Exception as e:
            print(f"  invalid: {e}")


def click_n(ax, n, prompt_text):
    print(f"  >> {prompt_text}  (click {n} point{'s' if n>1 else ''})")
    pts = plt.ginput(n=n, timeout=0, show_clicks=True)
    return pts


def axis_calibrate(p1, p2, v1, v2, log=False):
    """Return a function px -> data value, given two (pixel, data) anchors."""
    if log:
        l1, l2 = np.log10(v1), np.log10(v2)
        m = (l2 - l1) / (p2 - p1)
        b = l1 - m * p1
        return lambda px: 10 ** (m * px + b)
    m = (v2 - v1) / (p2 - p1)
    b = v1 - m * p1
    return lambda px: m * px + b


def extract_markers(crop_bgr, sample_rgb, color_tol=35,
                    min_area=15, max_area=2500, min_circ=0.45):
    """Find blob-like components matching sample colour, reject elongated lines."""
    # work in Lab space — perceptually uniform, robust to JPEG noise
    sample_bgr = sample_rgb[::-1]  # rgb -> bgr
    sample = np.uint8([[sample_bgr]])
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2Lab)[0, 0].astype(int)
    crop_lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab).astype(int)
    dist = np.sqrt(((crop_lab - sample_lab) ** 2).sum(axis=2))
    mask = (dist < color_tol).astype(np.uint8) * 255

    # close small gaps inside markers
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = []
    for i in range(1, n_lbl):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        # circularity = 4πA / P²; a square ~0.79, line ~0.05
        comp_mask = (lbl == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        peri = cv2.arcLength(contours[0], True)
        if peri == 0:
            continue
        circ = 4 * np.pi * area / (peri ** 2)
        ar = max(w, h) / max(min(w, h), 1)
        if circ < min_circ and ar > 2.5:
            continue  # line-like
        cx, cy = cents[i]
        keep.append((cx, cy, area, circ))
    return keep, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--reuse-cal", action="store_true",
                    help="reload calibration json and skip steps 1-3")
    args = ap.parse_args()

    img_path = args.image
    base = os.path.splitext(img_path)[0]
    cal_path = base + ".cal.json"
    csv_path = base + ".csv"

    bgr = cv2.imread(img_path)
    if bgr is None:
        sys.exit(f"could not read {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cal = {}
    if args.reuse_cal and os.path.exists(cal_path):
        cal = json.load(open(cal_path))
        print(f"loaded calibration: {cal_path}")

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(rgb)
    ax.set_title(os.path.basename(img_path))
    plt.tight_layout()
    plt.show(block=False)

    # ---- Crop ----
    if "crop" not in cal:
        pts = click_n(ax, 2, "Crop: click two opposite corners of the axis box")
        x0, x1 = sorted(int(p[0]) for p in pts)
        y0, y1 = sorted(int(p[1]) for p in pts)
        cal["crop"] = [x0, y0, x1, y1]
    x0, y0, x1, y1 = cal["crop"]
    crop = bgr[y0:y1, x0:x1]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    ax.clear(); ax.imshow(crop_rgb); ax.set_title("cropped plot region")
    fig.canvas.draw()

    # ---- X axis ----
    if "x_cal" not in cal:
        pts = click_n(ax, 2, "X-axis: click two known reference points (e.g., gridlines or tick marks)")
        v1 = prompt("  data value at the FIRST x click", float)
        v2 = prompt("  data value at the SECOND x click", float)
        log = prompt("  is X-axis logarithmic? (y/n)", bool, default="y")
        cal["x_cal"] = {"p1": pts[0][0], "p2": pts[1][0], "v1": v1, "v2": v2, "log": log}
    xc = cal["x_cal"]
    x_fn = axis_calibrate(xc["p1"], xc["p2"], xc["v1"], xc["v2"], xc["log"])

    # ---- Y axis ----
    if "y_cal" not in cal:
        pts = click_n(ax, 2, "Y-axis: click two known reference points")
        v1 = prompt("  data value at the FIRST y click", float)
        v2 = prompt("  data value at the SECOND y click", float)
        log = prompt("  is Y-axis logarithmic? (y/n)", bool, default="n")
        cal["y_cal"] = {"p1": pts[0][1], "p2": pts[1][1], "v1": v1, "v2": v2, "log": log}
    yc = cal["y_cal"]
    y_fn = axis_calibrate(yc["p1"], yc["p2"], yc["v1"], yc["v2"], yc["log"])

    json.dump(cal, open(cal_path, "w"), indent=2)
    print(f"calibration saved -> {cal_path}")

    # ---- Colour sample ----
    pts = click_n(ax, 1, "Click ON a marker to sample its colour")
    cx, cy = int(pts[0][0]), int(pts[0][1])
    # average a 3x3 around the click
    patch = crop_rgb[max(0,cy-1):cy+2, max(0,cx-1):cx+2].reshape(-1, 3)
    sample_rgb = patch.mean(axis=0).astype(int)
    print(f"  sampled colour RGB={tuple(sample_rgb)}")

    # ---- Extract ----
    color_tol = 40
    min_circ = 0.45
    while True:
        keep, mask = extract_markers(crop, sample_rgb,
                                      color_tol=color_tol, min_circ=min_circ)
        ax.clear()
        ax.imshow(crop_rgb)
        if keep:
            xs = [k[0] for k in keep]; ys = [k[1] for k in keep]
            ax.scatter(xs, ys, s=80, facecolors="none", edgecolors="lime", lw=1.5)
        ax.set_title(f"{len(keep)} points  |  tol={color_tol}  circ≥{min_circ:.2f}  "
                     f"(t/T tolerance ±, c/C circ ±, e=edit, s=save, q=quit)")
        fig.canvas.draw()

        cmd = input("[t]ol- [T]ol+ [c]irc- [C]irc+ [e]dit [s]ave [q]uit > ").strip().lower()
        if cmd == "t": color_tol = max(5, color_tol - 5)
        elif cmd in ("T", "t+"): color_tol += 5
        elif cmd == "c": min_circ = max(0.1, min_circ - 0.05)
        elif cmd in ("C", "c+"): min_circ = min(0.95, min_circ + 0.05)
        elif cmd == "e":
            print("  Edit mode — left-click to ADD, right-click on a green circle to DELETE; press Enter when done.")
            added, removed = [], set()
            def onclick(ev):
                if ev.inaxes != ax: return
                if ev.button == 1:
                    added.append((ev.xdata, ev.ydata))
                    ax.scatter([ev.xdata], [ev.ydata], s=80, facecolors="none",
                               edgecolors="cyan", lw=1.5)
                elif ev.button == 3:
                    # nearest existing keep point
                    if not keep: return
                    pxs = np.array([(k[0], k[1]) for k in keep])
                    d = np.hypot(pxs[:, 0] - ev.xdata, pxs[:, 1] - ev.ydata)
                    j = int(np.argmin(d))
                    if d[j] < 25:
                        removed.add(j)
                        ax.scatter([keep[j][0]], [keep[j][1]], s=120, marker="x",
                                   color="red")
                fig.canvas.draw()
            cid = fig.canvas.mpl_connect("button_press_event", onclick)
            input("  press Enter when done editing")
            fig.canvas.mpl_disconnect(cid)
            keep = [k for j, k in enumerate(keep) if j not in removed]
            keep += [(a[0], a[1], 0, 0) for a in added]
        elif cmd == "s":
            break
        elif cmd == "q":
            print("aborted, no CSV written"); return

    # ---- Convert to data coords and save ----
    rows = []
    for cx, cy, area, circ in keep:
        # cx/cy are inside the crop; convert back to original image coords for axis fns
        px_orig = cx + cal["crop"][0]
        py_orig = cy + cal["crop"][1]
        x_data = float(x_fn(px_orig))
        y_data = float(y_fn(py_orig))
        rows.append((x_data, y_data, px_orig, py_orig, area, circ))

    rows.sort()
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "x_pixel", "y_pixel", "area_px", "circularity"])
        w.writerows(rows)
    print(f"\nsaved {len(rows)} points -> {csv_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
