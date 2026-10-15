"""
main.py - Batch underwater image enhancement + evaluation.
python main.py --input raw-890\raw-890 --limit 20 --output results
Example:
    python main.py --input raw-890 --output results
    python main.py --input raw-890 --reference reference-890 --output results
    python main.py --input raw-890 --limit 50 --max-size 640   # quick test
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods import METHODS
from metrics import all_metrics

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(x, **kw):
        return x

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------ helpers
def load(path, max_size):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max_size and max(h, w) > max_size:
        s = max_size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def find_reference(ref_dir, stem):
    if ref_dir is None:
        return None
    for ext in EXTS:
        for e in (ext, ext.upper()):
            p = ref_dir / f"{stem}{e}"
            if p.exists():
                return p
    return None


def make_grid(images, titles, height=300):
    tiles = []
    for im, t in zip(images, titles):
        s = height / im.shape[0]
        tile = cv2.resize(im, (int(im.shape[1] * s), height))
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(tile, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def save_histograms(orig, enh, path, name):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for row, (im, title) in enumerate([(orig, "Original"), (enh, "Enhanced (fusion)")]):
        ax[row, 0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax[row, 0].set_title(f"{title} - {name}")
        ax[row, 0].axis("off")
        for ch, c in zip(range(3), ("b", "g", "r")):
            ax[row, 1].plot(cv2.calcHist([im], [ch], None, [256], [0, 256]), color=c)
        ax[row, 1].set_title(f"{title} RGB histogram")
        ax[row, 1].set_xlim(0, 255)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_summary_plot(summary, path):
    cols = [c for c in summary.columns]
    fig, ax = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    ax = np.atleast_1d(ax)
    for a, c in zip(ax, cols):
        summary[c].plot.bar(ax=a, color="steelblue", edgecolor="k")
        a.set_title(f"Mean {c}")
        a.set_xlabel("")
        a.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Underwater image enhancement project")
    ap.add_argument("--input", required=True, help="folder with raw images (e.g. raw-890)")
    ap.add_argument("--reference", default=None, help="optional folder with reference images (reference-890)")
    ap.add_argument("--output", default="results")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--limit", type=int, default=None, help="process only first N images")
    ap.add_argument("--max-size", type=int, default=800, help="resize longer side (0 = keep original)")
    ap.add_argument("--grids", type=int, default=20, help="number of side-by-side comparison images to save")
    args = ap.parse_args()

    in_dir = Path(args.input)
    ref_dir = Path(args.reference) if args.reference else None
    out = Path(args.output)
    for m in args.methods:
        (out / m).mkdir(parents=True, exist_ok=True)
    (out / "comparisons").mkdir(exist_ok=True)

    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No images found in {in_dir}")
    print(f"Found {len(files)} images | methods: {', '.join(args.methods)}")

    rows = []
    for i, f in enumerate(tqdm(files, desc="Enhancing")):
        img = load(f, args.max_size)
        if img is None:
            print(f"  skipped unreadable file: {f.name}")
            continue
        ref_path = find_reference(ref_dir, f.stem)
        ref = load(ref_path, args.max_size) if ref_path else None

        results = {"original": img}
        for m in args.methods:
            results[m] = METHODS[m](img)
            cv2.imwrite(str(out / m / f"{f.stem}.png"), results[m])

        for name, im in results.items():
            rows.append({"image": f.name, "method": name, **all_metrics(im, ref)})

        if i < args.grids:
            grid = make_grid(list(results.values()) + ([ref] if ref is not None else []),
                             list(results) + (["reference"] if ref is not None else []))
            cv2.imwrite(str(out / "comparisons" / f"{f.stem}.jpg"), grid)
            if i == 0 and "fusion" in results:
                save_histograms(img, results["fusion"], out / "histograms.png", f.name)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_per_image.csv", index=False)
    summary = df.groupby("method").mean(numeric_only=True)
    summary = summary.reindex(["original"] + args.methods)
    summary.round(4).to_csv(out / "metrics_summary.csv")
    save_summary_plot(summary, out / "metrics_summary.png")

    print("\n=== Mean metrics over dataset ===")
    print(summary.round(3).to_string())
    print(f"\nAll results saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
