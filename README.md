# Underwater Image Enhancement (Digital Image Processing Project)

Enhances the 890 raw underwater images in `raw-890` (UIEB-style dataset) using classical
DIP techniques, and compares them with quantitative metrics. Pure Python (NumPy + OpenCV).

## Problem
Underwater images suffer from:
1. **Colour cast** - red light is absorbed first, so images look blue/green.
2. **Low contrast / haze** - light scattering by particles.
3. **Blur** - forward scattering reduces edge detail.

## Methods (`methods.py`)
| Name | Technique | Fixes |
|---|---|---|
| `gray_world` | Gray-World white balance | colour cast |
| `he` | Histogram equalisation (V channel of HSV) | contrast |
| `clahe` | CLAHE on L channel of LAB | local contrast |
| `udcp` | Red compensation + Underwater Dark Channel Prior + guided-filter transmission | haze |
| `fusion` (**proposed**) | Red compensation + gray-world, then Laplacian-pyramid fusion of a sharpened input and a CLAHE input using contrast, saliency, saturation and exposedness weights | colour, contrast, detail |

## Metrics (`metrics.py`)
* No-reference: **Entropy**, **UIQM** (simplified), **UCIQE**
* Full-reference (when `--reference` given): **PSNR**, **SSIM**

## Setup & run
```bash
pip install -r requirements.txt

# folder layout
#   raw-890/        <- your 890 images
#   reference-890/  <- (optional) UIEB reference images with the same file names

python main.py --input raw-890 --output results
python main.py --input raw-890 --reference reference-890 --output results

# quick test
python main.py --input raw-890 --limit 20
```
Options: `--methods`, `--limit`, `--max-size` (default 800, `0` = original size), `--grids`.

## Output (`results/`)
* `<method>/` - enhanced images for every method
* `comparisons/` - side-by-side: original | each method | reference
* `histograms.png` - RGB histograms before/after (first image)
* `metrics_per_image.csv`, `metrics_summary.csv`, `metrics_summary.png`

## Report tips
* Use `comparisons/` images as qualitative results and `metrics_summary.csv` as the table.
* UIQM/UCIQE here are simplified implementations; mention this in your report.
* Discuss why a method wins/loses on certain metrics (e.g. UDCP raises contrast but can over-darken; HE can over-amplify noise; fusion keeps natural colours).

## References
* Ancuti et al., "Enhancing Underwater Images and Videos by Fusion", CVPR 2012
* He et al., "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009
* Li et al., "An Underwater Image Enhancement Benchmark Dataset" (UIEB), 2019
* Panetta et al., UIQM, 2016; Yang & Sowmya, UCIQE, 2015
