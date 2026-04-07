"""One-shot script: extract figures from the FL-TAC paper PDF.

We render specific page regions to PNG. Bounding boxes were eyeballed from
the rendered pages; if you want to re-extract, render the page first
(`page.get_pixmap(dpi=200)`) and read the y-coords from the rendered image.
"""
import fitz  # pymupdf
from pathlib import Path

PDF = "2404.15384v1.pdf"
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

# (figure name, page index 0-based, clip rect in PDF points or None for full)
# PDF point coords: origin top-left, 1 pt = 1/72 inch.  Page is ~612 x 792.
SPECS = [
    # Figure 1: framework, page 2 (index 1)
    ("framework", 1, fitz.Rect(120, 440, 510, 595)),
    # Figure 2: data dist + radar, page 5 (index 4)
    ("data_distribution_and_radar", 4, fitz.Rect(85, 60, 530, 290)),
    # Figure 3: approximation error vs LoRA rank, page 5 (index 4)
    ("approx_error_vs_rank", 4, fitz.Rect(310, 510, 555, 690)),
    # Figure 4: UMAP clustering progression, page 9 (index 8)
    ("umap_clustering", 8, fitz.Rect(85, 305, 530, 450)),
]

doc = fitz.open(PDF)
for name, page_idx, rect in SPECS:
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=300, clip=rect)
    out_path = OUT / f"{name}.png"
    pix.save(out_path)
    print(f"  -> {out_path}  ({pix.width}x{pix.height})")
doc.close()
