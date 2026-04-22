#!/usr/bin/env python3
"""Center-crop images in a directory to 1:1 (largest inscribed square)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_one(src: Path, dst: Path, quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        out = center_crop_square(im)
        suffix = dst.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            out.save(dst, quality=quality, optimize=True)
        else:
            out.save(dst)


def _job(args: tuple[str, str, int]) -> None:
    s, d, q = args
    process_one(Path(s), Path(d), q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder of images (read-only).")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <input_dir>_square next to input.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=8,
        help="Parallel workers (processes).",
    )
    args = parser.parse_args()

    inp = args.input_dir
    if not inp.is_dir():
        raise SystemExit(f"Not a directory: {inp}")

    # List files from the resolved target (symlinks OK); place default output
    # beside the path the user passed, not necessarily beside the real directory.
    inp_resolved = inp.resolve()
    out_root = (
        args.output_dir.resolve()
        if args.output_dir
        else (inp.parent.resolve() / f"{inp.name}_square")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p
        for p in inp_resolved.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise SystemExit(f"No images found in {inp}")

    jobs = []
    for p in paths:
        jobs.append((str(p), str(out_root / p.name), args.jpeg_quality))

    if args.workers <= 1:
        for s, d, q in jobs:
            process_one(Path(s), Path(d), q)
            print(Path(d).name)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for fut in as_completed(futs):
                fut.result()

    print(f"Done: {len(paths)} images -> {out_root}")


if __name__ == "__main__":
    main()
