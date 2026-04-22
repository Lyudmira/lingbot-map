#!/usr/bin/env python3
# Fork-only: this file is not part of Meta Platforms, Inc.'s original lingbot-map
# source distribution. Copyright (c) the contributors to this repository fork.
"""
CLI for flat-folder head / rig masks. Library implementation: lingbot_map.utils.head_mask_v3_flat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from lingbot_map.utils.head_mask_v3_flat import FlatHeadMaskConfig, run_flat


def _write_meta_stub(meta_path: Path, *, width: int, height: int, camera: str = "camera_0") -> None:
    meta = {
        "calibrationInfo": {
            camera: {
                "intrinsics": {
                    "camera_type": "OpenCVFisheye",
                    "width": width,
                    "height": height,
                    "camera_matrix": [
                        float(width),
                        0.0,
                        width / 2.0,
                        0.0,
                        float(height),
                        height / 2.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
                },
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }
        },
        "data": [],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote stub meta: {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image_dir", type=Path, help="Flat folder of sequential images.")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Mask output root (default: <image_dir>_head_masks next to the path you pass).",
    )
    p.add_argument("--extension", type=str, default=".jpeg")
    p.add_argument(
        "--mode",
        choices=("head", "head_and_lens"),
        default="head",
        help="head: single compact occluder; head_and_lens: keep all candidate blobs.",
    )
    p.add_argument("--sample-fraction", type=float, default=0.25)
    p.add_argument("--downsample-factor", type=int, default=4)
    p.add_argument(
        "--write-meta-stub",
        type=Path,
        default=None,
        help="If set, write identity OpenCVFisheye meta JSON to this path.",
    )
    args = p.parse_args()

    image_dir_user = args.image_dir
    image_dir_resolved = image_dir_user.resolve()
    out = (
        args.output_dir.resolve()
        if args.output_dir
        else (image_dir_user.parent.resolve() / f"{image_dir_user.name}_head_masks")
    )

    if args.write_meta_stub is not None:
        ext = args.extension if args.extension.startswith(".") else f".{args.extension}"
        ref = cv2.imread(str(next(image_dir_resolved.glob(f"*{ext}"))), cv2.IMREAD_COLOR)
        if ref is None:
            raise SystemExit("Cannot read a reference image for dimensions.")
        h, w = ref.shape[:2]
        _write_meta_stub(args.write_meta_stub.resolve(), width=w, height=h)

    run_flat(
        FlatHeadMaskConfig(
            image_dir=image_dir_resolved,
            output_dir=out,
            extension=args.extension.lower() if args.extension.startswith(".") else f".{args.extension}",
            mode=args.mode,
            sample_fraction=args.sample_fraction,
            downsample_factor=args.downsample_factor,
        )
    )


if __name__ == "__main__":
    main()
