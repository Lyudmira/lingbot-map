# Fork-only: this file is not part of Meta Platforms, Inc.'s original lingbot-map
# source distribution. Copyright (c) the contributors to this repository fork.
"""Load static head / rig masks and gate depth confidence (separate from sky ONNX path).

When demo uses --crop_aspect, cached masks match uncropped source pixels; ``input_center_crops``
replays the same rectangle before resizing to ``depth_conf`` size so gating matches the model grid.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

# Match sky pipeline: high mask value keeps confidence.
_HEAD_MASK_SOFT_THRESHOLD = 0.1


def resolve_head_mask_path(mask_dir: str | Path, image_basename: str) -> Optional[str]:
    """Return first existing mask file for an image basename (PNG bytes may use image basename)."""
    md = Path(mask_dir)
    stem = Path(image_basename).stem
    candidates = [
        md / image_basename,
        md / f"{image_basename}.png",
        md / f"{stem}.png",
        md / f"{stem}.jpeg.png",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


# (left, top, crop_w, crop_h, orig_w, orig_h) in original-image pixels; legacy 5-tuple uses crop_w=crop_h=side.
InputCenterCrop = Union[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int, int]]


def _unpack_input_center_crop(crop: InputCenterCrop) -> Tuple[int, int, int, int, int, int]:
    if len(crop) == 5:
        left, top, side, orig_w, orig_h = crop
        return left, top, side, side, orig_w, orig_h
    left, top, crop_w, crop_h, orig_w, orig_h = crop
    return left, top, crop_w, crop_h, orig_w, orig_h


def _mask_to_orig_hw_then_center_crop(
    m: np.ndarray,
    crop: InputCenterCrop,
) -> np.ndarray:
    """Match ``load_fn.center_aspect_crop``: resize mask to orig_w×orig_h if needed, then crop."""
    left, top, crop_w, crop_h, orig_w, orig_h = _unpack_input_center_crop(crop)
    h, w = m.shape[:2]
    if h != orig_h or w != orig_w:
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return m[top : top + crop_h, left : left + crop_w]


def head_masks_complete(mask_dir: str | Path, image_basenames: List[str]) -> bool:
    if not image_basenames:
        return False
    md = Path(mask_dir)
    if not md.is_dir():
        return False
    for name in image_basenames:
        if resolve_head_mask_path(md, name) is None:
            return False
    return True


def load_head_masks(
    *,
    image_paths: List[str],
    mask_head_dir: str,
    target_h: int,
    target_w: int,
    num_frames: Optional[int] = None,
    input_center_crops: Optional[Sequence[Optional[InputCenterCrop]]] = None,
) -> Optional[np.ndarray]:
    """Stack per-frame uint8 masks resized to (H,W), float01 in [0,1]."""
    n = len(image_paths)
    if num_frames is not None:
        n = min(n, num_frames)
    if n == 0:
        return None

    masks: List[np.ndarray] = []
    md = mask_head_dir
    for i in tqdm(range(n), desc="Loading head masks", unit="frame"):
        base = os.path.basename(image_paths[i])
        mp = resolve_head_mask_path(md, base)
        if mp is None:
            print(f"Warning: missing head mask for {base} under {mask_head_dir}")
            return None
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"Warning: failed to read head mask {mp}")
            return None
        if input_center_crops is not None and i < len(input_center_crops) and input_center_crops[i] is not None:
            m = _mask_to_orig_hw_then_center_crop(m, input_center_crops[i])
        if m.shape[:2] != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        masks.append(m.astype(np.float32) / 255.0)

    return np.stack(masks, axis=0)


def apply_head_mask(
    conf: np.ndarray,
    *,
    image_paths: Optional[List[str]] = None,
    image_folder: Optional[str] = None,
    mask_head_dir: str,
    num_frames: Optional[int] = None,
    input_center_crops: Optional[Sequence[Optional[InputCenterCrop]]] = None,
) -> np.ndarray:
    """Multiply conf by binary head mask (255 / high = keep).

    ``input_center_crops``: per-frame boxes in original-image pixels; required when masks were
    built on full frames but ``conf`` is on cropped inference geometry (see demo.py note).
    """
    from lingbot_map.vis.sky_segmentation import _list_image_files

    S, H, W = conf.shape
    if image_paths is None:
        if image_folder is None:
            print("Warning: apply_head_mask needs image_paths or image_folder; skipping")
            return conf
        image_paths = _list_image_files(image_folder)

    if num_frames is not None:
        image_paths = image_paths[:num_frames]

    if len(image_paths) < S:
        print(
            f"Warning: only {len(image_paths)} paths for head mask but conf has S={S}; "
            "leaving tail frames unmasked"
        )

    arr = load_head_masks(
        image_paths=image_paths,
        mask_head_dir=mask_head_dir,
        target_h=H,
        target_w=W,
        num_frames=S,
        input_center_crops=input_center_crops,
    )
    if arr is None:
        return conf

    if arr.shape[0] < S:
        padded = np.zeros((S, H, W), dtype=arr.dtype)
        padded[: arr.shape[0]] = arr
        arr = padded
    elif arr.shape[0] > S:
        arr = arr[:S]

    binary = (arr > _HEAD_MASK_SOFT_THRESHOLD).astype(np.float32)
    return conf * binary
