# Fork-only: this file is not part of Meta Platforms, Inc.'s original lingbot-map
# source distribution. Copyright (c) the contributors to this repository fork.

"""Export LingBot-MAP predictions to COLMAP text format (cameras / images / points3D)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def _parse_pc_chunk(
    pc: np.ndarray,
    color: np.ndarray,
    conf: Optional[np.ndarray],
    conf_threshold: float,
    downsample_factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match PointCloudViewer.parse_pc_data filtering (no GUI, no border)."""
    if pc.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    pred_pts = pc.reshape(-1, 3)
    if np.isnan(color).any():
        color_img = np.zeros((pred_pts.shape[0], 3), dtype=np.float32)
        color_img[:, 2] = 1.0
    else:
        color_img = color.reshape(-1, 3)

    valid = np.isfinite(pred_pts).all(axis=1)
    if not valid.all():
        pred_pts = pred_pts[valid]
        color_img = color_img[valid]
        if conf is not None:
            conf = conf.reshape(-1)[valid]

    if conf is not None:
        conf_flat = conf.reshape(-1) if conf.ndim > 1 else conf
        mask = conf_flat > conf_threshold
        pred_pts = pred_pts[mask]
        color_img = color_img[mask]

    if len(pred_pts) == 0:
        return pred_pts, np.zeros((0, 3), np.uint8)

    if downsample_factor > 1:
        idx = np.arange(0, len(pred_pts), downsample_factor)
        pred_pts = pred_pts[idx]
        color_img = color_img[idx]

    if color_img.dtype != np.uint8:
        color_u8 = (np.clip(color_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        color_u8 = color_img
    return pred_pts.astype(np.float32), color_u8


def _w2c_to_colmap_quat_t(w2c: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Convert a world-to-camera 3x4 pose into COLMAP's ``images.txt`` fields."""
    R_wc = w2c[:3, :3].astype(np.float64)
    t_wc = w2c[:3, 3].astype(np.float64)
    quat_xyzw = R.from_matrix(R_wc).as_quat()
    qw, qx, qy, qz = float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])
    return qw, qx, qy, qz, float(t_wc[0]), float(t_wc[1]), float(t_wc[2])


def export_colmap_sparse(
    sparse_dir: Path | str,
    vis_predictions: Dict[str, Any],
    pc_list: Sequence[np.ndarray],
    color_list: Sequence[np.ndarray],
    conf_list: Sequence[np.ndarray],
    *,
    conf_threshold: float,
    downsample_factor: int,
    max_points: int = 5_000_000,
) -> None:
    """Write COLMAP text model under ``sparse_dir`` (typically ``.../sparse/0``).

    Point coordinates are the same depth-unprojection world frame as GLB export (model scale).
    Cameras use predicted PINHOLE intrinsics at model input resolution; poses follow the same
    world-to-camera convention used by the viewer and depth unprojection.
    """
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    extr = np.asarray(vis_predictions["extrinsic"])
    intr = np.asarray(vis_predictions["intrinsic"])
    images_arr = np.asarray(vis_predictions["images"])
    if images_arr.ndim != 4 or images_arr.shape[1] != 3:
        raise ValueError("vis_predictions['images'] must be (S, 3, H, W)")
    s, _, h, w = images_arr.shape
    if extr.shape[0] != s or intr.shape[0] != s:
        raise ValueError("extrinsic/intrinsic length must match number of frames")
    paths = vis_predictions.get("image_paths")
    if paths is None or len(paths) != s:
        paths = [f"{i:06d}.png" for i in range(s)]
    names = [os.path.basename(str(p)) for p in paths]

    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    points_path = sparse_dir / "points3D.txt"

    # --- cameras.txt: one PINHOLE per frame (K can differ slightly per timestep) ---
    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for i in range(s):
        fx, fy = float(intr[i, 0, 0]), float(intr[i, 1, 1])
        cx, cy = float(intr[i, 0, 2]), float(intr[i, 1, 2])
        cam_lines.append(f"{i + 1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}")
    cameras_path.write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    # --- images.txt ---
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for i in range(s):
        qw, qx, qy, qz, tx, ty, tz = _w2c_to_colmap_quat_t(extr[i])
        img_lines.append(
            f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i + 1} {names[i]}"
        )
        img_lines.append("")  # no 2D keypoints / tracks
    images_path.write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    # --- points3D.txt (merged, same filtering as GLB) ---
    all_pts: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    for step in range(s):
        c_step = np.asarray(conf_list[step])
        pts, cols = _parse_pc_chunk(
            np.asarray(pc_list[step]),
            np.asarray(color_list[step]),
            c_step if c_step.size > 0 else None,
            conf_threshold,
            downsample_factor,
        )
        if len(pts) > 0:
            all_pts.append(pts)
            all_rgb.append(cols)

    if not all_pts:
        pts_merged = np.zeros((0, 3), np.float64)
        rgb_merged = np.zeros((0, 3), np.uint8)
    else:
        pts_merged = np.concatenate(all_pts, axis=0)
        rgb_merged = np.concatenate(all_rgb, axis=0)

    if len(pts_merged) > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts_merged), size=max_points, replace=False)
        sel.sort()
        pts_merged = pts_merged[sel]
        rgb_merged = rgb_merged[sel]
        print(f"  COLMAP: subsampled points to {max_points} (cap)")

    pt_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for j in range(len(pts_merged)):
        x, y, z = pts_merged[j]
        r, g, b = rgb_merged[j]
        pt_lines.append(f"{j + 1} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0")
    points_path.write_text("\n".join(pt_lines) + "\n", encoding="utf-8")

    print(f"COLMAP text model written to {sparse_dir} ({len(pts_merged)} points, {s} images)")
