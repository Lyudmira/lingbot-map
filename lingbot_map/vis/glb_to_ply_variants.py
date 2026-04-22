# Fork-only: this file is not part of Meta Platforms, Inc.'s original lingbot-map
# source distribution. Copyright (c) the contributors to this repository fork.

"""
Convert an existing GLB scene into three colored PLY point clouds.

The module reuses the point-cloud export helpers used by PointCloudViewer:
- `_full.ply`: the complete point cloud
- `_s.ply`: up to 600,000 points
- `_m.ply`: up to 2,400,000 points

Usage:
    python -m lingbot_map.vis.glb_to_ply_variants \
        /path/to/input.glb \
        --output_dir /path/to/output_dir
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - import error is surfaced at runtime.
    trimesh = None

from lingbot_map.vis.point_cloud_viewer import PointCloudViewer


VariantSpec = Tuple[str, Optional[int]]

DEFAULT_VARIANTS: Sequence[VariantSpec] = (
    ("full", None),
    ("s", 600_000),
    ("m", 2_400_000),
)


def _as_uint8_colors(colors: np.ndarray, num_points: int) -> np.ndarray:
    """Normalize point colors into an (N, 4) uint8 array."""
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != num_points:
        raise ValueError(f"colors must have shape (N, C) matching points, got {colors.shape}")

    if colors.dtype != np.uint8:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.size > 0 and float(np.max(colors)) <= 1.0 + 1e-6:
            colors = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)

    if colors.shape[1] == 3:
        alpha = np.full((num_points, 1), 255, dtype=np.uint8)
        colors = np.concatenate([colors, alpha], axis=1)
    elif colors.shape[1] != 4:
        raise ValueError(f"colors must have 3 or 4 channels, got {colors.shape}")

    return colors


def extract_point_cloud_from_scene(scene: "trimesh.Scene") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all PointCloud geometries from a trimesh scene and apply node transforms.

    The output is a single concatenated point cloud suitable for PLY export.
    """
    if trimesh is None:
        raise ImportError("trimesh is required. Install it with: pip install trimesh")

    all_vertices: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph.get(node_name)
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.points.PointCloud):
            continue

        vertices = np.asarray(geom.vertices, dtype=np.float32)
        colors = getattr(geom, "colors", None)
        if colors is None:
            colors = getattr(getattr(geom, "visual", None), "vertex_colors", None)
        if colors is None:
            colors = np.full((len(vertices), 4), 255, dtype=np.uint8)
        colors = _as_uint8_colors(colors, len(vertices))

        vertices = PointCloudViewer._apply_transform_to_points(vertices, transform)
        all_vertices.append(vertices)
        all_colors.append(colors)

    if not all_vertices:
        raise ValueError("No PointCloud geometries found in the input GLB")

    return np.concatenate(all_vertices, axis=0), np.concatenate(all_colors, axis=0)


def export_glb_to_ply_variants(
    glb_path: str,
    output_dir: Optional[str] = None,
    output_stem: Optional[str] = None,
    variants: Sequence[VariantSpec] = DEFAULT_VARIANTS,
) -> List[str]:
    """
    Convert an existing GLB into three PLY point clouds.

    Returns the written output paths in export order.
    """
    if trimesh is None:
        raise ImportError("trimesh is required. Install it with: pip install trimesh")

    scene = trimesh.load(glb_path, force="scene")
    vertices, colors = extract_point_cloud_from_scene(scene)

    if output_dir is None:
        output_dir = os.path.dirname(glb_path) or "."
    if output_stem is None:
        output_stem = os.path.splitext(os.path.basename(glb_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    full_vertices = np.asarray(vertices, dtype=np.float32)
    full_colors = np.asarray(colors, dtype=np.uint8)
    written_paths: List[str] = []

    for suffix, max_points in variants:
        if max_points is None:
            ply_vertices = full_vertices
            ply_colors = full_colors
        else:
            indices = PointCloudViewer._sample_point_indices(len(full_vertices), max_points)
            if indices is None:
                ply_vertices = full_vertices
                ply_colors = full_colors
            else:
                ply_vertices = full_vertices[indices]
                ply_colors = full_colors[indices]

        output_path = os.path.join(output_dir, f"{output_stem}_{suffix}.ply")
        n_pts = PointCloudViewer._write_point_cloud_ply(
            output_path,
            vertices=ply_vertices,
            colors_rgba=ply_colors,
        )
        written_paths.append(output_path)
        print(f"PLY exported to {output_path} ({n_pts:,} points)")

    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an existing GLB point-cloud scene into _full/_s/_m PLY variants."
    )
    parser.add_argument("glb_path", type=str, help="Path to the input GLB file.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to the GLB's directory.",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="Output filename stem. Defaults to the GLB basename without extension.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    export_glb_to_ply_variants(
        glb_path=args.glb_path,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
