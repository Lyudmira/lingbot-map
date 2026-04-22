# Copyright (c) Meta Platforms, Inc. and affiliates.
# V3-style temporal head / rig occluder mask for a flat image folder (library entry).

from __future__ import annotations

import typing as t
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_STATISTIC_SMOOTH_CLOSE_FACTOR: int = 40
_DISP_NOISE_FLOOR: float = 0.01
_DISP_FALLBACK_RANGE_FRACTION: float = 0.08
_DISP_FALLBACK_MAX_COMPONENT_RATIO: float = 0.15
_TEMPORAL_MASS_GATE_MAX_CENTERS: int = 8


def _bounded_odd(value: int, upper_bound: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    if upper_bound <= 1:
        return 1
    if upper_bound % 2 == 0:
        upper_bound -= 1
    return max(1, min(value, upper_bound))


def _load_frame_lab(image_path: Path, *, downsample_factor: int) -> np.ndarray:
    bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = bgr.shape[:2]
    bgr = cv2.resize(
        bgr,
        (max(1, w // downsample_factor), max(1, h // downsample_factor)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _sample_image_paths(
    image_paths: t.List[Path],
    *,
    sample_fraction: float,
    min_sample_frames: int,
    random_seed: int,
    camera_name: str,
) -> t.List[Path]:
    if len(image_paths) <= min_sample_frames:
        return list(image_paths)
    sample_count = max(min_sample_frames, int(np.ceil(len(image_paths) * sample_fraction)))
    sample_count = min(len(image_paths), sample_count)
    stable_seed = zlib.crc32(camera_name.encode("utf-8"), random_seed) & 0xFFFFFFFF
    rng = np.random.default_rng(stable_seed)
    sampled_indices = np.sort(rng.permutation(len(image_paths))[:sample_count])
    return [image_paths[i] for i in sampled_indices]


def _smooth_candidate_mask(candidate_mask: np.ndarray) -> np.ndarray:
    candidate_mask = candidate_mask.astype(np.uint8)
    h, w = candidate_mask.shape
    ks = _bounded_odd(min(h, w) // _STATISTIC_SMOOTH_CLOSE_FACTOR, min(h, w))
    if ks <= 1:
        return candidate_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)


def _apply_dispersion_fallback(
    dispersion_map: np.ndarray,
    disp_stable: np.ndarray,
    *,
    global_disp_threshold: float,
) -> np.ndarray:
    if global_disp_threshold > _DISP_NOISE_FLOOR:
        return disp_stable.astype(np.uint8)
    vmin = float(dispersion_map.min())
    vmax = float(dispersion_map.max())
    range_thr = vmin + _DISP_FALLBACK_RANGE_FRACTION * (vmax - vmin)
    disp_stable = (dispersion_map <= range_thr).astype(np.uint8)
    h_d, w_d = disp_stable.shape
    max_px = int(h_d * w_d * _DISP_FALLBACK_MAX_COMPONENT_RATIO)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(disp_stable, connectivity=8)
    for lbl in range(1, n_lbl):
        if int(stats[lbl, cv2.CC_STAT_AREA]) > max_px:
            disp_stable[lbl_map == lbl] = 0
    return disp_stable


def _compute_trimmed_consensus_mask(
    candidate_mask: np.ndarray,
    *,
    keep_weight_fraction: float,
) -> np.ndarray:
    binary = candidate_mask.astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return np.zeros_like(binary)

    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    kept_labels: t.List[int] = []
    kept_weights: t.List[float] = []
    kept_centers: t.List[t.Tuple[float, float]] = []
    for lbl in range(1, n_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        kept_labels.append(lbl)
        kept_weights.append(float(area))
        kept_centers.append((float(centroids[lbl, 0]), float(centroids[lbl, 1])))

    if not kept_labels:
        return np.zeros_like(binary)

    centers_arr = np.array(kept_centers, dtype=np.float32)
    weights_arr = np.array(kept_weights, dtype=np.float32)
    keep_weight_fraction = float(np.clip(keep_weight_fraction, 0.0, 1.0))
    target_weight = max(float(weights_arr.max()), keep_weight_fraction * float(weights_arr.sum()))

    best_idx = 0
    best_radius = float("inf")
    best_inlier_weight = -1.0
    for idx, center in enumerate(centers_arr):
        distances = np.linalg.norm(centers_arr - center[None], axis=1)
        order = np.argsort(distances, kind="stable")
        cumulative = np.cumsum(weights_arr[order])
        cover_idx = min(int(np.searchsorted(cumulative, target_weight, side="left")), len(order) - 1)
        radius = float(distances[order[cover_idx]])
        inlier_weight = float(weights_arr[distances <= radius + 1e-6].sum())
        if radius < best_radius - 1e-6 or (
            abs(radius - best_radius) <= 1e-6 and inlier_weight > best_inlier_weight
        ):
            best_idx = idx
            best_radius = radius
            best_inlier_weight = inlier_weight

    final_distances = np.linalg.norm(centers_arr - centers_arr[best_idx][None], axis=1)
    selected_mask = np.zeros_like(binary)
    for lbl, dist in zip(kept_labels, final_distances):
        if dist <= best_radius + 1e-6:
            selected_mask[labels == lbl] = 1
    return selected_mask


def _compute_power2_dispersion(
    frames_lab: np.ndarray,
    *,
    global_y_p5: float,
    global_y_p95: float,
) -> np.ndarray:
    L = frames_lab[..., 0]
    a = frames_lab[..., 1]
    b = frames_lab[..., 2]
    Y = ((L + 16.0) / 116.0) ** 3.0
    y_range = max(global_y_p95 - global_y_p5, 1e-6)
    Y_norm = np.clip((Y - global_y_p5) / y_range, 0.0, 1.0)
    frames_t = np.stack([Y_norm**2, a / 100.0, b / 100.0], axis=-1)
    med = np.median(frames_t, axis=0)
    mad = np.median(np.abs(frames_t - med[None]), axis=0)
    return mad.mean(axis=-1)


def _compute_temporal_mass_gate_map(
    frames_lab: np.ndarray,
    *,
    global_y_p5: float,
    global_y_p95: float,
    tau: float,
) -> np.ndarray:
    L = frames_lab[..., 0]
    a = frames_lab[..., 1]
    b = frames_lab[..., 2]
    Y = ((L + 16.0) / 116.0) ** 3.0
    y_range = max(global_y_p95 - global_y_p5, 1e-6)
    Y_norm = np.clip((Y - global_y_p5) / y_range, 0.0, 1.0)
    features = np.stack([Y_norm, a / 100.0, b / 100.0], axis=-1).astype(np.float32)
    count = max(1, min(features.shape[0], _TEMPORAL_MASS_GATE_MAX_CENTERS))
    centers = np.unique(np.round(np.linspace(0, features.shape[0] - 1, count)).astype(np.int32))
    best_mass = np.zeros(features.shape[1:3], dtype=np.float32)
    for idx in centers:
        distances = np.linalg.norm(features - features[idx][None], axis=-1)
        mass = (distances <= tau).mean(axis=0, dtype=np.float32)
        best_mass = np.maximum(best_mass, mass)
    return best_mass


class _CameraMaskBuilder:
    def __init__(
        self,
        frames_lab: np.ndarray,
        *,
        original_size: t.Tuple[int, int],
        global_y_p5: float,
        global_y_p95: float,
        global_disp_threshold: float,
        temporal_mass_gate_tau: float,
        temporal_mass_gate_threshold: float,
        outward_dilation_pixels: int,
        consensus_keep_fraction: float,
        mode: t.Literal["head", "head_and_lens"],
        validate_mask_color: t.Literal["white", "black"],
    ) -> None:
        self._frames_lab = frames_lab
        self._original_size = original_size
        self._global_y_p5 = global_y_p5
        self._global_y_p95 = global_y_p95
        self._global_disp_threshold = global_disp_threshold
        self._temporal_mass_gate_tau = temporal_mass_gate_tau
        self._temporal_mass_gate_threshold = temporal_mass_gate_threshold
        self._outward_dilation_pixels = outward_dilation_pixels
        self._consensus_keep_fraction = consensus_keep_fraction
        self._mode = mode
        self._validate_mask_color = validate_mask_color

    def _dilate(self, mask: np.ndarray) -> np.ndarray:
        ks = _bounded_odd(max(0, self._outward_dilation_pixels) * 2 + 1, min(mask.shape))
        if ks <= 1:
            return mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    def build(self) -> np.ndarray:
        disp_map = _compute_power2_dispersion(
            self._frames_lab,
            global_y_p5=self._global_y_p5,
            global_y_p95=self._global_y_p95,
        )
        disp_stable = (disp_map <= self._global_disp_threshold).astype(np.uint8)
        disp_stable = _apply_dispersion_fallback(
            disp_map,
            disp_stable,
            global_disp_threshold=self._global_disp_threshold,
        )

        temporal_mass = _compute_temporal_mass_gate_map(
            self._frames_lab,
            global_y_p5=self._global_y_p5,
            global_y_p95=self._global_y_p95,
            tau=self._temporal_mass_gate_tau,
        )
        temporal_mass_stable = (temporal_mass >= self._temporal_mass_gate_threshold).astype(np.uint8)

        combined = _smooth_candidate_mask(disp_stable & temporal_mass_stable)
        if self._mode == "head":
            selected = _compute_trimmed_consensus_mask(
                combined,
                keep_weight_fraction=self._consensus_keep_fraction,
            )
        else:
            selected = combined
        occluder_lowres = self._dilate(selected)

        orig_h, orig_w = self._original_size
        occluder_full = cv2.resize(occluder_lowres, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if self._validate_mask_color == "white":
            output_mask = np.ones_like(occluder_full, dtype=np.uint8) * 255
            output_mask[occluder_full > 0] = 0
        else:
            output_mask = np.zeros_like(occluder_full, dtype=np.uint8)
            output_mask[occluder_full > 0] = 255
        return output_mask


def _write_camera_masks(
    *,
    camera_image_paths: t.List[Path],
    image_dir: Path,
    mask_dir: Path,
    mask: np.ndarray,
    pool_workers: int,
) -> None:
    """Write PNG-encoded masks using the same basename as each source image (for demo loading)."""

    def _write_single_mask(image_path: Path) -> None:
        out_file = mask_dir / image_path.name
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for {image_path.name}")
        out_file.write_bytes(buf.tobytes())

    with ThreadPoolExecutor(max_workers=pool_workers) as executor:
        list(executor.map(_write_single_mask, camera_image_paths))


@dataclass(kw_only=True)
class FlatHeadMaskConfig:
    image_dir: Path
    output_dir: Path
    extension: str = ".jpeg"
    validate_mask_color: t.Literal["white", "black"] = "white"
    downsample_factor: int = 4
    pool_workers: int = 16
    sample_fraction: float = 0.25
    min_sample_frames: int = 4
    random_seed: int = 0
    stability_fraction: float = 0.0325
    temporal_mass_gate_tau: float = 0.12
    temporal_mass_gate_threshold: float = 0.92
    mode: t.Literal["head", "head_and_lens"] = "head"
    consensus_keep_fraction: float = 0.70
    outward_dilation_pixels: int = 8
    global_y_percentile_low: float = 5.0
    global_y_percentile_high: float = 95.0
    virtual_camera_name: str = "flat_cam"


def run_flat(cfg: FlatHeadMaskConfig) -> None:
    assert cfg.image_dir.is_dir(), cfg.image_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    exts = (cfg.extension.lower(),)
    camera_image_paths: t.List[Path] = sorted(
        p for p in cfg.image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    assert camera_image_paths, f"No *{cfg.extension} under {cfg.image_dir}"

    sampled = _sample_image_paths(
        camera_image_paths,
        sample_fraction=cfg.sample_fraction,
        min_sample_frames=cfg.min_sample_frames,
        random_seed=cfg.random_seed,
        camera_name=cfg.virtual_camera_name,
    )
    with ThreadPoolExecutor(max_workers=cfg.pool_workers) as executor:
        frames_list = list(
            executor.map(
                lambda p: _load_frame_lab(p, downsample_factor=cfg.downsample_factor),
                sampled,
            )
        )
    frames = np.stack(frames_list, axis=0)
    all_y = ((frames[..., 0] + 16.0) / 116.0) ** 3.0
    all_y_arr = all_y.ravel()
    global_y_p5 = float(np.percentile(all_y_arr, cfg.global_y_percentile_low))
    global_y_p95 = float(np.percentile(all_y_arr, cfg.global_y_percentile_high))
    print(f"Global Y: p5={global_y_p5:.4f}  p95={global_y_p95:.4f}")

    disp = _compute_power2_dispersion(frames, global_y_p5=global_y_p5, global_y_p95=global_y_p95)
    global_disp_thr = float(np.quantile(disp, cfg.stability_fraction))
    print(f"Global dispersion threshold: {global_disp_thr:.6f}")

    ref = cv2.imread(camera_image_paths[0].as_posix(), cv2.IMREAD_GRAYSCALE)
    assert ref is not None, camera_image_paths[0]
    original_size = (int(ref.shape[0]), int(ref.shape[1]))

    camera_mask = _CameraMaskBuilder(
        frames,
        original_size=original_size,
        global_y_p5=global_y_p5,
        global_y_p95=global_y_p95,
        global_disp_threshold=global_disp_thr,
        temporal_mass_gate_tau=cfg.temporal_mass_gate_tau,
        temporal_mass_gate_threshold=cfg.temporal_mass_gate_threshold,
        outward_dilation_pixels=cfg.outward_dilation_pixels,
        consensus_keep_fraction=cfg.consensus_keep_fraction,
        mode=cfg.mode,
        validate_mask_color=cfg.validate_mask_color,
    ).build()

    _write_camera_masks(
        camera_image_paths=camera_image_paths,
        image_dir=cfg.image_dir,
        mask_dir=cfg.output_dir,
        mask=camera_mask,
        pool_workers=cfg.pool_workers,
    )
    print(f"Wrote {len(camera_image_paths)} head-mask files under {cfg.output_dir}")
