"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Defaults: lingbot-map-long.pt + dinov2_vitl14_reg_official.pth under ./checkpoints/
    # Also default: windowed mode, offload to CPU, export GLB to output/<folder_name>.glb, no viewer
    python demo.py --image_folder /path/to/images/

    # Open viewer instead (skip GLB with --no_export_glb)
    python demo.py --image_folder /path/to/images/ --no_export_glb --viewer

    # Streaming on GPU, custom GLB path
    python demo.py --image_folder /path/to/images/ --mode streaming --no-offload_to_cpu \
        --export_glb /path/to/out.glb

    # Default --crop_aspect is 16_9: ~16GB VRAM OK without --mask_head; ~24GB+ typical with --mask_head
    python demo.py --image_folder /path/to/images/
    python demo.py --image_folder /path/to/images/ --mask_head
    # Other aspects or no crop: --crop_aspect square|16_10|4_3|5_4|none
    python demo.py --image_folder /path/to/images/ --crop_aspect square --mask_head
    python demo.py --image_folder /path/to/images/ --square   # same as --crop_aspect square
"""

import argparse
import glob
import os
import time
from pathlib import Path

_DEFAULT_MODEL_PATH = "/data/users/mia/current/lingbot-map/checkpoints/lingbot-map-long.pt"
_DEFAULT_PRETRAINED_PATH = (
    "/data/users/mia/current/lingbot-map/checkpoints/dinov2_vitl14_reg_official.pth"
)
_EXPORT_GLB_AUTO = "__AUTO__"

# Must be set before `import torch` / any CUDA init. Reduces the reserved-vs-allocated
# memory gap by letting the caching allocator grow segments on demand instead of
# pre-reserving fixed-size blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# FlashInfer JIT/workspace: base directory; the library appends `.cache/flashinfer/...`.
# Use repo root so caches land in `<repo>/.cache/flashinfer/` (not `$HOME/.cache/...`).
# Override with `export FLASHINFER_WORKSPACE_BASE=...` if needed.
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE",
    str(Path(__file__).resolve().parent),
)

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def collect_media_paths(
    image_folder=None,
    video_path=None,
    fps=10,
    image_ext=".jpg,.jpeg,.png",
    first_k=None,
    stride=1,
):
    """Collect ordered image paths from a folder or extract frames from a video.

    Returns:
        (paths, resolved_image_folder): paths after ``first_k`` / ``stride``, and the folder
        used for head-mask naming and sky cache (extract folder for video, ``image_folder`` otherwise).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    return paths, resolved_folder


def _auto_export_frame_stride(num_frames: int) -> int:
    """GLB depth projection uses every N-th frame (1 = all). Steps by 3000-frame bands above 3000."""
    if num_frames <= 3000:
        return 1
    return 2 + (num_frames - 3001) // 3000


def _apply_auto_glb_decimation(
    num_frames: int, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Set --downsample_factor / --export_frame_stride from input length when omitted.

    Host RAM / swap (not VRAM): defaults target **~64 GiB RAM + ≥32 GiB swap** (see ``notes.md``).
    **Marginal:** 24 GB VRAM, **48 GiB RAM**, **~48 GiB swap** — may barely finish; use stricter
    ``--downsample_factor`` / ``--export_frame_stride`` or fewer frames if the process is killed.
    """
    if args.downsample_factor is None:
        if num_frames > 2000:
            args.downsample_factor = 3
        elif num_frames > 1000:
            args.downsample_factor = 2
        else:
            args.downsample_factor = 1
        if num_frames > 1000:
            print(
                f"Auto-selected --downsample_factor={args.downsample_factor} for {num_frames} input frames "
                f"(>1000→2, >2000→3 when omitted; see notes.md — ~64 GiB RAM + ≥32 GiB swap recommended)."
            )
    if args.export_frame_stride is None:
        args.export_frame_stride = _auto_export_frame_stride(num_frames)
        if num_frames > 3000:
            print(
                f"Auto-selected --export_frame_stride={args.export_frame_stride} for {num_frames} input frames "
                f"(>3000→2, >6000→3, >9000→4, … step every 3000; see notes.md for RAM/swap)."
            )
    if args.downsample_factor < 1 or args.export_frame_stride < 1:
        parser.error("--downsample_factor and --export_frame_stride must be >= 1.")


def _validate_media_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast with a clear message if paths from the CLI are missing (e.g. doc placeholder)."""
    if args.video_path is not None:
        vp = os.path.abspath(os.path.expanduser(args.video_path))
        if not os.path.isfile(vp):
            parser.error(f"Video not found or not a file: {vp}")
        return
    folder = os.path.abspath(os.path.expanduser(args.image_folder))
    if not os.path.isdir(folder):
        parser.error(
            f"Image folder does not exist or is not a directory: {folder}\n"
            f"Use a real path on this machine (paths like /path/to/images are documentation placeholders only)."
        )


def preprocess_media_paths(
    paths, image_size=518, patch_size=14, crop_aspect: str | None = None
):
    """Run ``load_and_preprocess_images`` on existing paths; optional centered aspect crop first."""
    print(f"Loading {len(paths)} images...")
    images, center_crop_boxes = load_and_preprocess_images(
        paths,
        mode="pad",
        image_size=image_size,
        patch_size=patch_size,
        center_aspect_crop=crop_aspect,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using aspect-preserving pad mode")
    if crop_aspect:
        print(f"Center {crop_aspect.replace('_', ':')} aspect crop applied before resize/pad.")
    else:
        print("No center aspect crop (full frame, pad/resize only).")
    return images, center_crop_boxes


def _unique_glb_path(output_dir: Path, stem: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{stem}.glb"
    if not candidate.exists():
        return str(candidate)
    i = 1
    while True:
        numbered = output_dir / f"{stem}_({i}).glb"
        if not numbered.exists():
            return str(numbered)
        i += 1


def _default_head_mask_dir(resolved_image_folder: str) -> Path:
    p = Path(resolved_image_folder)
    return p.parent.resolve() / f"{p.name}_head_masks"


def _ensure_head_masks(
    paths: list,
    resolved_image_folder: str,
    head_mask_mode: str,
) -> str:
    mask_dir = _default_head_mask_dir(resolved_image_folder)
    basenames = [os.path.basename(p) for p in paths]
    from lingbot_map.vis.head_mask_apply import head_masks_complete

    if head_masks_complete(mask_dir, basenames):
        print(f"Using existing head masks under {mask_dir}")
        return str(mask_dir)
    print(f"Generating head masks under {mask_dir} ...")
    from lingbot_map.utils.head_mask_v3_flat import FlatHeadMaskConfig, run_flat

    ext = Path(paths[0]).suffix.lower() if paths else ".jpeg"
    run_flat(
        FlatHeadMaskConfig(
            image_dir=Path(resolved_image_folder).resolve(),
            output_dir=mask_dir,
            extension=ext,
            mode=head_mask_mode,
        )
    )
    return str(mask_dir)


def _resolve_export_glb_path(export_glb_flag, resolved_media_folder: str) -> str | None:
    if export_glb_flag is None:
        return None
    if export_glb_flag == _EXPORT_GLB_AUTO:
        stem = Path(resolved_media_folder).name
        out_dir = Path(__file__).resolve().parent / "output"
        return _unique_glb_path(out_dir, stem)
    return export_glb_flag


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        pretrained_path=args.pretrained_path,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help=f"GCT checkpoint (default: {_DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=_DEFAULT_PRETRAINED_PATH,
        help=f"DINOv2 backbone weights (default: {_DEFAULT_PRETRAINED_PATH})",
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="windowed", choices=["streaming", "windowed"],
                        help="windowed: overlapping windows (default); streaming: frame-by-frame with KV cache")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Streaming only. Every N-th frame after scale frames is kept as a keyframe. 1 = every frame. "
             "If unset, auto-selected: 1 when num_frames <= 320, else ceil(num_frames / 320).",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head iterative-refinement steps. Default 4; set 1 for faster inference "
                             "(skips 3 refinement passes at a small accuracy cost).")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload per-frame predictions to CPU (default: on). Use --no-offload_to_cpu to keep tensors on GPU.",
    )
    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16, help="Overlap between windows")


    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=None,
        help="After conf filter, keep every N-th point in GLB/viewer export. "
        "Default: auto from input count — 2 if >1000 frames, 3 if >2000 (pass e.g. 1 to force dense). "
        "Auto values assume ~64 GiB RAM + ≥32 GiB swap (see notes.md); marginal: 48 GiB RAM + ~48 GiB swap.",
    )
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument(
        "--mask_head",
        action="store_true",
        help="Pipeline generates head/rig masks next to the input folder (<name>_head_masks) if needed, "
        "then gates depth confidence (no extra paths to pass). "
        "VRAM-heavy vs no mask: plan for ~24GB+ on long runs; without this flag, ~16GB often suffices with default 16_9 or --crop_aspect 16_10.",
    )
    parser.add_argument(
        "--mask_head_mode",
        type=str,
        default="head",
        choices=("head", "head_and_lens"),
        help="Head-mask generation mode when masks are auto-generated (see head_mask_v3_flat).",
    )
    parser.add_argument(
        "--crop_aspect",
        type=str,
        default=argparse.SUPPRESS,
        choices=("none", "square", "16_9", "16_10", "4_3", "5_4"),
        help="Center-crop each input to the largest inscribed rectangle of this aspect (before resize/pad). "
        "Default when omitted: 16_9 (shorter H after pad than square/5_4/4_3 on typical landscape). "
        "Without --mask_head, 16_9 or 16_10 is appropriate for ~16GB VRAM; with --mask_head, prefer 16_9 and more VRAM. "
        "Use none for no center aspect crop (full frame before pad/resize only). "
        "With --mask_head, on-disk masks stay full-frame; viewer aligns via head_center_crops when crop is used. "
        "Aliases: 16_9=16:9, 16_10=16:10, 4_3=4:3, 5_4=5:4.",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="Shorthand for --crop_aspect square.",
    )
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument(
        "--export_preprocessed",
        type=str,
        default=None,
        help="Export preprocessed model-input images (after optional --crop_aspect + resize/pad) as 000000.png, …",
    )
    parser.add_argument(
        "--export_glb",
        nargs="?",
        const=_EXPORT_GLB_AUTO,
        default=_EXPORT_GLB_AUTO,
        metavar="PATH",
        help="GLB export path, or omit PATH for output/<input_folder_name>.glb (default). "
        "Use --no_export_glb to skip.",
    )
    parser.add_argument(
        "--no_export_glb",
        action="store_true",
        default=False,
        help="Skip GLB export (overrides default export).",
    )
    parser.add_argument(
        "--export_frame_stride",
        type=int,
        default=None,
        help="When exporting GLB: project depth to points every N frames (1 = all frames). "
        "Default: auto — 2 if >3000, 3 if >6000, 4 if >9000, … (+1 per extra 3000 frames). "
        "See notes.md for RAM/swap; pass a larger stride if export still gets killed.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        default=False,
        help="Launch interactive viser after inference (default: off; GLB export returns before viewer).",
    )

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"
    if args.square:
        if hasattr(args, "crop_aspect"):
            parser.error("Use either --square or --crop_aspect, not both.")
        args.crop_aspect = "square"
    elif not hasattr(args, "crop_aspect"):
        args.crop_aspect = "16_9"

    _validate_media_args(parser, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    paths, resolved_image_folder = collect_media_paths(
        image_folder=args.image_folder,
        video_path=args.video_path,
        fps=args.fps,
        first_k=args.first_k,
        stride=args.stride,
    )
    if not paths:
        parser.error(
            f"No images found under {resolved_image_folder!r} (check extensions and --first_k / --stride)."
        )

    _apply_auto_glb_decimation(len(paths), args, parser)

    mask_head_dir_for_vis: str | None = None
    if args.mask_head:
        mask_head_dir_for_vis = _ensure_head_masks(
            paths,
            resolved_image_folder,
            args.mask_head_mode,
        )

    crop_for_preprocess = None if args.crop_aspect == "none" else args.crop_aspect
    images, head_center_crops = preprocess_media_paths(
        paths,
        image_size=args.image_size,
        patch_size=args.patch_size,
        crop_aspect=crop_for_preprocess,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    # Pick inference dtype; autocast still runs for the ops that need fp32 (e.g. LayerNorm).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the
    # redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved,
    # no measurable quality change). gct_base._predict_* upcasts inputs to fp32 and
    # runs each head under `autocast(enabled=False)`, so camera/depth/point heads
    # keep fp32 weights automatically.
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: "
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
            print(
                f"Auto-selected --keyframe_interval={args.keyframe_interval} "
                f"(num_frames={num_frames} > 320)."
            )
        else:
            args.keyframe_interval = 1

    if args.mode != "streaming" and args.keyframe_interval != 1:
        print("Warning: --keyframe_interval only applies to --mode streaming. Ignoring it for windowed inference.")
        args.keyframe_interval = 1
    elif args.mode == "streaming" and args.keyframe_interval > 1:
        print(
            f"Keyframe streaming enabled: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                output_device=output_device,
            )

    print(f"Inference done in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(
            f"GPU peak during inference: "
            f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ── Post-process ─────────────────────────────────────────────────────────
    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]  # already CPU
    else:
        images_for_post = images

    predictions, images_cpu = postprocess(predictions, images_for_post)

    # ── Visualize ────────────────────────────────────────────────────────────
    vis_predictions = prepare_for_visualization(predictions, images_cpu)
    vis_predictions["image_paths"] = list(paths)
    # Head masks are generated and cached on disk at full source-frame resolution (before
    # --crop_aspect), while depth_conf / model tensors live in cropped+resized geometry.
    # The model does not see original dimensions; this metadata is only for post-hoc alignment:
    # Viewer / GLB load full-res masks, apply the same (left, top, crop_w, crop_h) as RGB
    # preprocessing, then resize to conf H×W. Without it, scaling a full-frame mask straight
    # to conf would mis-register pixels relative to the cropped inference grid.
    if head_center_crops is not None:
        vis_predictions["head_center_crops"] = head_center_crops

    export_glb_flag = None if args.no_export_glb else args.export_glb
    export_glb_path = _resolve_export_glb_path(export_glb_flag, resolved_image_folder)
    if export_glb_path:
        try:
            from types import SimpleNamespace
            from lingbot_map.vis import PointCloudViewer

            export_path = export_glb_path
            output_dir = os.path.dirname(export_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            viewer = PointCloudViewer.__new__(PointCloudViewer)
            viewer.vis_threshold = args.conf_threshold
            viewer.show_camera = True
            viewer.point_size = args.point_size
            viewer.original_images = []

            # Match the interactive viewer's default behavior: use depth-based
            # unprojection for export. The point-map head is not always
            # available in checkpoints, and falling back to world_points here
            # can silently export a random/untrained branch.
            use_point_map = False
            pc_list, color_list, conf_list, cam_dict = PointCloudViewer._process_pred_dict(
                viewer,
                vis_predictions,
                use_point_map=use_point_map,
                mask_sky=args.mask_sky,
                image_folder=resolved_image_folder,
                sky_mask_dir=args.sky_mask_dir,
                sky_mask_visualization_dir=args.sky_mask_visualization_dir,
                mask_head=args.mask_head,
                mask_head_dir=mask_head_dir_for_vis,
                depth_stride=max(1, args.export_frame_stride),
            )
            viewer.pcs, viewer.all_steps = PointCloudViewer.read_data(
                viewer, pc_list, color_list, conf_list, edge_color_list=None
            )
            viewer.cam_dict = cam_dict

            viewer.glb_output_path = SimpleNamespace(value=export_path)
            viewer.glb_show_cam_checkbox = SimpleNamespace(value=True)
            viewer.glb_cam_scale_slider = SimpleNamespace(value=1.0)
            viewer.glb_frustum_thickness_slider = SimpleNamespace(value=3.0)
            viewer.glb_trajectory_checkbox = SimpleNamespace(value=True)
            viewer.glb_trajectory_radius_slider = SimpleNamespace(value=0.005)
            viewer.glb_mode_dropdown = SimpleNamespace(value="Points")
            viewer.glb_sphere_radius_slider = SimpleNamespace(value=0.005)
            viewer.glb_max_sphere_pts_slider = SimpleNamespace(value=100000)
            viewer.glb_opacity_slider = SimpleNamespace(value=1.0)
            viewer.glb_saturation_slider = SimpleNamespace(value=1.0)
            viewer.glb_brightness_slider = SimpleNamespace(value=1.0)
            viewer.downsample_slider = SimpleNamespace(value=args.downsample_factor)
            viewer.glb_status = SimpleNamespace(value="Ready")

            viewer._export_glb()
            print(f"GLB exported to {export_path}")
        except ImportError:
            print("trimesh not installed. Install with: pip install trimesh")
            print(f"Predictions contain keys: {list(predictions.keys())}")
        return

    if not args.viewer:
        print("Skipping interactive viewer (pass --viewer to open).")
        return

    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=vis_predictions,
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
            mask_head=args.mask_head,
            mask_head_dir=mask_head_dir_for_vis,
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
