#!/usr/bin/env python3
# One-shot download for LingBot-Map GCT weights (HuggingFace) and DINOv2 ViT-L/14 reg (Meta).
# Comments and CLI messages in English.

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Default checkpoints dir: <repo>/lingbot-map/checkpoints
_REPO_ROOT = Path(__file__).resolve().parent.parent

GCT_REPO = "robbyant/lingbot-map"
# Official Meta DINOv2 register checkpoint (saves as the filename expected by demo.py).
DINO_VITL14_REG_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
)
DINO_LOCAL_NAME = "dinov2_vitl14_reg_official.pth"
GCT_FILES_FULL = ("lingbot-map.pt", "lingbot-map-long.pt", "lingbot-map-stage1.pt")
GCT_FILES_MINIMAL = ("lingbot-map-long.pt",)


def _import_hf():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e
    return hf_hub_download


def _download_dino(
    out_path: Path,
    url: str,
    force: bool,
) -> None:
    from tqdm import tqdm

    if out_path.is_file() and not force:
        print(f"Skip (exists): {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")
    part.unlink(missing_ok=True)

    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.getheader("Content-Length") or 0)
            with open(part, "wb") as f, tqdm(
                desc=DINO_LOCAL_NAME,
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
            ) as bar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        part.replace(out_path)
    except urllib.error.URLError as e:
        part.unlink(missing_ok=True)
        raise SystemExit(f"Failed to download DINOv2 weights: {e}") from e
    print(f"Saved: {out_path}")


def _download_gct_files(
    hf_hub_download,
    out_dir: Path,
    files: tuple[str, ...],
    repo_id: str,
    token: str | None,
    force: bool,
) -> None:
    for name in files:
        dest = out_dir / name
        if dest.is_file() and not force:
            print(f"Skip (exists): {dest}")
            continue
        p = hf_hub_download(
            repo_id=repo_id,
            filename=name,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Saved: {p}")


def main() -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    p = argparse.ArgumentParser(
        description="Download GCT weights from HuggingFace and DINOv2 ViT-L/14 reg from Meta."
    )
    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=_REPO_ROOT / "checkpoints",
        help="Directory to store .pt and DINOv2 (default: <repo>/checkpoints)",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="Only download lingbot-map-long.pt + DINOv2 (matches demo.py defaults)",
    )
    p.add_argument("--gct-only", action="store_true", help="Skip DINOv2")
    p.add_argument("--dino-only", action="store_true", help="Skip GCT files from HuggingFace")
    p.add_argument(
        "--repo-id",
        default=GCT_REPO,
        help=f"HuggingFace model id (default: {GCT_REPO})",
    )
    p.add_argument(
        "--dino-url",
        default=DINO_VITL14_REG_URL,
        help="URL for DINOv2 Vit-L/14 register backbone",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (optional; or set HF_TOKEN)",
    )
    args = p.parse_args()

    if args.gct_only and args.dino_only:
        print("Nothing to do: both --gct-only and --dino-only are set.", file=sys.stderr)
        return 1

    out = args.checkpoints_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    gct_list = GCT_FILES_MINIMAL if args.minimal else GCT_FILES_FULL

    if not args.dino_only:
        hf_hub_download = _import_hf()
        _download_gct_files(
            hf_hub_download,
            out,
            gct_list,
            args.repo_id,
            args.token,
            args.force,
        )
    if not args.gct_only:
        _download_dino(out / DINO_LOCAL_NAME, args.dino_url, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
