# LingBot-MAP demo — operational notes

This file summarizes behaviors and flags that are easy to miss from `--help` alone (especially around memory, crops, masks, and export).

## Hardware (VRAM, RAM, swap)

### GPU (VRAM)

- **Without `--mask_head`:** **~16 GB VRAM** is usually enough for **normal inference**, as long as you keep a **wide** center crop — use **`--crop_aspect 16_9`** (default) or **`--crop_aspect 16_10`**. Taller aspects (`square`, `5_4`, `4_3`, …) raise per-frame `H` after pad and are more likely to OOM on 16 GB.
- **With `--mask_head`:** treat **~24 GB VRAM** as the practical floor for long / full-batch runs; still prefer **`16_9`** (default). Extra VRAM pressure comes from the full **(S, 3, H, W)** batch on GPU plus the rest of the model.

### Host RAM / swap (long jobs, GLB, masks)

For **long** runs (thousands of frames), **`--mask_head`**, default **GLB export**, and head-mask loading on CPU:

- **Marginal minimum:** **24 GB VRAM** and **48 GiB system RAM**, plus about **48 GiB swap** — enough to **barely** finish in many cases, but expect **heavy swap**, long runtimes, and risk of the OS OOM killer if anything else is running.
- **Recommended:** **64 GiB RAM** (or more) and **at least 32 GiB swap** as headroom for export, predictions on CPU, and spikes during `trimesh` / point concatenation.

Auto **`--downsample_factor`** / **`--export_frame_stride`** defaults are calibrated for roughly the **recommended** tier. On the **marginal** tier, set those flags **more aggressively** yourself, or reduce frames (`--first_k`, `--stride`), skip GLB (`--no_export_glb`). **`--crop_aspect`** defaults to **`16_9`** (omit the flag); use **`--crop_aspect none`** to disable center aspect crop (legacy full-frame-before-pad behavior).

## Inputs, paths, and frame count

- Use a **real** `--image_folder` or `--video_path` on disk; the demo **validates** existence early (documentation placeholders like `/path/to/images` will fail fast).
- **`--first_k`** and **`--stride`** shrink the path list **before** loading tensors and **before** auto GLB decimation runs — thresholds such as “>3000 frames” use **`len(paths)` after** these filters, not the raw folder size.
- **`--video_path`**: frames are extracted to a sibling `*_frames` directory first; head masks use that resolved folder.

## VRAM, `--mask_head`, and `--crop_aspect`

- **No mask:** full image batch on GPU is still the main cost; **`16_9`** (default) or **`16_10`** keeps **`H`** lower after pad on typical landscape — enough for **~16 GB** cards in normal use. Avoid tall aspects on small VRAM.
- **With `--mask_head`:** same batch pressure plus a heavier footprint overall; **~24 GB** is the practical target for long runs. **`--mask_head`** keeps a full **(S, 3, H, W)** tensor on **GPU** (`images.to(device)`), plus weights and attention/KV.
- **Default `--crop_aspect` is `16_9`** (omit the flag). After `mode="pad"`, the long side is tied to `--image_size` (e.g. 518). Use **`--crop_aspect none`** only if you need no center crop (uses more pixels per frame; worse on 16 GB).
- **`--offload_to_cpu`** moves **prediction** tensors to CPU during/after inference; it does **not** stop the **full input stack** from living on GPU before/during the forward pass.
- **GLB export / viewer**: very long `S` and dense points can exhaust **system RAM** (`Killed` by the OOM killer). Mitigate with **`--export_frame_stride`** (fewer frames contribute points) and **`--downsample_factor`** (sparser points per frame).
- **Auto decimation (when you omit these flags)** after the input list is built — aimed at the **recommended** host profile above; on **48 GiB RAM + 48 GiB swap** or smaller, pass **more aggressive** values explicitly (larger **`--downsample_factor`** and/or larger **`--export_frame_stride`**):
  - **`--downsample_factor`**: 2 if **>1000** frames, 3 if **>2000**.
  - **`--export_frame_stride`**: 1 if **≤3000**; then **2** if **>3000**, **3** if **>6000**, **4** if **>9000**, … (+1 per additional 3000 frames). Override anytime (e.g. `--downsample_factor 1 --export_frame_stride 1`).
- **Meaning (for tuning)**:
  - Larger **`--export_frame_stride`** → fewer **time samples** contribute 3D points (thinner temporal coverage).
  - Larger **`--downsample_factor`** → fewer points **per projected frame** (thinner density, same frame selection).

## `--crop_aspect` and `--square`

- Choices: **`none`** (no center crop), **`square`**, **`16_9`**, **`16_10`**, **`4_3`**, **`5_4`**. **Default when omitted: `16_9`**. **`--square`** is shorthand for **`--crop_aspect square`** (do not pass both `--square` and `--crop_aspect`).
- Crop is the **largest centered rectangle** of that aspect inside each source image; it only changes **in-memory** preprocessing and what the model receives. **1:1** sources with a **non-square** aspect still **lose** horizontal or vertical strips.
- **Pipeline order with `--mask_head`**: head masks are ensured on **full** source frames **first**, then RGB is aspect-cropped and resized — so on-disk `<input>_head_masks` stay **full-frame**; Viewer/GLB use **`head_center_crops`** to match **`depth_conf`**.

## GLB default and dependencies

- By default the demo **exports GLB** (unless **`--no_export_glb`**) and then **returns** — it does **not** start **`--viewer`** in the same process. Use **`--no_export_glb --viewer`** for interactive only, or run twice.
- Export needs **`trimesh`** (`pip install trimesh` if import fails).

## Head masks vs. crop geometry

- Head masks are generated/cached on **full-resolution source frames** (before `--crop_aspect`). The model sees **cropped + resized** RGB; **`head_center_crops`** in the vis/GLB path aligns disk masks to **`depth_conf`** geometry. Details are in comments near `vis_predictions["head_center_crops"]` in `demo.py`.
- The viewer still accepts legacy **`head_square_crops`** (5-tuple) if present; new runs use **6-tuple** **`head_center_crops`** `(left, top, crop_w, crop_h, orig_w, orig_h)`.

## Input aspect ratio (portrait / vertical)

- **Portrait (tall / vertical) inputs are not recommended.** With `mode="pad"`, the **larger** dimension is driven toward `image_size`; on skinny vertical frames this can make the **post-pad `H`×`W`** awkward and **increase per-frame pixel count or padding** compared with well-behaved landscape crops, which hurts memory and can surprise geometry.
- **Prefer** inputs that are already **square** or **moderately wide** (close to **16:9** or similar). If you only have portrait material, **pre-crop or letterbox** to a square or near-landscape canvas before running the demo when possible.

## Quick reference

**~16 GB VRAM, no head mask** (default crop is already `16_9`; optional explicit `16_10`):

```bash
python demo.py --image_folder /path/to/frames
```

**~24 GB VRAM + head masks** — pair with **RAM / swap** in **Hardware** for long jobs:

```bash
python demo.py --image_folder /path/to/frames --mask_head
```

Adjust `--first_k`, `--stride`, `--export_frame_stride`, and `--downsample_factor` if you still hit GPU or RAM limits (auto defaults may not be enough on huge runs).

Optional **`--export_preprocessed`** (see **`python demo.py --help`**) dumps preprocessed frames as PNG; default reconstruction does not need it.
