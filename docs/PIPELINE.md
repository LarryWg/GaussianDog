# Pipeline

The commands below assume the repository root is the working directory and the CUDA environment from [SETUP.md](SETUP.md) is active.

```bash
export BITE_SOURCE="$PWD/external/bite"
export TRELLIS_SOURCE="$PWD/external/TRELLIS"
mkdir -p work
```

## 1. Prepare the photograph

Use one RGB image containing one centered dog. Keep the paws, ears, and tail visible where possible. Avoid heavy crops and strong perspective distortion.

For a seated or lying dog, also create a standing reference that preserves the same head, ears, markings, body proportions, and coat. This is a manual generative image step in the current pipeline. Do not use the standing reference for the anatomical fit. BITE should fit the original photograph.

## 2. Fit BITE and export canonical D-SMAL

```bash
python scripts/bite/run.py \
  --bite-source "$BITE_SOURCE" \
  --image input.png \
  --output work/bite-fit \
  --steps 301
```

Do not continue unless `gates.json` reports successful checkpoint loading, CUDA rasterization, image inference, and a finite optimization step.

Inspect:

- `fit-overlay.png` for the original seated fit
- `fit-silhouette.png` for body coverage
- `canonical-preview.png` for the standing anatomy
- `canonical.obj` for paws, body proportions, ears, and tail

`canonical.npz` preserves shape coefficients, limb scales, full vertex offsets, identity canonical pose, joints, topology, and the coordinate transform.

## 3. Generate the standing proxy

Run TRELLIS using the standing reference:

```bash
python scripts/trellis_proxy.py \
  --source "$TRELLIS_SOURCE" \
  --image standing-reference.png \
  --output work/proxy
```

This creates a Gaussian PLY, structural OBJ, optional GLB, processed input, and provenance record. Inspect the proxy from front, side, rear, and three-quarter views before training.

## 4. Initialize alignment

The default initializer tests upright rotations, scale, translation, and trimmed ICP. Coat thickness and a straight generated tail can bias a whole-surface match. For difficult dogs, provide `work/alignment-init.json` with reviewed anatomical correspondences.

Supported keys include rotation, scale, translation, `joint_rotvec`, and `limb_values`. Paw anchors and a floor constraint are preferred over matching paw tips to nearby shanks.

Run a short fit first:

```bash
python scripts/smal_pets/train.py \
  --bite-source "$BITE_SOURCE" \
  --bite-fit work/bite-fit/canonical.npz \
  --proxy-ply work/proxy/proxy.ply \
  --proxy-mesh work/proxy/proxy.obj \
  --output work/training \
  --stop-after 100
```

Add `--initialization work/alignment-init.json` when using reviewed alignment values. Inspect `initial_alignment.png` and `initial_alignment.obj` before the long optimization.

## 5. Train bound and free Gaussians

Resume the exact saved state:

```bash
python scripts/smal_pets/train.py \
  --bite-source "$BITE_SOURCE" \
  --bite-fit work/bite-fit/canonical.npz \
  --proxy-ply work/proxy/proxy.ply \
  --proxy-mesh work/proxy/proxy.obj \
  --output work/training \
  --resume work/training/checkpoint.pt
```

The default schedule uses:

| Stage | Steps | Purpose |
|---|---:|---|
| Bound | 15,000 | Fit dog shape and appearance with four surface Gaussians per face |
| Free | 25,000 | Release positions, rotations, and scales to recover coat detail |
| DGE | 1,000 | Optional multi-view appearance edit over 20 views |

Training uses 96 calibrated 512 by 512 views. Eight interleaved views are held out. The camera radius is 2 and vertical field of view is 40 degrees.

Adaptive density control runs every 100 free-stage steps during the first 15,000 free iterations. The default ceiling is 150,000 Gaussians.

## 6. Optional DGE refinement

Prepare and validate the calibrated export:

```bash
python scripts/smal_pets/dge_refine.py \
  --training work/training \
  --dge-source external/DGE \
  --output work/training/dge \
  --prepare-only
```

Remove `--prepare-only` to run the full edit. Compare the output against the original dog from several angles. More refinement can reduce identity by softening markings, eyes, or muzzle details. An edited PLY should only replace the trained PLY after visual acceptance.

## 7. Export animation and bindings

```bash
python scripts/smal_pets/export.py \
  --bite-source "$BITE_SOURCE" \
  --bite-fit work/bite-fit/canonical.npz \
  --checkpoint work/training/checkpoint.pt \
  --output work/export
```

Add `--ply path/to/accepted-edit.ply` to bind a visually accepted DGE result.

The exporter produces:

- Progressive Gaussian PLY files
- Rest vertices and triangle topology
- Ten face IDs and normalized weights per Gaussian
- Full baked mesh motion at 30 FPS
- A 35-joint GLB for inspection and playback timing
- A mouth landmark and measured gait cadence
- Numerical checks and training provenance

The default actions are Idle, Walk, Run, Sit, Jump, Bark, Paw, Spin, Play bow, Sniff, Dig, and Wag.

## 8. Acceptance checklist

- The fitted silhouette covers the dog without swallowing large background regions.
- Paws, muzzle, ears, torso length, and tail remain plausible in the canonical pose.
- Front, side, rear, and three-quarter views preserve the dog's markings.
- Python deformation passes identity, rigid motion, stretch, articulation, and degenerate-face checks.
- Gaussian weight sums stay within tolerance of one.
- Baked motion remains above the floor and within paw-contact limits.
- Density files contain real trained rows without padding or duplication.
- The exported package records actual completed step counts.

A shortened run must not be described as a completed SMAL-pets reproduction.
