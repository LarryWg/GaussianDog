# Setup

GaussianDog was tested on Ubuntu 22.04 with Python 3.10, CUDA 11.8, PyTorch 2.4.0, torchvision 0.19.0, and PyTorch3D 0.7.9. Keep BITE, TRELLIS, Gaussian optimization, and DGE in separate environments when their dependency requirements conflict.

## 1. System packages

Install a CUDA 11.8 development toolkit, Git, a C++ compiler, Ninja, and OpenGL build dependencies. The NVIDIA driver may be newer than the container CUDA toolkit.

The tested cloud machine was an RTX A6000 with 48 GB of VRAM, 32 GB or more of system memory, and 100 GB of workspace storage.

## 2. GaussianDog environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

git clone --depth 1 --branch v0.7.9 https://github.com/facebookresearch/pytorch3d.git external/pytorch3d
pip install external/pytorch3d --no-build-isolation

pip install -r requirements.txt
```

Set the compilation environment before running gsplat or PyTorch3D:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
```

Change `TORCH_CUDA_ARCH_LIST` for a different GPU architecture.

## 3. BITE

Clone the official source at the tested revision:

```bash
git clone https://github.com/runa91/bite_release.git external/bite
git -C external/bite checkout 85fa6b1b126a97b592039b7e16b76e7a44f76ab3
```

Read and accept the [BITE license](https://github.com/runa91/bite_release/blob/master/LICENSE) before downloading its data or checkpoints. BITE is restricted to non-commercial scientific research.

The runner expects this structure inside `external/bite`:

```text
external/bite/
├── checkpoint/
│   ├── barc_normflow_pret/rgbddog_v3_model.pt
│   └── cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0_forrelease_v0/checkpoint.pth.tar
├── data/
├── src/
└── requirements.txt
```

The tested checkpoint and model data were obtained from the author's [Hugging Face Space revision](https://huggingface.co/spaces/runa91/bite_gradio/tree/4b24155a5bf43ea09b2280386e69783f7af6fab8). Do not commit these files.

Install BITE's runtime dependencies in the GaussianDog environment or a compatible isolated environment. Keep the tested numerical pins:

```text
numpy==1.23.5
scipy==1.10.1
kornia==0.4.0
torch==2.4.0
torchvision==0.19.0
pytorch3d==0.7.9
```

The upstream dependency list also includes OpenCV, Matplotlib, trimesh, Chumpy, pycocotools, pymp, openpyxl, dominate, importlib-resources, and FrEIA. Install only the packages required by inference rather than replacing the tested Torch stack with the obsolete upstream pins.

`scripts/bite/compat.py` applies three narrow source compatibility edits for PyTorch3D 0.7.9 and modern Pillow. The original checkout remains attributable to BITE.

## 4. TRELLIS

Follow the official [TRELLIS installation](https://github.com/microsoft/TRELLIS#installation) in its own environment:

```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git external/TRELLIS
```

TRELLIS officially requires Linux, an NVIDIA GPU with at least 16 GB of memory, and a CUDA toolkit for compiled extensions. The official setup defaults to PyTorch 2.4.0 with CUDA 11.8, matching this pipeline.

The proxy script downloads `microsoft/TRELLIS-image-large` at the pinned model revision on first use.

## 5. Optional DGE environment

DGE is only needed for the optional final editing pass. Clone its official source and follow its CUDA 11.8 setup:

```bash
git clone https://github.com/silent-chen/DGE.git external/DGE
```

Run `scripts/smal_pets/dge_refine.py --prepare-only` before enabling diffusion editing. This validates the calibrated view export without spending time on the edit or fit.

## 6. Smoke checks

Run CPU checks first:

```bash
pip install -r requirements-cpu.txt
make check
```

Then validate the CUDA rasterizer:

```bash
python scripts/smal_pets/check_rasterizer.py --output work/rasterizer-check.json
```

BITE inference itself validates checkpoint loading, a CUDA silhouette render with finite gradients, finite image predictions, and a finite first optimization step.
