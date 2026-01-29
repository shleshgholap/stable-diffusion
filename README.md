# Stable Diffusion from Scratch

End-to-end implementation of Stable Diffusion for image generation, built from scratch in PyTorch.

## Features

- **Custom VAE**: Variational Autoencoder for encoding images to latent space
- **U-Net Denoiser**: Time-conditioned U-Net with cross-attention for text conditioning
- **DDPM/DDIM Schedulers**: Noise scheduling for forward diffusion and reverse sampling
- **CLIP Text Encoder**: Integration with CLIP for text-to-image conditioning
- **Classifier-Free Guidance (CFG)**: Improved sample quality through guidance
- **Multiple Pipelines**: Text-to-image, image-to-image, and inpainting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stable-diffusion.git
cd stable-diffusion

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For training support
pip install -e ".[train]"

# For development
pip install -e ".[all]"
```

## Quick Start

### Text-to-Image Generation

```python
from sd import Text2ImagePipeline

# Load pipeline with pretrained weights
pipeline = Text2ImagePipeline.from_pretrained("path/to/checkpoint")
pipeline.to("cuda")

# Generate image
image = pipeline(
    prompt="a photo of a cat wearing a hat",
    num_inference_steps=50,
    guidance_scale=7.5,
)
image.save("output.png")
```

### Image-to-Image

```python
from sd import Image2ImagePipeline
from sd.utils import load_image

pipeline = Image2ImagePipeline.from_pretrained("path/to/checkpoint")
pipeline.to("cuda")

# Load input image
init_image = load_image("input.png", size=(512, 512))

# Transform image
image = pipeline(
    prompt="a watercolor painting",
    image=init_image,
    strength=0.75,
    num_inference_steps=50,
    guidance_scale=7.5,
)
image.save("output.png")
```

### Inpainting

```python
from sd import InpaintPipeline
from sd.utils import load_image

pipeline = InpaintPipeline.from_pretrained("path/to/checkpoint")
pipeline.to("cuda")

# Load image and mask
image = load_image("image.png", size=(512, 512))
mask = load_image("mask.png", size=(512, 512), mode="L")

# Inpaint
result = pipeline(
    prompt="a red flower",
    image=image,
    mask=mask,
    num_inference_steps=50,
    guidance_scale=7.5,
)
result.save("inpainted.png")
```

## Command Line Interface

### Text-to-Image

```bash
python scripts/infer_text2img.py \
    --checkpoint path/to/checkpoint \
    --prompt "a beautiful sunset over mountains" \
    --output outputs/sunset.png \
    --steps 50 \
    --guidance-scale 7.5 \
    --seed 42
```

### Image-to-Image

```bash
python scripts/infer_img2img.py \
    --checkpoint path/to/checkpoint \
    --image input.png \
    --prompt "oil painting style" \
    --strength 0.7 \
    --output outputs/styled.png
```

### Inpainting

```bash
python scripts/infer_inpaint.py \
    --checkpoint path/to/checkpoint \
    --image input.png \
    --mask mask.png \
    --prompt "a golden retriever" \
    --output outputs/inpainted.png
```

## Training

### Toy Training (Pixel-space DDPM)

Train a small diffusion model on CIFAR-10 to validate the training pipeline:

```bash
python scripts/train_unet_pixel_ddpm.py \
    --config configs/toy/train_pixel_ddpm.yaml \
    --output-dir logs/toy_ddpm
```

### Toy Training (Latent Diffusion with Text)

Train a small latent diffusion model with text conditioning:

```bash
python scripts/train_latent_diffusion.py \
    --config configs/toy/train_latent_diffusion.yaml \
    --output-dir logs/toy_ldm
```

### SD-Scale Training

For full Stable Diffusion scale training (requires multi-GPU setup):

```bash
# Using accelerate for distributed training
accelerate launch --multi_gpu --num_processes 8 \
    scripts/train_latent_diffusion.py \
    --config configs/sdscale/train_sd.yaml \
    --output-dir logs/sd_full
```

**Requirements for SD-scale training:**
- 8x A100 (40GB) or equivalent
- ~2TB storage for dataset (e.g., LAION-5B subset)
- Training time: 2-4 weeks for convergence

## Loading Pretrained Weights

### From Diffusers Format

```python
from sd.checkpoints import convert_diffusers_checkpoint

# Convert diffusers checkpoint to our format
convert_diffusers_checkpoint(
    diffusers_path="runwayml/stable-diffusion-v1-5",
    output_path="checkpoints/sd15.safetensors"
)
```

### From Safetensors

```python
from sd import Text2ImagePipeline

# Load directly from safetensors
pipeline = Text2ImagePipeline.from_pretrained("checkpoints/sd15.safetensors")
```

## Project Structure

```
stable-diffusion/
├── sd/                         # Main package
│   ├── models/                 # Neural network architectures
│   │   ├── vae.py             # Variational Autoencoder
│   │   ├── unet.py            # U-Net denoiser
│   │   └── text_encoder.py    # CLIP text encoder wrapper
│   ├── schedulers/            # Noise schedulers
│   │   ├── ddpm.py            # DDPM scheduler
│   │   └── ddim.py            # DDIM scheduler
│   ├── pipelines/             # Generation pipelines
│   │   ├── text2img.py        # Text-to-image
│   │   ├── img2img.py         # Image-to-image
│   │   └── inpaint.py         # Inpainting
│   ├── guidance/              # Guidance utilities
│   │   └── cfg.py             # Classifier-free guidance
│   ├── checkpoints/           # Checkpoint loading
│   │   ├── loader.py          # Load checkpoints
│   │   └── converter.py       # Convert between formats
│   └── utils/                 # Utilities
│       ├── device.py          # Device/seed management
│       ├── ema.py             # EMA for training
│       ├── image_io.py        # Image I/O
│       └── logging.py         # Logging utilities
├── configs/                   # Configuration files
│   ├── toy/                   # Toy training configs
│   ├── sdscale/              # Full-scale training configs
│   └── inference/            # Inference configs
├── scripts/                   # CLI scripts
└── notebooks/                 # Demo notebooks
```

## Architecture Overview

```
Text Prompt → Tokenizer → CLIP Text Encoder → Text Embeddings
                                                    ↓
Random Noise → [Scheduler] → Timesteps →        U-Net      → Predicted Noise
                                          ↑       ↓               ↓
                                    Latents ← [Scheduler] ← Noise Update
                                       ↓
                              VAE Decoder → Output Image
```

### Key Components

1. **VAE (Variational Autoencoder)**
   - Encoder: Compresses 512x512 RGB images to 64x64 latent representations
   - Decoder: Reconstructs images from latents
   - Scaling factor: 0.18215 (for SD 1.x)

2. **U-Net Denoiser**
   - Input: Noisy latents + timestep + text embeddings
   - Architecture: ResNet blocks + self-attention + cross-attention
   - Output: Predicted noise (or v-prediction)

3. **DDPM Scheduler**
   - Forward process: q(x_t | x_0) adds Gaussian noise
   - Reverse process: p(x_{t-1} | x_t) denoises iteratively
   - Supports linear and scaled-linear beta schedules

4. **Classifier-Free Guidance**
   - Trains with random prompt dropout
   - Inference: ε = ε_uncond + s * (ε_cond - ε_uncond)
   - Typical guidance scale: 7.0-8.5

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

## License

MIT License - see [LICENSE](LICENSE) for details.
