from setuptools import setup, find_packages

setup(
    name="stable-diffusion-scratch",
    version="0.1.0",
    description="End-to-end implementation of Stable Diffusion from scratch",
    author="Shlesh Gholap",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "einops>=0.6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "safetensors>=0.3.0",
        "omegaconf>=2.3.0",
    ],
    extras_require={
        "train": [
            "accelerate>=0.20.0",
            "tensorboard>=2.13.0",
            "webdataset>=0.2.0",
        ],
        "dev": [
            "matplotlib>=3.7.0",
        ],
    },
)
