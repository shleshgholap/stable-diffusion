"""Dataset implementations for training diffusion models."""

import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image


class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 512, center_crop: bool = True, random_flip: bool = True, tokenizer: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        
        transform_list = [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS)]
        transform_list.append(transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size))
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        self.transform = transforms.Compose(transform_list)
        self.samples = self._find_samples()
    
    def _find_samples(self):
        samples = []
        metadata_path = self.data_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            for filename, caption in metadata.items():
                image_path = self.data_dir / filename
                if image_path.exists():
                    samples.append((image_path, caption))
        else:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for image_path in self.data_dir.rglob(ext):
                    caption = self._find_caption(image_path)
                    samples.append((image_path, caption))
        return samples
    
    def _find_caption(self, image_path: Path) -> str:
        for ext in [".txt", ".caption"]:
            caption_path = image_path.with_suffix(ext)
            if caption_path.exists():
                with open(caption_path) as f:
                    return f.read().strip()
        return image_path.stem.replace("_", " ").replace("-", " ")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        result = {"image": image, "caption": caption}
        if self.tokenizer is not None:
            result["tokens"] = self.tokenizer(caption)
        return result


class WebDatasetWrapper(IterableDataset):
    def __init__(self, shards: str, image_size: int = 512, image_key: str = "jpg", caption_key: str = "txt",
                 shuffle_buffer: int = 10000, center_crop: bool = True, random_flip: bool = True):
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError("webdataset not installed. Run: pip install webdataset")
        
        self.shards = shards
        
        transform_list = [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS)]
        transform_list.append(transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size))
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        self.transform = transforms.Compose(transform_list)
        self.dataset = (
            wds.WebDataset(shards, shardshuffle=True)
            .shuffle(shuffle_buffer)
            .decode("pil")
            .rename(image=image_key, caption=caption_key)
            .map_dict(image=self.transform)
        )
    
    def __iter__(self):
        return iter(self.dataset)


def create_dataloader(config: Dict[str, Any], tokenizer: Optional[Callable] = None, distributed: bool = False, rank: int = 0, world_size: int = 1) -> DataLoader:
    data_type = config.get("type", "folder")
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)
    image_size = config.get("image_size", 512)
    
    if data_type == "webdataset":
        dataset = WebDatasetWrapper(
            shards=config["train_shards"], image_size=image_size,
            image_key=config.get("image_key", "jpg"), caption_key=config.get("caption_key", "txt"),
            shuffle_buffer=config.get("shuffle_buffer", 10000),
            center_crop=config.get("center_crop", True), random_flip=config.get("random_flip", True),
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    dataset = ImageCaptionDataset(
        data_dir=config["data_dir"], image_size=image_size,
        center_crop=config.get("center_crop", True), random_flip=config.get("random_flip", True), tokenizer=tokenizer,
    )
    
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                     num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    result = {"images": images, "captions": captions}
    if "tokens" in batch[0]:
        result["tokens"] = {key: torch.stack([item["tokens"][key] for item in batch]) for key in batch[0]["tokens"]}
    return result
