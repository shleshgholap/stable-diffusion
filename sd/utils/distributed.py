"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0)) if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    world_size = get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)
