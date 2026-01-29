"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: Optional[str] = None, log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_name = log_file if log_file else "train.log"
        file_handler = logging.FileHandler(log_dir / file_name)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class MetricLogger:
    def __init__(self, delimiter: str = "  "):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.meters:
                self.meters[key] = SmoothedValue()
            self.meters[key].update(value)
    
    def __getitem__(self, key: str):
        return self.meters[key]
    
    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())
    
    def get_avg(self, key: str) -> float:
        return self.meters[key].avg if key in self.meters else 0.0


class SmoothedValue:
    def __init__(self, window_size: int = 20):
        self.values = []
        self.window_size = window_size
        self.total = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        self.total += value * n
        self.count += n
    
    @property
    def median(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        return sorted_values[len(sorted_values) // 2]
    
    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    @property
    def global_avg(self) -> float:
        return self.total / max(1, self.count)
    
    def __str__(self):
        return f"{self.avg:.4f} (global: {self.global_avg:.4f})"
