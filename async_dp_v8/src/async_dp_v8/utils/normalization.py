"""Normalization utilities for dataset statistics."""
import json
import numpy as np
from pathlib import Path
from typing import Dict


class Normalizer:
    def __init__(self, stats: Dict[str, Dict[str, np.ndarray]]):
        self.stats = stats

    def normalize(self, key: str, data: np.ndarray) -> np.ndarray:
        if key not in self.stats:
            return data
        mean = self.stats[key]["mean"]
        std = self.stats[key]["std"]
        std = np.where(std < 1e-6, 1.0, std)
        return (data - mean) / std

    def denormalize(self, key: str, data: np.ndarray) -> np.ndarray:
        if key not in self.stats:
            return data
        mean = self.stats[key]["mean"]
        std = self.stats[key]["std"]
        std = np.where(std < 1e-6, 1.0, std)
        return data * std + mean

    @classmethod
    def from_json(cls, path: str) -> "Normalizer":
        with open(path, "r") as f:
            raw = json.load(f)
        stats = {}
        for key, val in raw.items():
            stats[key] = {
                "mean": np.array(val["mean"], dtype=np.float32),
                "std": np.array(val["std"], dtype=np.float32),
            }
        return cls(stats)

    def to_json(self, path: str):
        raw = {}
        for key, val in self.stats.items():
            raw[key] = {
                "mean": val["mean"].tolist(),
                "std": val["std"].tolist(),
            }
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
