"""Custom collate function for v8 dataset."""
from typing import Dict, Any, List
import torch


def v8_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch of samples, handling variable-length and non-tensor fields."""
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            collated[key] = values
        else:
            collated[key] = values

    return collated
