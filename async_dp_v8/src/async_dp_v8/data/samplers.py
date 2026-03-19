"""Phase-balanced and hard-negative samplers for v8 training."""
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from typing import Iterator, Optional
from collections import Counter

from async_dp_v8.constants import NUM_PHASES, PHASE_CLOSE, PHASE_LIFT


class PhaseBalancedSampler(Sampler[int]):
    """Oversample underrepresented phases (especially close/lift)."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        hard_negative_ratio: float = 0.25,
        seed: int = 42,
    ):
        self.index_df = index_df
        self.hard_negative_ratio = hard_negative_ratio
        self.rng = np.random.RandomState(seed)
        self._build_phase_indices()

    def _build_phase_indices(self):
        self.phase_indices = {}
        for phase_id in range(NUM_PHASES):
            mask = self.index_df["phase"] == phase_id
            self.phase_indices[phase_id] = np.where(mask.to_numpy())[0]

        counts = Counter(self.index_df["phase"])
        max_count = max(counts.values()) if counts else 1
        self.sample_weights = {}
        for phase_id in range(NUM_PHASES):
            c = counts.get(phase_id, 1)
            self.sample_weights[phase_id] = max_count / c

        # Extra weight for close and lift
        self.sample_weights[PHASE_CLOSE] *= 1.5
        self.sample_weights[PHASE_LIFT] *= 1.5

    def __iter__(self) -> Iterator[int]:
        n = len(self.index_df)
        indices = []

        # Standard balanced sampling
        n_standard = int(n * (1 - self.hard_negative_ratio))
        for _ in range(n_standard):
            phase = self.rng.choice(
                NUM_PHASES,
                p=self._normalized_weights(),
            )
            if len(self.phase_indices[phase]) > 0:
                idx = self.rng.choice(self.phase_indices[phase])
                indices.append(int(idx))

        # Hard negative mining: oversample close/lift
        n_hard = n - n_standard
        hard_phases = [PHASE_CLOSE, PHASE_LIFT]
        for _ in range(n_hard):
            phase = self.rng.choice(hard_phases)
            if len(self.phase_indices[phase]) > 0:
                idx = self.rng.choice(self.phase_indices[phase])
                indices.append(int(idx))

        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.index_df)

    def _normalized_weights(self) -> np.ndarray:
        w = np.array([self.sample_weights.get(i, 1.0) for i in range(NUM_PHASES)])
        return w / w.sum()
