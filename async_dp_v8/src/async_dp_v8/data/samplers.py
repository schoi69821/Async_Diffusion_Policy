"""Phase-balanced and hard-negative samplers for v8 training."""
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
from typing import Iterator, List
from collections import Counter
import logging

from async_dp_v8.constants import NUM_PHASES

logger = logging.getLogger(__name__)


class PhaseBalancedSampler(Sampler[int]):
    """Oversample underrepresented phases using only phases present in data."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        hard_negative_ratio: float = 0.25,
        boost_rare: float = 1.5,
        seed: int = 42,
    ):
        self.index_df = index_df
        self.hard_negative_ratio = hard_negative_ratio
        self.boost_rare = boost_rare
        self.rng = np.random.RandomState(seed)
        self._build_phase_indices()

    def _build_phase_indices(self):
        # Build indices only for phases that actually exist in data
        self.phase_indices = {}
        counts = Counter(self.index_df["phase"])

        for phase_id in range(NUM_PHASES):
            mask = self.index_df["phase"] == phase_id
            idxs = np.where(mask.to_numpy())[0]
            if len(idxs) > 0:
                self.phase_indices[phase_id] = idxs

        self.active_phases = sorted(self.phase_indices.keys())
        if not self.active_phases:
            logger.warning("No phases found in index — falling back to uniform sampling")
            self.active_phases = []
            return

        max_count = max(counts[p] for p in self.active_phases)

        # Inverse-frequency weights for active phases only
        self.sample_weights = {}
        for phase_id in self.active_phases:
            self.sample_weights[phase_id] = max_count / counts[phase_id]

        # Identify rare phases (< 5% of data) and boost them
        total = len(self.index_df)
        self.hard_phases: List[int] = []
        for phase_id in self.active_phases:
            if counts[phase_id] / total < 0.05:
                self.sample_weights[phase_id] *= self.boost_rare
                self.hard_phases.append(phase_id)

        logger.info(
            f"PhaseBalancedSampler: active={self.active_phases}, "
            f"counts={dict(counts)}, hard_phases={self.hard_phases}"
        )

    def __iter__(self) -> Iterator[int]:
        n = len(self.index_df)

        if not self.active_phases:
            # Fallback: uniform random
            indices = self.rng.permutation(n).tolist()
            return iter(indices)

        indices = []
        weights = self._normalized_weights()

        # Standard balanced sampling
        n_standard = int(n * (1 - self.hard_negative_ratio))
        phases = self.rng.choice(self.active_phases, size=n_standard, p=weights)
        for phase in phases:
            idx = self.rng.choice(self.phase_indices[phase])
            indices.append(int(idx))

        # Hard negative mining: oversample rare phases
        n_hard = n - n_standard
        if self.hard_phases:
            hard_choices = self.rng.choice(self.hard_phases, size=n_hard)
            for phase in hard_choices:
                idx = self.rng.choice(self.phase_indices[phase])
                indices.append(int(idx))
        else:
            # No rare phases — fill with balanced sampling
            phases = self.rng.choice(self.active_phases, size=n_hard, p=weights)
            for phase in phases:
                idx = self.rng.choice(self.phase_indices[phase])
                indices.append(int(idx))

        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.index_df)

    def _normalized_weights(self) -> np.ndarray:
        w = np.array([self.sample_weights[p] for p in self.active_phases])
        return w / w.sum()
