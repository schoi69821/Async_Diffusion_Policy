"""AsyncDP v8 dataset loader."""
from typing import Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import logging

from async_dp_v8.constants import (
    OBS_HORIZON, PRED_HORIZON, NUM_JOINTS, NUM_MOTORS,
    IMAGE_SIZE, CROP_SIZE,
)
from async_dp_v8.utils.normalization import Normalizer

logger = logging.getLogger(__name__)


class AsyncDPv8Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        index_df: pd.DataFrame,
        obs_horizon: int = OBS_HORIZON,
        pred_horizon: int = PRED_HORIZON,
        image_size: tuple = IMAGE_SIZE,
        crop_size: tuple = CROP_SIZE,
        use_crop: bool = True,
        transform=None,
        stats_path: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.df = index_df.reset_index(drop=True)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_size = image_size
        self.crop_size = crop_size
        self.use_crop = use_crop
        self.transform = transform

        # Normalization
        self.normalizer = None
        if stats_path and Path(stats_path).exists():
            self.normalizer = Normalizer.from_json(stats_path)
            logger.info(f"Loaded normalization stats from {stats_path}")

        # Episode cache with LRU limit
        self._episode_cache: Dict[str, pd.DataFrame] = {}
        self._cache_order: list = []
        self._max_cache_size: int = 50

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        episode_id = row["episode_id"]
        frame_idx = int(row["frame_idx"])

        ep_df = self._get_episode(episode_id)
        obs = self._load_obs_window(ep_df, frame_idx)
        targets = self._load_future_targets(ep_df, frame_idx)

        batch = {
            "obs_image_wrist": obs["image_wrist"],
            "obs_image_crop": obs["image_crop"],
            "obs_qpos": obs["qpos"],
            "obs_qvel": obs["qvel"],
            "obs_ee_pose": obs["ee_pose"],
            "obs_gripper": obs["gripper"],
            "obs_current": obs["current"],
            "obs_pwm": obs["pwm"],
            "phase_curr": obs["phase"][-1],
            "contact_curr": obs["contact"][-1],
            "target_phase_next": targets["phase_next"],
            "target_grip_token": targets["grip_token"],
            "target_contact": targets["contact"],
            "target_arm_chunk": targets["arm_chunk"],
            "mask_valid": targets["mask"],
        }
        return batch

    def _get_episode(self, episode_id: str) -> pd.DataFrame:
        if episode_id not in self._episode_cache:
            path = self.data_dir / f"{episode_id}.parquet"
            self._episode_cache[episode_id] = pd.read_parquet(path)
            self._cache_order.append(episode_id)
            # Evict oldest if cache is full
            while len(self._cache_order) > self._max_cache_size:
                oldest = self._cache_order.pop(0)
                self._episode_cache.pop(oldest, None)
        return self._episode_cache[episode_id]

    def _normalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using stored stats if available."""
        if self.normalizer is None:
            return tensor
        np_val = tensor.numpy()
        np_normed = self.normalizer.normalize(key, np_val)
        return torch.from_numpy(np_normed).float()

    def _load_obs_window(self, ep_df: pd.DataFrame, frame_idx: int) -> Dict[str, torch.Tensor]:
        T = self.obs_horizon
        start = max(0, frame_idx - T + 1)
        indices = list(range(start, frame_idx + 1))
        # Pad at beginning if needed
        while len(indices) < T:
            indices.insert(0, indices[0])

        images_wrist = []
        images_crop = []
        qpos_list = []
        qvel_list = []
        ee_pose_list = []
        gripper_list = []
        current_list = []
        pwm_list = []
        phase_list = []
        contact_list = []

        for fi in indices:
            row = ep_df.iloc[fi]
            # Image loading
            img = self._load_image(row.get("image_path", ""), self.image_size)
            images_wrist.append(img)

            if self.use_crop:
                crop = self._load_image(row.get("crop_path", ""), self.crop_size)
                images_crop.append(crop)
            else:
                images_crop.append(torch.zeros(3, *self.crop_size))

            qpos_list.append(self._to_tensor(row, "qpos", NUM_JOINTS))
            qvel_list.append(self._to_tensor(row, "qvel", NUM_JOINTS))
            ee_pose_list.append(self._to_tensor(row, "ee_pose", 7))
            gripper_list.append(self._to_tensor(row, "gripper", 2))
            current_list.append(self._to_tensor(row, "current", NUM_MOTORS))
            pwm_list.append(self._to_tensor(row, "pwm", NUM_MOTORS))
            phase_list.append(int(row.get("phase", 0)))
            contact_list.append(float(row.get("contact_soft", 0.0)))

        obs = {
            "image_wrist": torch.stack(images_wrist),
            "image_crop": torch.stack(images_crop),
            "qpos": torch.stack(qpos_list),
            "qvel": torch.stack(qvel_list),
            "ee_pose": torch.stack(ee_pose_list),
            "gripper": torch.stack(gripper_list),
            "current": torch.stack(current_list),
            "pwm": torch.stack(pwm_list),
            "phase": torch.tensor(phase_list, dtype=torch.long),
            "contact": torch.tensor(contact_list, dtype=torch.float32),
        }

        # Apply normalization
        obs["qpos"] = self._normalize("qpos", obs["qpos"])
        obs["qvel"] = self._normalize("qvel", obs["qvel"])
        obs["ee_pose"] = self._normalize("ee_pose", obs["ee_pose"])
        obs["gripper"] = self._normalize("gripper", obs["gripper"])
        obs["current"] = self._normalize("current", obs["current"])
        obs["pwm"] = self._normalize("pwm", obs["pwm"])

        if self.transform is not None:
            obs["image_wrist"] = self.transform(obs["image_wrist"])
            obs["image_crop"] = self.transform(obs["image_crop"])

        return obs

    def _load_future_targets(self, ep_df: pd.DataFrame, frame_idx: int) -> Dict[str, torch.Tensor]:
        H = self.pred_horizon
        max_idx = len(ep_df) - 1

        arm_chunks = []
        mask = []
        for h in range(H):
            fi = frame_idx + 1 + h
            if fi <= max_idx:
                row = ep_df.iloc[fi]
                arm_chunks.append(self._to_tensor(row, "action_arm", NUM_JOINTS))
                mask.append(1.0)
            else:
                arm_chunks.append(torch.zeros(NUM_JOINTS))
                mask.append(0.0)

        # Next frame labels
        next_idx = min(frame_idx + 1, max_idx)
        next_row = ep_df.iloc[next_idx]

        arm_chunk = torch.stack(arm_chunks)
        arm_chunk = self._normalize("action_arm", arm_chunk)

        return {
            "arm_chunk": arm_chunk,
            "mask": torch.tensor(mask, dtype=torch.float32),
            "phase_next": torch.tensor(int(next_row.get("phase", 0)), dtype=torch.long),
            "grip_token": torch.tensor(int(next_row.get("grip_token", 1)), dtype=torch.long),
            "contact": torch.tensor(float(next_row.get("contact_hard", 0)), dtype=torch.float32),
        }

    def _load_image(self, path: str, size: tuple) -> torch.Tensor:
        if path and Path(path).exists():
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            if path:
                logger.warning(f"Image not found: {path}")
            img = torch.zeros(3, size[0], size[1])
        return img

    def _to_tensor(self, row, prefix: str, dim: int) -> torch.Tensor:
        cols = [f"{prefix}_{i}" for i in range(dim)]
        available = [c for c in cols if c in row.index]
        if available:
            return torch.tensor([float(row[c]) for c in available], dtype=torch.float32)

        if prefix in row.index:
            val = row[prefix]
            if isinstance(val, (list, np.ndarray)):
                return torch.tensor(val, dtype=torch.float32)

        logger.debug(f"Missing data for '{prefix}', using zeros (dim={dim})")
        return torch.zeros(dim, dtype=torch.float32)
