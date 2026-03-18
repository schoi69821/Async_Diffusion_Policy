"""
Vision Dataset - HDF5 loader for image + joint observations.
Loads episodes saved by collect_episodes.py.

Supports temporal_stride for using high-frequency data (50Hz) to predict
wide time windows without downsampling:
  stride=1: action[t, t+1, ..., t+15]  (at recording freq)
  stride=3: action[t, t+3, ..., t+45]  (3x wider window, same 16 steps)

HDF5 structure per episode:
    /observations/qpos:   (T, 7) float32 - Joint positions
    /observations/images:  (T, 224, 224, 3) uint8 - Camera images
    /action:               (T, 7) float32 - Actions
"""
import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    raise ImportError("h5py required: pip install h5py")


class VisionDiffusionDataset(Dataset):
    """
    Dataset for vision-based diffusion policy training.

    Args:
        data_dir: Directory containing episode HDF5 files
        pred_horizon: Future action steps to predict (default: 16)
        obs_horizon: Past observation steps (default: 1)
        action_dim: Action dimension (default: 7)
        temporal_stride: Stride for action trajectory indexing (default: 1)
            stride=3 on 50Hz data → 16 steps span 0.9s (~15Hz effective)
    """

    def __init__(self, data_dir, pred_horizon=16, obs_horizon=1, action_dim=7,
                 temporal_stride=1, **kwargs):
        self.data_dir = data_dir
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.temporal_stride = temporal_stride

        # Find all episode files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        self.files += sorted(glob.glob(os.path.join(data_dir, "*.h5")))

        if not self.files:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

        # Load episodes and build index
        self.episodes = []
        self.indices = []
        self._load_episodes()
        self._compute_stats()

        # ImageNet normalization
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_episodes(self):
        # Action trajectory span: pred_horizon steps * stride
        traj_span = (self.pred_horizon - 1) * self.temporal_stride + 1

        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    qpos = f['observations/qpos'][:].astype(np.float32)
                    action = f['action'][:].astype(np.float32)
                    T = len(qpos)

                    episode = {
                        'file_path': file_path,
                        'qpos': qpos,
                        'action': action,
                        'length': T,
                    }
                    ep_idx = len(self.episodes)
                    self.episodes.append(episode)

                    # Valid start indices: need traj_span frames from t
                    max_start = T - traj_span
                    for t in range(max(0, max_start)):
                        self.indices.append((ep_idx, t))

            except Exception as e:
                print(f"[VisionDataset] Error loading {file_path}: {e}")

        stride_info = f", stride={self.temporal_stride}" if self.temporal_stride > 1 else ""
        print(f"[VisionDataset] Loaded {len(self.episodes)} episodes, "
              f"{len(self.indices)} samples{stride_info}")

    def _compute_stats(self):
        """Compute normalization stats for qpos and action."""
        all_qpos = np.concatenate([ep['qpos'] for ep in self.episodes])
        all_action = np.concatenate([ep['action'] for ep in self.episodes])

        self.stats = {
            'qpos': {'min': all_qpos.min(axis=0), 'max': all_qpos.max(axis=0)},
            'action': {'min': all_action.min(axis=0), 'max': all_action.max(axis=0)},
        }
        print(f"[VisionDataset] Stats computed from {len(all_qpos)} frames")

    def normalize_qpos(self, qpos):
        mn, mx = self.stats['qpos']['min'], self.stats['qpos']['max']
        rng = mx - mn
        rng[rng < 1e-6] = 1.0
        return (qpos - mn) / rng * 2 - 1

    def unnormalize_action(self, action):
        mn, mx = self.stats['action']['min'], self.stats['action']['max']
        return (action + 1) / 2 * (mx - mn) + mn

    def normalize_action(self, action):
        mn, mx = self.stats['action']['min'], self.stats['action']['max']
        rng = mx - mn
        rng[rng < 1e-6] = 1.0
        return (action - mn) / rng * 2 - 1

    def normalize_image(self, img):
        img = img.astype(np.float32) / 255.0
        img = (img - self.img_mean) / self.img_std
        return img.transpose(2, 0, 1)  # HWC → CHW

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, t = self.indices[idx]
        episode = self.episodes[ep_idx]
        s = self.temporal_stride

        # Observation indices (obs_horizon frames, spaced by stride)
        obs_indices = [max(0, t - (self.obs_horizon - 1 - i) * s) for i in range(self.obs_horizon)]

        # Load images
        with h5py.File(episode['file_path'], 'r') as f:
            if self.obs_horizon == 1:
                img_norm = self.normalize_image(f['observations/images'][t])
                img_tensor = torch.from_numpy(img_norm)
            else:
                imgs = [self.normalize_image(f['observations/images'][ot]) for ot in obs_indices]
                img_tensor = torch.from_numpy(np.stack(imgs))

        # Load qpos
        if self.obs_horizon == 1:
            qpos_tensor = torch.from_numpy(self.normalize_qpos(episode['qpos'][t]))
        else:
            qposs = [self.normalize_qpos(episode['qpos'][ot]) for ot in obs_indices]
            qpos_tensor = torch.from_numpy(np.stack(qposs))

        # Action trajectory with stride: action[t], action[t+s], action[t+2s], ...
        action_indices = [t + i * s for i in range(self.pred_horizon)]
        action = episode['action'][action_indices]  # (pred_horizon, 7)

        # Progress
        ep_len = episode['length']
        progress = np.array([t / max(ep_len - 1, 1)], dtype=np.float32)

        action_norm = self.normalize_action(action)

        return {
            'image': img_tensor,
            'qpos': qpos_tensor,
            'action': torch.from_numpy(action_norm),
            'progress': torch.from_numpy(progress),
        }
