"""
Aloha Diffusion Dataset - HDF5 Data Loader
Loads demonstration data for training Diffusion Policy
"""
import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from src.utils.math_utils import normalize_data, get_stats

# Optional h5py import with fallback
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("[Dataset] Warning: h5py not installed. Using dummy data generator.")


class AlohaDiffusionDataset(Dataset):
    """
    Dataset for loading Aloha robot demonstration data from HDF5 files.

    Expected HDF5 structure:
        /observations/qpos: (T, 14) - Joint positions
        /observations/qvel: (T, 14) - Joint velocities (optional)
        /action: (T, 14) - Actions taken

    Args:
        data_dir: Directory containing HDF5 files
        pred_horizon: Number of future steps to predict (default: 16)
        obs_horizon: Number of observation steps (default: 2)
        normalize: Whether to normalize data to [-1, 1]
    """

    def __init__(self, data_dir, pred_horizon=16, obs_horizon=2, normalize=True):
        self.data_dir = data_dir
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.normalize = normalize

        # Find all HDF5 files
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        self.files += sorted(glob.glob(os.path.join(data_dir, "*.h5")))

        # Statistics for normalization
        self.stats = {
            'qpos': {'min': -3.14, 'max': 3.14},
            'action': {'min': -3.14, 'max': 3.14}
        }

        # Build index mapping (episode_idx, start_timestep)
        self.indices = []
        self.episodes = []

        if self.files and HDF5_AVAILABLE:
            print(f"[Dataset] Found {len(self.files)} HDF5 files in {data_dir}")
            self._load_episodes()
            self._compute_stats()
        else:
            if not self.files:
                print(f"[Dataset] No HDF5 files found in {data_dir}. Using dummy data.")
            self._use_dummy_mode()

    def _load_episodes(self):
        """Load all episodes and build index"""
        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    qpos = f['observations/qpos'][:]
                    action = f['action'][:]

                    # Optional: load qvel if available
                    qvel = None
                    if 'observations/qvel' in f:
                        qvel = f['observations/qvel'][:]

                    episode = {
                        'qpos': qpos.astype(np.float32),
                        'action': action.astype(np.float32),
                        'qvel': qvel.astype(np.float32) if qvel is not None else None,
                        'length': len(qpos)
                    }

                    episode_idx = len(self.episodes)
                    self.episodes.append(episode)

                    # Create indices for valid starting points
                    # Need obs_horizon past frames and pred_horizon future frames
                    max_start = episode['length'] - self.pred_horizon - self.obs_horizon + 1
                    for t in range(max(0, max_start)):
                        self.indices.append((episode_idx, t + self.obs_horizon - 1))

            except Exception as e:
                print(f"[Dataset] Error loading {file_path}: {e}")

        print(f"[Dataset] Loaded {len(self.episodes)} episodes, {len(self.indices)} samples")

    def _compute_stats(self):
        """Compute normalization statistics from data"""
        if not self.episodes:
            return

        all_qpos = np.concatenate([ep['qpos'] for ep in self.episodes], axis=0)
        all_action = np.concatenate([ep['action'] for ep in self.episodes], axis=0)

        self.stats = {
            'qpos': {
                'min': float(all_qpos.min()),
                'max': float(all_qpos.max()),
                'mean': float(all_qpos.mean()),
                'std': float(all_qpos.std())
            },
            'action': {
                'min': float(all_action.min()),
                'max': float(all_action.max()),
                'mean': float(all_action.mean()),
                'std': float(all_action.std())
            }
        }
        print(f"[Dataset] Stats computed: qpos range [{self.stats['qpos']['min']:.2f}, {self.stats['qpos']['max']:.2f}]")

    def _use_dummy_mode(self):
        """Setup dummy data generation for testing"""
        self.dummy_mode = True
        # Create 100 dummy samples
        for i in range(100):
            self.indices.append((i, 0))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - qpos: (obs_horizon, action_dim) observation sequence
                - action: (pred_horizon, action_dim) action sequence to predict
        """
        if hasattr(self, 'dummy_mode') and self.dummy_mode:
            return self._get_dummy_item(idx)

        episode_idx, timestep = self.indices[idx]
        episode = self.episodes[episode_idx]

        # Extract observation window (past obs_horizon frames)
        obs_start = timestep - self.obs_horizon + 1
        obs_end = timestep + 1
        qpos = episode['qpos'][obs_start:obs_end].copy()  # (obs_horizon, dim)

        # Extract action window (future pred_horizon frames)
        action_start = timestep
        action_end = timestep + self.pred_horizon
        action = episode['action'][action_start:action_end].copy()  # (pred_horizon, dim)

        # Normalize if requested
        if self.normalize:
            qpos = self._normalize(qpos, self.stats['qpos'])
            action = self._normalize(action, self.stats['action'])

        return {
            'qpos': torch.from_numpy(qpos),
            'action': torch.from_numpy(action)
        }

    def _get_dummy_item(self, idx):
        """Generate dummy data for testing"""
        # Generate smooth trajectory-like data
        t = np.linspace(0, 2 * np.pi, self.pred_horizon + self.obs_horizon)
        base = np.sin(t + idx * 0.1)

        qpos = np.zeros((self.obs_horizon, 14), dtype=np.float32)
        action = np.zeros((self.pred_horizon, 14), dtype=np.float32)

        for i in range(14):
            phase = i * np.pi / 7
            full_traj = 0.5 * np.sin(t + phase + idx * 0.1)
            qpos[:, i] = full_traj[:self.obs_horizon]
            action[:, i] = full_traj[self.obs_horizon:]

        # Add small noise
        qpos += np.random.randn(*qpos.shape).astype(np.float32) * 0.01
        action += np.random.randn(*action.shape).astype(np.float32) * 0.01

        return {
            'qpos': torch.from_numpy(qpos),
            'action': torch.from_numpy(action)
        }

    def _normalize(self, data, stats):
        """Normalize data to [-1, 1] range"""
        data_min = stats['min']
        data_max = stats['max']
        if data_max - data_min < 1e-6:
            return data
        return ((data - data_min) / (data_max - data_min)) * 2 - 1

    def unnormalize(self, data, key='action'):
        """Unnormalize data from [-1, 1] to original range"""
        stats = self.stats[key]
        data = (data + 1) / 2
        return data * (stats['max'] - stats['min']) + stats['min']


def create_sample_hdf5(output_path, num_timesteps=500, action_dim=14):
    """
    Utility function to create a sample HDF5 file for testing.

    Args:
        output_path: Path to save the HDF5 file
        num_timesteps: Number of timesteps in the episode
        action_dim: Dimension of action/observation space
    """
    if not HDF5_AVAILABLE:
        print("[Dataset] h5py not available. Cannot create sample file.")
        return

    # Generate smooth trajectory
    t = np.linspace(0, 4 * np.pi, num_timesteps)

    qpos = np.zeros((num_timesteps, action_dim), dtype=np.float32)
    action = np.zeros((num_timesteps, action_dim), dtype=np.float32)

    for i in range(action_dim):
        phase = i * np.pi / action_dim
        qpos[:, i] = 0.5 * np.sin(t + phase)
        action[:, i] = 0.5 * np.sin(t + phase + 0.1)  # Slightly ahead

    # Add noise
    qpos += np.random.randn(*qpos.shape).astype(np.float32) * 0.02
    action += np.random.randn(*action.shape).astype(np.float32) * 0.02

    # Save to HDF5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=qpos)
        f.create_dataset('action', data=action)

    print(f"[Dataset] Sample HDF5 saved to {output_path}")
