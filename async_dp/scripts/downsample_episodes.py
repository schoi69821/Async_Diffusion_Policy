"""
Downsample 50Hz episode data to 15Hz for training.

The model predicts 16 steps. At 50Hz, that's 0.32s (too short to capture grasp transitions).
At 15Hz, 16 steps = 1.07s — enough to cover approach → grasp → lift in one prediction window.

Usage:
    uv run python scripts/downsample_episodes.py --src episodes/pen_fixed --dst episodes/pen_fixed_15hz
"""
import numpy as np
import h5py
import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_FREQ = 50.0
DST_FREQ = 15.0


def downsample_array(arr, src_freq, dst_freq):
    """Downsample array from src_freq to dst_freq using linear interpolation."""
    n_src = len(arr)
    duration = (n_src - 1) / src_freq
    n_dst = max(2, int(round(duration * dst_freq)) + 1)

    src_t = np.linspace(0, duration, n_src)
    dst_t = np.linspace(0, duration, n_dst)

    if arr.ndim == 1:
        return np.interp(dst_t, src_t, arr)

    out = np.zeros((n_dst, *arr.shape[1:]), dtype=arr.dtype)

    if arr.ndim == 2:
        # Joint positions / actions: interpolate each joint
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(dst_t, src_t, arr[:, j])
    elif arr.ndim == 4:
        # Images (T, H, W, C): nearest-neighbor (can't interpolate images)
        src_indices = np.round(np.interp(dst_t, src_t, np.arange(n_src))).astype(int)
        src_indices = np.clip(src_indices, 0, n_src - 1)
        out = arr[src_indices]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    return out


def downsample_episode(src_path, dst_path):
    """Downsample a single episode HDF5 file using module-level frequencies."""
    return downsample_episode_with_freq(src_path, dst_path, SRC_FREQ, DST_FREQ)


def downsample_episode_with_freq(src_path, dst_path, src_freq, dst_freq):
    """Downsample a single episode HDF5 file."""
    with h5py.File(src_path, 'r') as f:
        qpos = f['observations/qpos'][:].astype(np.float32)
        action = f['action'][:].astype(np.float32)
        images = f['observations/images'][:]  # uint8

    n_src = len(qpos)
    duration = (n_src - 1) / src_freq

    qpos_ds = downsample_array(qpos, src_freq, dst_freq)
    action_ds = downsample_array(action, src_freq, dst_freq)
    images_ds = downsample_array(images, src_freq, dst_freq)

    n_dst = len(qpos_ds)

    with h5py.File(dst_path, 'w') as f:
        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos_ds)
        obs.create_dataset('images', data=images_ds, chunks=(1, *images_ds.shape[1:]),
                           compression='gzip', compression_opts=4)
        f.create_dataset('action', data=action_ds)
        f.attrs['num_timesteps'] = n_dst
        f.attrs['src_freq'] = src_freq
        f.attrs['dst_freq'] = dst_freq
        f.attrs['src_frames'] = n_src

    return n_src, n_dst, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source episode directory (50Hz)")
    parser.add_argument("--dst", required=True, help="Destination directory (15Hz)")
    parser.add_argument("--src-freq", type=float, default=SRC_FREQ)
    parser.add_argument("--dst-freq", type=float, default=DST_FREQ)
    args = parser.parse_args()

    src_freq = args.src_freq
    dst_freq = args.dst_freq

    files = sorted(glob.glob(os.path.join(args.src, "*.hdf5")))
    files += sorted(glob.glob(os.path.join(args.src, "*.h5")))

    if not files:
        print(f"No HDF5 files found in {args.src}")
        return

    os.makedirs(args.dst, exist_ok=True)

    print(f"Downsampling {len(files)} episodes: {src_freq}Hz → {dst_freq}Hz")
    print(f"  Source: {args.src}")
    print(f"  Dest:   {args.dst}")
    print()

    total_src = 0
    total_dst = 0

    for fp in files:
        fname = os.path.basename(fp)
        dst_path = os.path.join(args.dst, fname)

        n_src, n_dst, dur = downsample_episode_with_freq(fp, dst_path, src_freq, dst_freq)
        total_src += n_src
        total_dst += n_dst
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"  {fname}: {n_src} → {n_dst} frames ({dur:.1f}s) [{size_mb:.0f}MB]")

    print(f"\nDone: {total_src} → {total_dst} total frames ({total_dst/total_src*100:.0f}%)")
    print(f"16-step window: {16/dst_freq:.2f}s (was {16/src_freq:.2f}s)")


if __name__ == "__main__":
    main()
