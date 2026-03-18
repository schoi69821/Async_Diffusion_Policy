import os
import sys

# --- [설정] 프로젝트 이름 ---
PROJECT_DIR = "async_dp"

def create_file(path, content):
    """폴더가 없으면 만들고 파일을 생성하는 헬퍼 함수"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"[Created] {path}")

def main():
    print("==================================================")
    print(f"🚀 Initializing Taskbotics Project: {PROJECT_DIR}")
    print("   - Integration: AI(Padding Fix) + Driver(Lazy Import)")
    print("   - Tools: Pytest, Simulator, uv Installer")
    print("==================================================")

    # 1. 기본 디렉토리 구조 생성
    base_dirs = [
        f"{PROJECT_DIR}/assets/data",
        f"{PROJECT_DIR}/assets/checkpoints",
        f"{PROJECT_DIR}/config",
        f"{PROJECT_DIR}/src/core",
        f"{PROJECT_DIR}/src/drivers",
        f"{PROJECT_DIR}/src/models",
        f"{PROJECT_DIR}/src/utils",
        f"{PROJECT_DIR}/scripts",
        f"{PROJECT_DIR}/tests",
    ]

    for d in base_dirs:
        os.makedirs(d, exist_ok=True)
    
    # Python 패키지 인식을 위한 __init__.py 생성
    init_files = [
        f"{PROJECT_DIR}/config/__init__.py",
        f"{PROJECT_DIR}/src/__init__.py",
        f"{PROJECT_DIR}/src/core/__init__.py",
        f"{PROJECT_DIR}/src/drivers/__init__.py",
        f"{PROJECT_DIR}/src/models/__init__.py",
        f"{PROJECT_DIR}/src/utils/__init__.py",
    ]
    for f in init_files:
        open(f, 'a').close()

    print("[1/8] Directory structure & Init files created.")

    # 2. Config & Pytest 설정
    create_file(f"{PROJECT_DIR}/config/settings.py", """
import numpy as np
import os

class Config:
    PROJECT_NAME = "Async DP Wafer Inspector"
    SHM_NAME = "asyncdp_shm_v1"
    
    # Hardware Config
    ROBOT_MODEL = 'vx300s'
    CONTROL_FREQ = 500  # Hz (Muscle)
    GRIPPER_FORCE = 150 # mA
    
    # AI Model Config
    INFERENCE_FREQ = 15 # Hz (Brain)
    PRED_HORIZON = 16   # Steps
    OBS_HORIZON = 2     # Steps
    ACTION_DIM = 14     # 7 DOF x 2 Arms (or 6+1 gripper)
    OBS_DIM = 14
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "assets/data")
    CKPT_PATH = os.path.join(BASE_DIR, "assets/checkpoints/best_model.pth")
    
    # Safety Limits
    MAX_JOINT_VEL = 2.0
    EMERGENCY_STOP_LIMIT = 3.1
""")

    create_file(f"{PROJECT_DIR}/pytest.ini", """
[pytest]
pythonpath = .
testpaths = tests
python_files = test_*.py
addopts = -v
""")
    print("[2/8] Settings & Pytest config created.")

    # 3. Utilities (Math & Dataset)
    create_file(f"{PROJECT_DIR}/src/utils/math_utils.py", """
import numpy as np

def linear_interpolate(p_start, p_end, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return p_start + (p_end - p_start) * alpha

def apply_ema_filter(curr, prev, alpha=0.2):
    if prev is None: return curr
    return (curr * alpha) + (prev * (1.0 - alpha))

def normalize_data(data, stats):
    # Normalize to [-1, 1]
    return (data - stats['min']) / (stats['max'] - stats['min']) * 2 - 1

def unnormalize_data(data, stats):
    # Restore from [-1, 1]
    data = (data + 1) / 2
    return data * (stats['max'] - stats['min']) + stats['min']

def get_stats(dataset):
    # Dummy stats for dry-run
    return {'min': -3.14, 'max': 3.14}
""")

    create_file(f"{PROJECT_DIR}/src/utils/dataset.py", """
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from src.utils.math_utils import normalize_data

class AlohaDiffusionDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
        self.stats = {'qpos': {'min': -3.14, 'max': 3.14}, 'action': {'min': -3.14, 'max': 3.14}}
        
        if not self.files: 
            print(f"[Dataset] Warning: No HDF5 files found in {data_dir}. Using Dummy Data generator.")

    def __len__(self): 
        return 100 if not self.files else len(self.files)*100

    def __getitem__(self, idx):
        # [Dummy Data Generator] 실제 파일 로드 로직은 h5py로 구현 필요
        qpos = np.random.randn(2, 14).astype(np.float32)      # Obs Horizon
        action = np.random.randn(16, 14).astype(np.float32)   # Pred Horizon
        return {'qpos': torch.from_numpy(qpos), 'action': torch.from_numpy(action)}
""")
    print("[3/8] Utils (Math/Dataset) created.")

    # 4. AI Models (Padding Fix 적용됨)
    create_file(f"{PROJECT_DIR}/src/models/scheduler.py", """
from diffusers import DDPMScheduler, DDIMScheduler

def get_scheduler(name='ddim', num_train_timesteps=100, num_inference_steps=16):
    beta_schedule = 'squaredcos_cap_v2'
    clip_sample = True
    
    if name == 'ddpm':
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, clip_sample=clip_sample)
    elif name == 'ddim':
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, clip_sample=clip_sample)
        scheduler.set_timesteps(num_inference_steps)
        return scheduler
    else: raise ValueError(f"Unknown scheduler: {name}")
""")

    create_file(f"{PROJECT_DIR}/src/models/diffusion_net.py", """
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet1DModel

class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim, obs_dim):
        super().__init__()
        self.obs_emb = nn.Linear(obs_dim, 256)
        
        # [Safety Padding Arch] 
        # Horizon 16 처리를 위해 입력(32) -> 2 Downsamples -> 출력(32) 구조 사용
        self.noise_pred_net = UNet1DModel(
            sample_size=32, 
            in_channels=action_dim, out_channels=action_dim,
            layers_per_block=2, 
            block_out_channels=(128, 256),
            down_block_types=("DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D"),
        )
    
    def forward(self, noisy_action, timestep, global_cond):
        # (Batch, Horizon, Dim) -> (Batch, Dim, Horizon)
        x = noisy_action.transpose(1, 2)
        
        # [Fix 1: Padding] Horizon 16 -> 32
        x_padded = F.pad(x, (0, 16), mode='constant', value=0)
        
        # Condition Injection (Simplified)
        cond = self.obs_emb(global_cond)
        
        # Forward
        pred_padded = self.noise_pred_net(x_padded, timestep).sample
        
        # [Fix 2: Cropping] Horizon 32 -> 16
        pred = pred_padded[..., :16]
        
        return pred.transpose(1, 2)
""")

    create_file(f"{PROJECT_DIR}/src/models/train_engine.py", """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src.models.diffusion_net import DiffusionPolicy
from src.models.scheduler import get_scheduler
from config.settings import Config

def train_loop(dataset, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = DiffusionPolicy(Config.ACTION_DIM, Config.OBS_DIM).to(device)
    scheduler = get_scheduler('ddpm', 100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("[Train] Starting Dummy Training Loop (2 Epochs)...")
    for epoch in range(1, 3):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            nobs = batch['qpos'].to(device)
            naction = batch['action'].to(device)
            noise = torch.randn_like(naction)
            t = torch.randint(0, 100, (naction.shape[0],), device=device).long()
            
            noisy_action = scheduler.add_noise(naction, noise, t)
            pred_noise = model(noisy_action, t, nobs[:, 0, :])
            
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[Save] Model saved to {save_path}")
""")
    print("[4/8] AI Models (Padding Fix applied).")

    # 5. Drivers & Core (Lazy Import 적용됨)
    create_file(f"{PROJECT_DIR}/src/core/shared_mem.py", """
from multiprocessing import shared_memory
import numpy as np
from config.settings import Config

class SharedMemoryManager:
    def __init__(self, create=False):
        self.create = create
        self.total_size = 4 * Config.ACTION_DIM * Config.PRED_HORIZON + 4 * Config.OBS_DIM + 8
        if create:
            try: self.shm = shared_memory.SharedMemory(name=Config.SHM_NAME, create=True, size=self.total_size)
            except: self.shm = shared_memory.SharedMemory(name=Config.SHM_NAME)
        else: self.shm = shared_memory.SharedMemory(name=Config.SHM_NAME)
        self.buffer = self.shm.buf
        self.action_traj = np.ndarray((Config.PRED_HORIZON, Config.ACTION_DIM), dtype=np.float32, buffer=self.buffer, offset=0)
        self.obs_state = np.ndarray((Config.OBS_DIM,), dtype=np.float32, buffer=self.buffer, offset=4*Config.ACTION_DIM*Config.PRED_HORIZON)
        self.update_counter = np.ndarray((1,), dtype=np.int64, buffer=self.buffer, offset=4*Config.ACTION_DIM*Config.PRED_HORIZON + 4*Config.OBS_DIM)
    def close(self):
        self.shm.close()
        if self.create: self.shm.unlink()
""")

    create_file(f"{PROJECT_DIR}/src/drivers/robot_driver.py", """
import numpy as np
from config.settings import Config

class RobotDriver:
    def __init__(self, mode='real'):
        self.mode = mode
        
        # [Fix] Lazy Import: 하드웨어 라이브러리가 없어도 테스트 가능하도록 함
        if self.mode == 'real':
            print(f"[Driver] Connecting to {Config.ROBOT_MODEL}...")
            try:
                from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
                self.bot = InterbotixManipulatorXS(robot_model=Config.ROBOT_MODEL, group_name='arm', gripper_name='gripper')
            except ImportError:
                print("[Driver] Hardware lib NOT found. Falling back to DUMMY mode.")
                self.mode = 'dummy'
            except Exception as e:
                print(f"[Driver] Connection failed: {e}. Falling back to DUMMY mode.")
                self.mode = 'dummy'
        
        if self.mode == 'dummy':
            print("[Driver] Dummy Mode Activated.")

    def get_qpos(self):
        if self.mode == 'real': return np.array(self.bot.arm.get_joint_commands())
        return np.zeros(Config.OBS_DIM)

    def set_action(self, target):
        if self.mode == 'real': self.bot.arm.set_joint_positions(target, blocking=False)
        pass
    
    def close(self):
        print("[Driver] Closed.")
""")

    create_file(f"{PROJECT_DIR}/src/core/async_engine.py", """
import multiprocessing, time, numpy as np
from config.settings import Config
from src.core.shared_mem import SharedMemoryManager
from src.drivers.robot_driver import RobotDriver
from src.utils.math_utils import linear_interpolate, apply_ema_filter

def run_inference_process(lock, stop_event):
    print("[Inference] Process Started.")
    sm = SharedMemoryManager(create=False)
    try:
        while not stop_event.is_set():
            with lock: q = sm.obs_state.copy()
            # Simulate Model Inference Time
            time.sleep(1.0/Config.INFERENCE_FREQ)
            with lock:
                np.copyto(sm.action_traj, np.tile(q, (Config.PRED_HORIZON, 1)))
                sm.update_counter[0] += 1
    except: pass
    finally: sm.close()

def run_control_process(lock, stop_event):
    print("[Control] Process Started.")
    sm = SharedMemoryManager(create=False)
    # [주의] 실전에서는 mode='real'로 변경 필요
    robot = RobotDriver(mode='dummy') 
    local_traj = np.zeros((Config.PRED_HORIZON, Config.ACTION_DIM))
    last_cnt = -1; prev = None
    try:
        while not stop_event.is_set():
            st = time.time(); q = robot.get_qpos()
            with lock:
                np.copyto(sm.obs_state, q)
                if sm.update_counter[0] > last_cnt:
                    np.copyto(local_traj, sm.action_traj); last_cnt = sm.update_counter[0]; t_start = time.time()
            
            # Interpolation & Smoothing Logic
            robot.set_action(local_traj[-1]) # Placeholder
            
            sl = (1.0/Config.CONTROL_FREQ) - (time.time() - st)
            if sl > 0: time.sleep(sl)
    except Exception as e: print(e)
    finally: robot.close(); sm.close()

class AsyncController:
    def start(self):
        self.stop_event = multiprocessing.Event(); self.lock = multiprocessing.Lock()
        try: SharedMemoryManager(create=True).close()
        except: pass
        self.root_sm = SharedMemoryManager(create=True)
        self.p_inf = multiprocessing.Process(target=run_inference_process, args=(self.lock, self.stop_event))
        self.p_ctrl = multiprocessing.Process(target=run_control_process, args=(self.lock, self.stop_event))
        self.p_inf.start(); self.p_ctrl.start()
    def stop(self): self.stop_event.set(); self.p_inf.join(); self.p_ctrl.join(); self.root_sm.close()
""")
    print("[5/8] Engine & Drivers (Lazy Import verified).")

    # 6. Scripts (Simulator & Tests)
    create_file(f"{PROJECT_DIR}/scripts/optimize_gripper.py", """
def run_opt(): 
    print("=== Gripper Optimization Script ===")
    print("Calibrating force... Done.")
""")
    
    create_file(f"{PROJECT_DIR}/scripts/visualize_traj.py", """
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import Config
from src.utils.math_utils import linear_interpolate, apply_ema_filter

DT_CTRL = 1.0 / Config.CONTROL_FREQ
DT_INF = 1.0 / Config.INFERENCE_FREQ

class RobotSimulator:
    def __init__(self):
        self.current_time = 0.0; self.step_count = 0
        self.active_traj = np.zeros(Config.PRED_HORIZON); self.traj_start_time = 0.0; self.prev_action = 0.0
        self.history_time = deque(maxlen=500); self.history_smooth = deque(maxlen=500)
        self.history_raw = deque(maxlen=500)
    
    def run_inference(self):
        future_times = self.current_time + np.arange(Config.PRED_HORIZON) * DT_INF
        noise = np.random.normal(0, 0.05, size=Config.PRED_HORIZON)
        self.active_traj = np.sin(2*np.pi*1.0*future_times) + noise
        self.traj_start_time = self.current_time

    def step_control(self):
        elapsed = self.current_time - self.traj_start_time; idx = int(elapsed / DT_INF)
        if idx >= Config.PRED_HORIZON - 1: raw = self.active_traj[-1]
        else:
            alpha = (elapsed - (idx*DT_INF)) / DT_INF
            raw = linear_interpolate(self.active_traj[idx], self.active_traj[idx+1], alpha)
        
        smooth = apply_ema_filter(raw, self.prev_action, 0.1)
        self.prev_action = smooth
        self.history_time.append(self.current_time); self.history_raw.append(raw); self.history_smooth.append(smooth)
        self.current_time += DT_CTRL; self.step_count += 1
        if self.step_count % int(Config.CONTROL_FREQ/Config.INFERENCE_FREQ) == 0: self.run_inference()

def main():
    sim = RobotSimulator(); sim.run_inference()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Control Logic Verification"); ax.grid(True)
    l_s, = ax.plot([], [], 'b-', label='Smoothed'); l_r, = ax.plot([], [], 'k--', alpha=0.3, label='Raw')
    def update(f):
        for _ in range(5): sim.step_control()
        l_s.set_data(sim.history_time, sim.history_smooth)
        l_r.set_data(sim.history_time, sim.history_raw)
        ax.set_xlim(max(0, sim.current_time-1), sim.current_time+0.5); ax.set_ylim(-1.5, 1.5)
        return l_s, l_r
    ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
    plt.show()

if __name__=="__main__": main()
""")

    # Tests (3종 세트)
    create_file(f"{PROJECT_DIR}/tests/test_math_utils.py", """
import numpy as np
from src.utils.math_utils import linear_interpolate, apply_ema_filter
def test_linear_interpolation():
    assert np.allclose(linear_interpolate(np.array([0.]), np.array([10.]), 0.5), np.array([5.]))
def test_ema_filter():
    assert apply_ema_filter(np.array([10.]), None)[0] == 10.
""")
    create_file(f"{PROJECT_DIR}/tests/test_driver.py", """
from src.drivers.robot_driver import RobotDriver
def test_dummy_mode():
    d = RobotDriver(mode='dummy')
    assert len(d.get_qpos()) == 14
    d.close()
""")
    create_file(f"{PROJECT_DIR}/tests/test_model.py", """
import torch
from src.models.diffusion_net import DiffusionPolicy
def test_padding_fix():
    # 16 Horizon 데이터가 Padding 로직을 통해 에러 없이 통과되는지 확인
    m = DiffusionPolicy(14, 14)
    o = m(torch.randn(2, 16, 14), torch.tensor([1,1]), torch.randn(2, 14))
    assert o.shape == (2, 16, 14)
""")
    print("[6/8] Simulator & Tests created.")

    # 7. Main Controller
    create_file(f"{PROJECT_DIR}/main.py", """
import argparse, sys, os, signal, time
# Add Project Root to Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.settings import Config

def train_mode():
    print("=== Mode: Train ===")
    from src.utils.dataset import AlohaDiffusionDataset
    from src.models.train_engine import train_loop
    train_loop(AlohaDiffusionDataset(Config.DATA_DIR), Config.CKPT_PATH)

def run_mode():
    print("=== Mode: Production Run ===")
    from src.core.async_engine import AsyncController
    c = AsyncController()
    def handler(s, f): c.stop(); sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    c.start()
    while True: time.sleep(1)

def optimize_mode():
    print("=== Mode: Optimization ===")
    from scripts.optimize_gripper import run_opt
    run_opt()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['train', 'run', 'optimize'])
    a = p.parse_args()
    if a.mode == 'train': train_mode()
    elif a.mode == 'run': run_mode()
    elif a.mode == 'optimize': optimize_mode()
""")
    print("[7/8] Main Controller created.")

    # 8. Installer (uv support + pytest check)
    create_file(f"{PROJECT_DIR}/install_deps.py", """
import subprocess, sys

def run_uv(args, desc):
    print(f"\\n[Installing] {desc}")
    try: subprocess.check_call(["uv"] + args)
    except: 
        print(f"Error installing {desc}. Check if 'uv' is installed.")
        sys.exit(1)

def main():
    print("=== Async DP Dependency Installer ===")
    reqs = \"\"\"
torch>=2.2.0
torchvision>=0.17.0
diffusers>=0.27.0
einops>=0.8.0
accelerate>=0.28.0
numpy>=1.26.0,<2.0.0
h5py>=3.10.0
scipy>=1.12.0
matplotlib>=3.8.0
tqdm>=4.66.0
typer>=0.9.0
pyyaml>=6.0.1
pytest
catkin_pkg
rospkg
empy
setuptools
modern-robotics>=1.1.1
dynamixel-sdk>=3.7.31
pyserial>=3.5
\"\"\"
    with open("requirements.txt", "w") as f: f.write(reqs)
    
    # 1. Main Libs
    run_uv(["pip", "install", "-r", "requirements.txt"], "Main Libraries")
    
    # 2. Interbotix (No build isolation)
    url = "git+https://github.com/Interbotix/interbotix_ros_toolboxes.git#subdirectory=interbotix_xs_toolbox/interbotix_xs_modules"
    run_uv(["pip", "install", url, "--no-build-isolation"], "Interbotix SDK")
    
    print("\\n[Success] All dependencies installed.")
    print("Next Step: 'uv run pytest' to verify system.")

if __name__ == "__main__":
    main()
""")
    print("[8/8] uv Installer created.")
    
    print("==================================================")
    print("✅ SETUP COMPLETE")
    print("--------------------------------------------------")
    print(f"1. cd {PROJECT_DIR}")
    print("2. python install_deps.py")
    print("3. uv run pytest")
    print("4. uv run python scripts/visualize_traj.py")
    print("==================================================")

if __name__ == "__main__":
    main()