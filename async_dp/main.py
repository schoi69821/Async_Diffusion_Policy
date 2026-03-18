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