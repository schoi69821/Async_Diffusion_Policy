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