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