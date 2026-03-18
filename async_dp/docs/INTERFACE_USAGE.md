# 로봇 통신 인터페이스 사용 가이드

Async DP와 외부 로봇 컨트롤러 간의 통신 방법을 설명합니다.

## 목차

1. [개요](#개요)
2. [빠른 시작: Dynamixel 컨트롤러](#빠른-시작-dynamixel-컨트롤러)
3. [SharedMemory 인터페이스](#sharedmemory-인터페이스)
4. [gRPC 인터페이스](#grpc-인터페이스)
5. [인터페이스 선택 가이드](#인터페이스-선택-가이드)
6. [전체 예제 코드](#전체-예제-코드)
7. [고급: 커스텀 컨트롤러 만들기](#고급-커스텀-컨트롤러-만들기)

---

## 개요

### 통신 구조

```
┌─────────────────────┐              ┌─────────────────────┐
│      Async DP       │              │   Robot Controller  │
│  (AI 추론 시스템)     │ ◀─────────▶ │   (모터 제어 시스템)  │
│                     │   Interface  │                     │
│  - Diffusion Policy │              │  - 모터 드라이버      │
│  - 궤적 생성 (15Hz)  │              │  - 센서 읽기         │
└─────────────────────┘              └─────────────────────┘
```

### 데이터 흐름

| 방향              | 데이터        | 설명              |
|------------------|--------------|-------------------|
| Robot → Async DP | `RobotState` | 현재 관절 위치/속도 |
| Async DP → Robot | `action`     | 단일 관절 명령      |
| Async DP → Robot | `trajectory` | 16스텝 궤적 명령    |

---

## 빠른 시작: Dynamixel 컨트롤러

가장 빠르게 시작하는 방법입니다. VX300s 등 Dynamixel 기반 로봇에서 바로 사용 가능합니다.

### 1단계: 로봇 컨트롤러 실행

```python
# run_robot_controller.py
from src.controllers import DynamixelController
from src.controllers.dynamixel_controller import create_vx300s_controller
import logging

logging.basicConfig(level=logging.INFO)

# VX300s 컨트롤러 생성 (6 DOF)
controller = create_vx300s_controller(
    port="/dev/ttyUSB0",      # Linux
    # port="COM3",            # Windows
    shm_name="my_robot"
)

# 시작
controller.start()

print("로봇 컨트롤러 실행 중...")
print("Async DP에서 'my_robot' 공유 메모리로 연결하세요")

# 실행 유지
try:
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    controller.stop()
```

### 2단계: Async DP 클라이언트 연결

```python
# run_async_dp.py
from src.interfaces import create_interface, InterfaceConfig
from src.models.diffusion_net import DiffusionPolicy
from src.models.scheduler import get_scheduler
import torch

# 인터페이스 연결
config = InterfaceConfig(action_dim=6, obs_dim=6, shm_name="my_robot")
interface = create_interface('shm', config, is_server=False)
interface.connect()

# 모델 로드
model = DiffusionPolicy(6, 6)
scheduler = get_scheduler('ddim', num_train_timesteps=100, num_inference_steps=16)

# 추론 루프
while True:
    state = interface.get_state()
    trajectory = model.get_action(state.qpos, scheduler)
    interface.send_trajectory(trajectory)
```

### 3단계: 실행

```bash
# 터미널 1: 로봇 컨트롤러
python run_robot_controller.py

# 터미널 2: Async DP
python run_async_dp.py
```

---

## SharedMemory 인터페이스

### 특징

- **지연시간**: ~1μs (초저지연)
- **제약사항**: 같은 PC에서만 동작
- **용도**: 실시간 제어가 필요한 경우

### 메모리 레이아웃

```
공유 메모리 구조 (총 ~1KB):
┌────────────────────────────────────────────────────────┐
│ action_command   (14 x float32)     = 56 bytes         │
│ action_trajectory (16 x 14 x float32) = 896 bytes      │
│ robot_state      (14 x float32)     = 56 bytes         │
│ robot_velocity   (14 x float32)     = 56 bytes         │
│ metadata         (4 x float64)      = 32 bytes         │
└────────────────────────────────────────────────────────┘
```

### 사용법 1: Robot Controller 측 (서버)

로봇 컨트롤러는 **서버**로 동작하며, 공유 메모리를 생성합니다.

```python
# robot_controller.py
import numpy as np
import time
from src.interfaces import create_interface, InterfaceConfig

# 1. 설정
config = InterfaceConfig(
    action_dim=14,      # 관절 수
    obs_dim=14,         # 관측 차원
    pred_horizon=16,    # 예측 horizon
    shm_name="my_robot" # 공유 메모리 이름
)

# 2. 서버로 인터페이스 생성 (is_server=True)
interface = create_interface('shm', config, is_server=True)

# 3. 연결
if not interface.connect():
    print("연결 실패!")
    exit(1)

print("서버 시작됨. Async DP 연결 대기중...")

# 4. 제어 루프 (500Hz)
try:
    while True:
        loop_start = time.time()

        # 4-1. 현재 로봇 상태 읽기 (실제로는 모터 드라이버에서)
        current_qpos = read_motor_positions()  # 사용자 구현 필요
        current_qvel = read_motor_velocities() # 사용자 구현 필요

        # 4-2. 상태를 공유 메모리에 업데이트
        interface.update_state(current_qpos, current_qvel)

        # 4-3. Async DP가 보낸 명령 읽기
        action = interface.get_action()        # 단일 명령
        # 또는
        trajectory = interface.get_trajectory() # 전체 궤적

        # 4-4. 모터에 명령 전송 (사용자 구현 필요)
        send_to_motors(action)

        # 4-5. 500Hz 유지
        elapsed = time.time() - loop_start
        sleep_time = (1.0 / 500) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("종료...")
finally:
    interface.disconnect()
```

### 사용법 2: Async DP 측 (클라이언트)

Async DP는 **클라이언트**로 동작하며, 기존 공유 메모리에 연결합니다.

```python
# async_dp_client.py
import numpy as np
from src.interfaces import create_interface, InterfaceConfig

# 1. 설정 (서버와 동일해야 함)
config = InterfaceConfig(
    action_dim=14,
    obs_dim=14,
    pred_horizon=16,
    shm_name="my_robot"  # 서버와 동일한 이름
)

# 2. 클라이언트로 인터페이스 생성 (is_server=False)
interface = create_interface('shm', config, is_server=False)

# 3. 연결 (서버가 먼저 실행되어야 함)
if not interface.connect():
    print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    exit(1)

print("서버에 연결됨!")

# 4. 추론 루프 (15Hz)
try:
    while True:
        # 4-1. 현재 로봇 상태 읽기
        state = interface.get_state()

        if not state.is_valid:
            print("유효하지 않은 상태!")
            continue

        obs = state.qpos  # (14,) numpy array

        # 4-2. Diffusion 추론 (사용자의 모델)
        action_trajectory = run_diffusion_inference(obs)  # (16, 14)

        # 4-3. 궤적 전송
        interface.send_trajectory(action_trajectory)

        # 또는 단일 액션만 전송
        # interface.send_action(action_trajectory[0])

        # 4-4. 15Hz 유지
        time.sleep(1.0 / 15)

except KeyboardInterrupt:
    print("종료...")
finally:
    interface.disconnect()
```

### Context Manager 사용

```python
from src.interfaces import create_interface, InterfaceConfig

config = InterfaceConfig(shm_name="my_robot")

# with 문으로 자동 연결/해제
with create_interface('shm', config, is_server=True) as server:
    # 서버 로직
    server.update_state(qpos)

with create_interface('shm', config, is_server=False) as client:
    # 클라이언트 로직
    state = client.get_state()
```

---

## gRPC 인터페이스

### 특징

- **지연시간**: ~1-10ms
- **장점**: 네트워크를 통한 원격 제어 가능
- **용도**: 분산 시스템, 원격 로봇 제어

### 설치

```bash
# gRPC 의존성 설치
uv pip install grpcio grpcio-tools

# 또는 optional dependency로 설치
uv pip install -e ".[grpc]"
```

### Proto 메시지 구조

```protobuf
// RobotState: 로봇 → Async DP
message RobotState {
    repeated float qpos = 1;      // 관절 위치
    repeated float qvel = 2;      // 관절 속도
    double timestamp = 3;         // 타임스탬프
    bool is_valid = 4;            // 유효성
}

// ActionCommand: Async DP → 로봇
message ActionCommand {
    repeated float action = 1;    // 관절 명령
    double timestamp = 2;         // 타임스탬프
}

// TrajectoryCommand: Async DP → 로봇
message TrajectoryCommand {
    repeated float trajectory = 1; // 평탄화된 궤적
    int32 horizon = 2;             // 스텝 수
    int32 action_dim = 3;          // 관절 수
}
```

### 사용법 1: Robot Controller 측 (gRPC 서버)

```python
# robot_grpc_server.py
import numpy as np
import time
from src.interfaces import InterfaceConfig
from src.interfaces.grpc_interface import GrpcServer

# 1. 설정
config = InterfaceConfig(
    action_dim=14,
    obs_dim=14,
    grpc_port=50051  # gRPC 포트
)

# 2. 서버 생성
server = GrpcServer(config)

# 3. 콜백 등록 (선택사항)
def on_action_received(action):
    """Async DP에서 액션을 받았을 때 호출"""
    print(f"액션 수신: {action[:3]}...")
    send_to_motors(action)

server.set_action_callback(on_action_received)

# 4. 서버 시작
server.start(blocking=False)
print(f"gRPC 서버 시작: port {config.grpc_port}")

# 5. 상태 업데이트 루프
try:
    while True:
        # 현재 로봇 상태 읽기
        qpos = read_motor_positions()
        qvel = read_motor_velocities()

        # 서버 상태 업데이트
        server.update_state(qpos, qvel)

        # 최신 명령 가져오기
        action = server.get_latest_action()
        trajectory = server.get_latest_trajectory()

        time.sleep(0.002)  # 500Hz

except KeyboardInterrupt:
    server.stop()
```

### 사용법 2: Async DP 측 (gRPC 클라이언트)

```python
# async_dp_grpc_client.py
import numpy as np
from src.interfaces import create_interface, InterfaceConfig

# 1. 설정 (원격 서버 주소)
config = InterfaceConfig(
    action_dim=14,
    obs_dim=14,
    grpc_host="192.168.1.100",  # 로봇 PC IP
    grpc_port=50051,
    grpc_timeout=5.0  # 연결 타임아웃 (초)
)

# 2. gRPC 인터페이스 생성
interface = create_interface('grpc', config)

# 3. 연결
if not interface.connect():
    print("gRPC 서버에 연결할 수 없습니다!")
    exit(1)

print(f"연결됨: {config.grpc_host}:{config.grpc_port}")

# 4. 추론 루프
try:
    while True:
        # 상태 읽기 (네트워크 통신)
        state = interface.get_state()

        # 추론
        trajectory = run_diffusion_inference(state.qpos)

        # 궤적 전송
        interface.send_trajectory(trajectory)

        time.sleep(1.0 / 15)

except KeyboardInterrupt:
    interface.disconnect()
```

---

## 인터페이스 선택 가이드

### 결정 흐름도

```
시작
  │
  ▼
같은 PC에서 실행? ──Yes──▶ SharedMemory 사용
  │                        (최고 성능)
  No
  │
  ▼
원격 제어 필요? ───Yes──▶ gRPC 사용
  │                       (네트워크 지원)
  No
  │
  ▼
테스트/개발 중? ───Yes──▶ Dummy 사용
                          (하드웨어 불필요)
```

### 비교표

| 항목    | SharedMemory | gRPC        | Dummy   |
|--------|--------------|-------------|---------|
| 지연시간 | ~1μs         | ~1-10ms    | N/A     |
| 같은 PC | ✅ 필수      | ✅ 가능     | ✅ 가능 |
| 원격    | ❌ 불가      | ✅ 가능     | N/A     |
| 설치    | 없음          | grpcio 필요 | 없음     |
| 복잡도  | 낮음          | 중간        | 매우 낮음 |
| 용도    | 실시간 제어    | 분산 시스템  | 테스트   |

---

## 전체 예제 코드

### 예제 1: SharedMemory 기반 전체 시스템

```python
# example_shm_system.py
"""
SharedMemory 기반 전체 시스템 예제
실행 순서:
1. 터미널 1: python example_shm_system.py --mode server
2. 터미널 2: python example_shm_system.py --mode client
"""
import argparse
import numpy as np
import time
from src.interfaces import create_interface, InterfaceConfig

def run_server():
    """로봇 컨트롤러 (서버) 모드"""
    config = InterfaceConfig(shm_name="example_robot")

    with create_interface('shm', config, is_server=True) as interface:
        print("[Server] 시작됨. 클라이언트 대기중...")

        # 시뮬레이션된 로봇 상태
        qpos = np.zeros(14, dtype=np.float32)

        for i in range(1000):
            # 상태 업데이트 (시뮬레이션)
            qpos += np.random.randn(14).astype(np.float32) * 0.01
            interface.update_state(qpos)

            # 명령 읽기
            action = interface.get_action()

            if i % 100 == 0:
                print(f"[Server] Step {i}: action[0]={action[0]:.3f}")

            time.sleep(0.002)  # 500Hz

def run_client():
    """Async DP (클라이언트) 모드"""
    config = InterfaceConfig(shm_name="example_robot")

    with create_interface('shm', config, is_server=False) as interface:
        print("[Client] 서버에 연결됨!")

        for i in range(100):
            # 상태 읽기
            state = interface.get_state()

            # 간단한 액션 생성 (실제로는 Diffusion 추론)
            action = state.qpos * 0.9  # 원점으로 복귀

            # 액션 전송
            interface.send_action(action)

            if i % 10 == 0:
                print(f"[Client] Step {i}: qpos[0]={state.qpos[0]:.3f}")

            time.sleep(0.066)  # 15Hz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    args = parser.parse_args()

    if args.mode == 'server':
        run_server()
    else:
        run_client()
```

### 예제 2: 인터페이스 상태 모니터링

```python
# example_monitor.py
"""인터페이스 상태 모니터링"""
from src.interfaces import create_interface, InterfaceConfig
from src.interfaces.factory import print_interface_info

# 사용 가능한 인터페이스 출력
print_interface_info()

# 인터페이스 상태 확인
config = InterfaceConfig()
interface = create_interface('shm', config, is_server=True)
interface.connect()

status = interface.get_status()
print("\n현재 상태:")
for key, value in status.items():
    print(f"  {key}: {value}")

interface.disconnect()
```

### 예제 3: Async DP 엔진과 통합

```python
# example_integration.py
"""Async DP 엔진과 인터페이스 통합 예제"""
import numpy as np
import torch
from src.interfaces import create_interface, InterfaceConfig
from src.models.diffusion_net import DiffusionPolicy
from src.models.scheduler import get_scheduler
from config.settings import Config

def main():
    # 1. 인터페이스 설정
    interface_config = InterfaceConfig(
        action_dim=Config.ACTION_DIM,
        obs_dim=Config.OBS_DIM,
        pred_horizon=Config.PRED_HORIZON,
        shm_name="asyncdp_robot"
    )

    # 2. 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionPolicy(Config.ACTION_DIM, Config.OBS_DIM).to(device)
    model.load_state_dict(torch.load(Config.CKPT_PATH, map_location=device))
    model.eval()

    scheduler = get_scheduler('ddim', num_train_timesteps=100, num_inference_steps=16)

    # 3. 인터페이스 연결
    with create_interface('shm', interface_config, is_server=False) as interface:
        print("로봇에 연결됨. 추론 시작...")

        while True:
            # 상태 읽기
            state = interface.get_state()
            if not state.is_valid:
                continue

            # Diffusion 추론
            with torch.no_grad():
                obs = torch.from_numpy(state.qpos).float().to(device).unsqueeze(0)
                trajectory = model.get_action(obs, scheduler, device=device)

            # 궤적 전송
            interface.send_trajectory(trajectory)

if __name__ == "__main__":
    main()
```

---

## 문제 해결

### 자주 발생하는 오류

#### 1. "Shared memory not found"

```
원인: 서버가 시작되지 않음
해결: 서버(is_server=True)를 먼저 실행하세요
```

#### 2. "gRPC connection timeout"

```
원인: 네트워크 연결 문제 또는 서버 미실행
해결:
  1. 서버 IP/포트 확인
  2. 방화벽 설정 확인
  3. ping 테스트
```

#### 3. "grpcio not installed"

```
원인: gRPC 의존성 미설치
해결: uv pip install grpcio grpcio-tools
```

### 디버깅 팁

```python
# 로깅 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 인터페이스 상태 확인
status = interface.get_status()
print(status)
```

---

## API 레퍼런스

### InterfaceConfig

```python
@dataclass
class InterfaceConfig:
    action_dim: int = 14          # 액션 차원
    obs_dim: int = 14             # 관측 차원
    pred_horizon: int = 16        # 예측 horizon
    shm_name: str = "asyncdp_robot_interface"  # SHM 이름
    grpc_host: str = "localhost"  # gRPC 호스트
    grpc_port: int = 50051        # gRPC 포트
    grpc_timeout: float = 1.0     # 연결 타임아웃
    max_retries: int = 5          # 재시도 횟수
    retry_delay: float = 0.1      # 재시도 간격
```

### RobotInterface 메서드

| 메서드                  | 설명           | 반환값        |
|------------------------|---------------|---------------|
| `connect()`             | 연결 수립     | `bool`        |
| `disconnect()`          | 연결 해제     | `None`        |
| `get_state()`           | 로봇 상태 읽기 | `RobotState` |
| `send_action(action)`   | 단일 액션 전송 | `bool`       |
| `send_trajectory(traj)` | 궤적 전송     | `bool`       |
| `get_status()`          | 상태 정보     | `dict`       |
| `is_connected`          | 연결 상태     | `bool`       |

### RobotState

```python
@dataclass
class RobotState:
    qpos: np.ndarray      # 관절 위치 (obs_dim,)
    qvel: np.ndarray      # 관절 속도 (obs_dim,) [선택]
    timestamp: float      # 타임스탬프
    is_valid: bool        # 유효성 플래그
```

---

## 고급: 커스텀 컨트롤러 만들기

Dynamixel 외의 로봇을 사용하거나 특수한 요구사항이 있는 경우 커스텀 컨트롤러를 만들 수 있습니다.

### BaseRobotController 상속

```python
# my_custom_controller.py
import numpy as np
from src.controllers.base_controller import BaseRobotController, ControllerConfig
from src.interfaces import InterfaceConfig

class MyRobotController(BaseRobotController):
    """
    커스텀 로봇 컨트롤러 예제
    BaseRobotController를 상속하면 다음이 자동으로 제공됩니다:
    - 스레드 안전한 상태 관리
    - 자동 재연결
    - Watchdog 타이머
    - 속도 제한
    """

    def __init__(self, config: ControllerConfig):
        super().__init__(config)
        self.my_robot = None  # 실제 로봇 객체

    def _connect_hardware(self) -> bool:
        """로봇 하드웨어에 연결"""
        try:
            # 예: 시리얼 포트 연결
            # self.my_robot = MyRobotSDK.connect("/dev/ttyUSB0")
            print("[MyRobot] 하드웨어 연결됨")
            return True
        except Exception as e:
            print(f"[MyRobot] 연결 실패: {e}")
            return False

    def _disconnect_hardware(self) -> None:
        """로봇 하드웨어 연결 해제"""
        if self.my_robot:
            # self.my_robot.disconnect()
            pass
        print("[MyRobot] 연결 해제됨")

    def _read_state(self) -> tuple[np.ndarray, np.ndarray]:
        """현재 로봇 상태 읽기 (500Hz로 호출됨)"""
        # 실제 구현:
        # qpos = self.my_robot.get_joint_positions()
        # qvel = self.my_robot.get_joint_velocities()

        # 시뮬레이션:
        qpos = self._current_qpos + np.random.randn(14).astype(np.float32) * 0.001
        qvel = np.zeros(14, dtype=np.float32)

        return qpos, qvel

    def _write_command(self, action: np.ndarray) -> bool:
        """로봇에 명령 전송 (500Hz로 호출됨)"""
        # 실제 구현:
        # self.my_robot.set_joint_positions(action)

        # 시뮬레이션:
        self._current_qpos = action.copy()
        return True


# 사용 예
if __name__ == "__main__":
    config = ControllerConfig(
        control_freq=500.0,
        interface_config=InterfaceConfig(
            action_dim=14,
            obs_dim=14,
            shm_name="my_custom_robot"
        ),
        watchdog_timeout=1.0,
        auto_reconnect=True
    )

    controller = MyRobotController(config)
    controller.start()
```

### 콜백 사용

```python
# 상태 변화 감지
def on_state_change(old_state, new_state):
    print(f"상태 변화: {old_state.value} → {new_state.value}")

# 에러 핸들링
def on_error(error):
    print(f"에러 발생: {error}")
    # 알림 전송, 로그 기록 등

# 액션 수신 감지
def on_action_received(action):
    print(f"새 액션: {action[:3]}...")

controller.set_callbacks(
    on_state_change=on_state_change,
    on_error=on_error,
    on_action_received=on_action_received
)
```

### gRPC 서버 사용 (원격 제어)

```python
# grpc_robot_server.py
from src.interfaces.grpc_server import GrpcRobotServer, GrpcServerConfig
import numpy as np
import time

# 서버 설정
config = GrpcServerConfig(
    host="0.0.0.0",
    port=50051,
    action_dim=14,
    obs_dim=14
)

server = GrpcRobotServer(config)

# 액션 수신 콜백
def handle_action(action):
    print(f"Received action: {action[:3]}...")
    # 로봇에 명령 전송

server.set_action_callback(handle_action)
server.start(blocking=False)

# 상태 업데이트 루프
while True:
    qpos = read_robot_state()  # 사용자 구현
    server.update_state(qpos)
    time.sleep(0.002)  # 500Hz
```

### Proto 컴파일 (gRPC 사용 시)

```bash
# gRPC 의존성 설치
uv pip install grpcio grpcio-tools

# Proto 컴파일
cd src/interfaces/proto
python compile_proto.py

# 생성되는 파일:
# - robot_interface_pb2.py
# - robot_interface_pb2_grpc.py
```

---

## 파일 구조

```
src/
├── interfaces/
│   ├── __init__.py
│   ├── base.py              # RobotInterface 추상 클래스
│   ├── factory.py           # create_interface() 팩토리
│   ├── shm_interface.py     # SharedMemory 구현
│   ├── grpc_interface.py    # gRPC 클라이언트
│   ├── grpc_server.py       # gRPC 서버
│   └── proto/
│       ├── robot_interface.proto  # Proto 정의
│       └── compile_proto.py       # 컴파일 스크립트
│
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py      # 컨트롤러 베이스 클래스
│   └── dynamixel_controller.py # Dynamixel 구현
```
