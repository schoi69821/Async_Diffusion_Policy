# Async DP 훈련 보고서

## 1. 프로젝트 개요

**목표**: VX300s 로봇 팔이 카메라 영상을 보고 펜을 집어 들어올렸다 내려놓는 동작을 자율 수행
**방법**: Vision-based Diffusion Policy (Chi et al., 2023) — 카메라 이미지 + 관절 위치로부터 미래 행동 궤적을 예측

---

## 2. 데이터셋

### 수집 환경
- **로봇**: Interbotix VX300s (6 DOF + 그리퍼, Dynamixel XM430/XM540)
- **카메라**: Logi C920e (손목 장착), 224x224 리사이즈
- **수집 방식**: 리더 팔을 사람이 조작 → 팔로워가 미러링 → 카메라/관절 기록
- **수집 빈도**: 50Hz
- **수집 스크립트**: `scripts/collect_episodes.py`

### 데이터 구조
```
episodes/pen_fixed/          # 50Hz 원본 (3.2GB, 20 에피소드)
episodes/pen_fixed_15hz/     # 15Hz 다운샘플 (976MB, 20 에피소드)
```

| 에피소드 | 프레임 수 (50Hz) | 시간 | 주요 동작 |
|----------|-----------------|------|-----------|
| 평균 | ~2,054 | ~41초 | 홈→하강→접근→잡기→들기→놓기→복귀 |
| 최소 | 1,573 | ~31초 | - |
| 최대 | 2,761 | ~55초 | - |

### HDF5 구조
```
/observations/qpos:    (T, 7)           float32  관절 위치 (6 DOF + 그리퍼)
/observations/images:  (T, 224, 224, 3) uint8    카메라 이미지
/action:               (T, 7)           float32  명령 위치 (리더 → 팔로워)
```

### 에피소드 타임라인 (Episode 0, 50Hz)
```
t=0~50     (prog 0.00~0.02): 홈 위치, 그리퍼 닫힘 → 열림 (-24°→-68°)
t=50~300   (prog 0.02~0.12): 대기 후 하강 시작
t=300~800  (prog 0.12~0.32): 팔 하강 (shoulder +96°→-11°, elbow -93°→-7°)
t=800~850  (prog 0.32~0.34): 펜 위에서 대기, 그리퍼 열림 (-35°)
t=850~900  (prog 0.34~0.36): ★ 그리퍼 닫힘 (-35°→-71°) — 0.24초 급격 전환
t=900~1500 (prog 0.36~0.61): 펜 잡고 들어올리기/내려놓기
t=1500~1600(prog 0.61~0.65): 그리퍼 열림 — 펜 놓기
t=1600~2468(prog 0.65~1.00): 홈 복귀
```

### 그리퍼 캘리브레이션
```
리더:    MIN=1545 (닫힘), MAX=2187 (열림)
팔로워:  MIN=1050 (닫힘), MAX=1965 (열림)
매핑:    리더 → 팔로워 선형 변환
라디안:  열림=-0.61 rad (-35°), 닫힘=-1.24 rad (-71°)
```

---

## 3. 모델 아키텍처

### Vision Diffusion Policy (`src/models/vision_policy.py`)

```
Input:
  이미지: (B, [T,] 3, 224, 224) — obs_horizon 프레임
  관절:   (B, [T,] 7) — 정규화된 qpos
  진행도: (B, 1) — 0.0~1.0

VisionEncoder (ResNet18, ImageNet pretrained):
  이미지 → 256-dim (초기 6블록 freeze, 마지막 2블록 fine-tune)

QposEncoder (MLP):
  7-dim → LayerNorm → GELU → 256-dim

ProgressEncoder (MLP):
  1-dim → GELU → 256-dim

ObsFusion:
  [img_feat, qpos_feat] × obs_horizon + progress_feat → 256-dim

ConditionalUNet1D (Diffusion Backbone):
  입력: (B, 7, 16) 노이즈 액션
  조건: 256-dim obs embedding (FiLM으로 모든 ResBlock에 주입)
  채널: (256, 512), encoder-bottleneck-decoder with skip connections
  출력: (B, 7, 16) 예측된 노이즈

Output: (B, 16, 7) — 16 스텝 행동 궤적
```

**파라미터**: ~19.3M (훈련 가능: ~18.6M)

### Gripper Classifier (`src/models/gripper_classifier.py`)

```
입력: 이미지 + 관절(그리퍼 제외, 6-dim) + 진행도
ResNet18 → 256-dim
QposEncoder(6-dim) → 128-dim
ProgressEncoder → 64-dim
Concat → MLP(256→64→1) → sigmoid

출력: P(그리퍼 닫힘) — 이진 분류
```

**파라미터**: ~11.6M

---

## 4. 훈련 이력

### v4: 첫 번째 본격 훈련 (50Hz 데이터, obs_horizon=1)

| 항목 | 값 |
|------|---|
| 데이터 | `pen_fixed` 50Hz, 41,000 샘플 |
| obs_horizon | 1 |
| temporal_stride | 1 |
| batch_size | 32 |
| epochs | 155 (early stopping) |
| best val_loss | **0.000963** |
| 16-step 윈도우 | 0.32초 (50Hz × 16) |
| epoch 시간 | ~77초 |

**결과**: 노이즈 예측 자체는 잘 학습됨. 그러나 실행 시:
- 50Hz 학습 → 15Hz 실행 = **3.3배 시간 불일치**
- 팔이 극도로 느리게 움직여 펜에 도달 못함
- Progress 신호 무효화 (어떤 progress에서도 동일 출력)

### v5: 15Hz 다운샘플 데이터 (obs_horizon=1)

| 항목 | 값 |
|------|---|
| 데이터 | `pen_fixed_15hz` 15Hz, **12,000 샘플 (v4의 30%)** |
| obs_horizon | 1 |
| temporal_stride | 1 |
| batch_size | 32 |
| epochs | 192 |
| best val_loss | **0.003100** (v4의 3.2배 나쁨) |
| 16-step 윈도우 | 1.07초 (15Hz × 16) |
| epoch 시간 | ~20초 |

**결과**: 시간 불일치 해소. 하지만:
- 데이터 70% 감소 → 학습 품질 저하
- 펜까지 도달은 하지만 즉시 복귀
- 그리퍼 여전히 안 닫힘

### v6: obs_horizon=2 (15Hz 다운샘플 데이터)

| 항목 | 값 |
|------|---|
| 데이터 | `pen_fixed_15hz` 15Hz, 12,000 샘플 |
| obs_horizon | **2** (속도 정보 포함) |
| temporal_stride | 1 |
| batch_size | 32 → 16 (GPU OOM) |
| epochs | 72 (CUDA 크래시로 중단) |
| best val_loss | **0.007027** (v5보다 나쁨) |

**결과**: 데이터 부족 + obs_horizon=2로 파라미터 증가 → 학습 악화. CUDA OOM으로 반복 크래시.

### v7: Temporal Stride (50Hz 원본 + stride=3 + obs_horizon=2) — 현재 버전

| 항목 | 값 |
|------|---|
| 데이터 | `pen_fixed` 50Hz, **40,150 샘플 (v5의 3.3배)** |
| obs_horizon | **2** |
| temporal_stride | **3** (50Hz에서 매 3프레임, ~16.7Hz 효과) |
| batch_size | 16 (GPU 6GB 제약) |
| epochs | 100 (원격 서버에서 완료) |
| best val_loss | **0.002722** |
| 16-step 윈도우 | 0.9초 (stride 3 × 15프레임 / 50Hz) |

**결과**:
- 궤적: 펜 위치까지 도달하고 **35 스텝(~23초) 체류** ✓
- 그리퍼: diffusion 모델은 여전히 -33°~-35° (열림) 유지 ✗
- 진동: 여전히 1~2도 범위로 까딱거림

### Gripper Classifier: 이진 그리퍼 분류기

| 항목 | 값 |
|------|---|
| 데이터 | `pen_fixed` 50Hz + stride=3, 40,150 샘플 |
| obs_horizon | 2 |
| qpos_dim | **6** (그리퍼 제외 — 데이터 누수 방지) |
| 라벨 | "16스텝 내 그리퍼 닫힘 여부" (미래 예측) |
| batch_size | 32 |
| epochs | 30 |
| best val_acc | **97.1%** |

**결과**:
- 첫 버전 (qpos 7-dim, 현재 상태 라벨): val_acc=99.8% — **데이터 누수** (qpos 그리퍼 값을 그대로 읽음)
- 수정 버전 (qpos 6-dim, 미래 라벨): val_acc=97.1% — 그러나 실행 시 `grip_p` 최대 0.10으로 임계값(0.5)에 못 미침
- 카메라 뷰가 훈련 시와 달라 일반화 실패

---

## 5. 추론 파이프라인 (`scripts/run_vision_policy.py`)

### Chi et al. 구현 요소

| 요소 | 상태 | 설명 |
|------|------|------|
| **Temporal Ensemble** | 구현됨 | 겹치는 예측의 가중평균 (k=0.01) |
| **Action Chunking** | 구현됨 | 16스텝 중 8스텝 실행 후 재예측 |
| **Deterministic Denoising** | 구현됨 | 고정 시드(42) generator |
| **obs_horizon=2** | 구현됨 | 2프레임 관측 버퍼 |
| **위치 기반 그리퍼** | 구현됨 | 그립 존 진입 시 강제 닫힘 |

### 그리퍼 제어 로직 (최종)
```
1. Diffusion 모델이 자연스럽게 그리퍼 제어 (초기 하강 패턴 유지)
2. 팔이 그립 존 진입 시 (shoulder < +5°, elbow -30°~-10°):
   - 3스텝 이상 체류 → 그리퍼 강제 닫힘 (-71°)
3. 잡은 후 progress가 25% 더 진행 → 그리퍼 강제 열림 (-35°)
```

### 안전 기능
- 관절 한계 클램핑 (7관절 각각)
- 스텝당 최대 이동량: 0.15 rad (8.6°)
- 관절 읽기 실패 시 비상 정지
- 안전 경유점을 통한 홈 복귀 (현위치 → MID → HOME)

---

## 6. 발견된 문제점과 원인 분석

### 문제 1: 50Hz→15Hz 시간 스케일 불일치 (v4) — ✅ 해결됨

**증상**: 팔이 극도로 느리게 움직여 펜에 도달 못함
**원인**: 모델은 16스텝=0.32초(50Hz) 궤적을 예측하지만, 실행은 16스텝=1.07초(15Hz)
**해결**: temporal_stride=3으로 50Hz 데이터에서 ~15Hz 간격으로 학습

### 문제 2: 그리퍼 미작동 (Diffusion Policy의 구조적 한계) — ⚠️ 부분 해결

**증상**: 그리퍼가 -33°~-35° (열림)에 머무르고 -71° (닫힘)으로 전환 안됨
**원인**:
- 그리퍼는 열림/닫힘 **이산적 이진 상태**
- Diffusion Policy는 연속 분포를 모델링 → 두 모드를 **평균** → 항상 열림쪽 예측
- 훈련 데이터에서 열림 58%, 닫힘 35%, 전환 7% — 다수결로 열림 승리
- 0.24초의 급격 전환은 16-step 윈도우에서도 평균화됨

**부분 해결**: 위치 기반 그리퍼 트리거 (그립 존 3스텝 체류 → 강제 닫힘)

### 문제 3: 진동/까딱거림 — ⚠️ 감소했으나 미해결

**증상**: 팔이 1~2도 범위로 지속적으로 떨림
**원인**:
- 매 추론마다 랜덤 노이즈에서 시작 → 약간 다른 궤적
- 연속 예측 간 불일치 → 팔이 두 예측 사이에서 진동
- Temporal ensemble의 `ens=1` — chunk_size=8과 pred_horizon=16이면 이전 예측이 정확히 만료
- 결과적으로 앙상블이 2개 이상의 예측을 동시에 활용하지 못함

**시도한 완화책**:
- EMA 0.9→0.6 (약간 개선)
- Deterministic denoising (seed=42) (약간 개선)
- Temporal ensemble (구조적으로 overlap 부족)

**근본 해결 방안**: chunk_size를 4로 줄여 실제 overlap 확보, 또는 noise conditioning으로 연속 예측 일관성 확보

### 문제 4: 궤적 비일관성 (실행마다 다른 경로) — ❌ 미해결

**증상**: 동일 체크포인트, 동일 시드로도 실행마다 다른 궤적
**원인**:
- 카메라 이미지가 매 실행마다 다름 (조명, 펜 위치, 팔 시작 각도 미세 차이)
- 20개 에피소드만으로는 이런 변동을 커버할 수 없음
- 모델이 특정 카메라 뷰에 과적합
- 그리퍼 오버라이드가 모델의 궤적 예측을 변경 (그리퍼 상태가 모델 입력의 일부)

### 문제 5: 잡은 후 들어올리기 동작 없음 — ❌ 미해결

**증상**: 그립 존에서 그리퍼가 닫힌 후 팔이 제자리에 머무름
**원인**:
- 그리퍼가 닫힌 상태의 관측(qpos -71°)을 모델이 충분히 보지 못함
- 훈련 중 그리퍼 닫힘 구간은 전체의 35%뿐
- 위치 기반 그리퍼 오버라이드로 인해 qpos에 -71°가 입력되면 모델이 미경험 상태에 돌입
- Progress=1.0 이후에도 모델이 "복귀" 궤적을 생성하지 못함

### 문제 6: GPU 메모리 제약 — 반복 발생

**증상**: RTX 3060 Laptop (6GB)에서 CUDA OOM 또는 launch failure
**원인**: obs_horizon=2 → 이미지 2장 처리, batch_size=32 불가
**완화**: batch_size=16, num_workers=0
**근본 해결**: 원격 서버(더 큰 GPU)에서 훈련

---

## 7. 버전별 성능 비교

| 버전 | 데이터 | 샘플 수 | 윈도우 | obs | val_loss | 궤적 | 그리퍼 | 진동 |
|------|--------|---------|--------|-----|----------|------|--------|------|
| v4 | 50Hz | 41K | 0.32s | 1 | **0.001** | 느림 | ✗ | 심함 |
| v5 | 15Hz | 12K | 1.07s | 1 | 0.003 | 빠르지만 불안정 | ✗ | 보통 |
| v6 | 15Hz | 12K | 1.07s | 2 | 0.007 | 나쁨 | ✗ | 보통 |
| **v7** | **50Hz+s3** | **40K** | **0.9s** | **2** | **0.003** | **좋음** | **위치기반** | **보통** |

---

## 8. 현재 상태 및 권장 다음 단계

### 현재 동작
1. 홈 → 펜 위치까지 하강 ✅
2. 펜 위에서 체류 ✅
3. 그리퍼 닫힘 (위치 기반 트리거) ✅
4. 펜 들어올리기 ❌ (제자리 머무름)
5. 펜 내려놓기 ❌
6. 홈 복귀 ⚠️ (타임아웃 후 safe_go_home으로 복귀)

### 권장 다음 단계 (우선순위순)

1. **에피소드 추가 수집 (50~100개)**
   - 현재 20개는 부족. 다양한 펜 위치, 조명에서 수집
   - 그리퍼 닫힌 상태에서의 관측 비율 증가
   - 가장 확실한 개선 방법

2. **Action Chunking 개선**
   - chunk_size=4로 줄여 temporal ensemble overlap 확보
   - 진동 감소 기대

3. **그리퍼 제어 분리 아키텍처**
   - Diffusion Policy: 6관절 (팔)만 제어
   - 별도 분류기: 그리퍼 open/close
   - 훈련 시 action_dim=6으로 학습, 그리퍼 별도 학습

4. **ACT (Action Chunking with Transformers) 전환 검토**
   - Diffusion Policy 대비 bimodal action에 강점
   - CVAE 기반으로 그리퍼 전환 학습에 유리
   - 같은 데이터로 재학습 가능

---

## 9. 파일 구조

```
async_dp/
├── checkpoints/
│   ├── vision_policy/          # v1 (100 epochs)
│   ├── vision_policy_v2/       # v2 (200 epochs)
│   ├── vision_policy_v3/       # v3 (200 epochs)
│   ├── vision_policy_v4/       # v4 (155 epochs, 50Hz)
│   ├── vision_policy_v5/       # v5 (192 epochs, 15Hz)
│   ├── vision_policy_v6/       # v6 (172 epochs, 15Hz+obs2)
│   ├── vision_policy_v7/       # v7 (100 epochs, 50Hz+stride3+obs2) ← 현재
│   └── gripper_classifier/     # 이진 분류기
├── episodes/
│   ├── pen_fixed/              # 50Hz 원본 (3.2GB)
│   └── pen_fixed_15hz/         # 15Hz 다운샘플 (976MB)
├── scripts/
│   ├── collect_episodes.py     # 데이터 수집
│   ├── train_vision_policy.py  # Diffusion Policy 훈련
│   ├── train_gripper_classifier.py  # 그리퍼 분류기 훈련
│   ├── downsample_episodes.py  # 50Hz→15Hz 변환
│   └── run_vision_policy.py    # 추론 실행 (Chi et al. pipeline)
└── src/models/
    ├── vision_policy.py        # VisionDiffusionPolicy
    ├── gripper_classifier.py   # GripperClassifier
    └── scheduler.py            # DDPM/DDIM 스케줄러
```
