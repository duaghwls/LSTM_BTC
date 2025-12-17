# 🚀 Bitcoin Price Prediction with LSTM

PyTorch 기반 LSTM 신경망을 활용한 비트코인(BTC-USD) 가격 예측 프로젝트입니다.

## 📊 프로젝트 개요

이 프로젝트는 **Encoder-Decoder LSTM** 구조를 사용하여 과거 10일간의 비트코인 가격 데이터를 바탕으로 미래 2일간의 가격을 예측합니다.

### 주요 특징

- **모델**: Encoder-Decoder LSTM (3 layers, 128 hidden units)
- **입력**: 과거 10일 가격 데이터 (Open, High, Low, Close, Volume)
- **출력**: 미래 2일 종가 예측
- **전처리**: Percentage Change + MinMax Scaling
- **최적화**: Adam Optimizer, Learning Rate Scheduling, Early Stopping

## 🛠️ 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/YOUR_USERNAME/LSTM_BTC.git
cd LSTM_BTC
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

## 📁 프로젝트 구조

```
LSTM_BTC/
│
├── src/                             # 소스 코드
│   ├── main.py                      # 모델 학습 스크립트
│   └── visualize.py                 # 예측 시각화 스크립트
│
├── docs/                            # 문서
│   ├── SETUP.md                     # 설치 가이드
│   ├── DATA_GUIDE.md                # 데이터 가이드
│   └── MODEL_ARCHITECTURE.md        # 모델 아키텍처
│
├── data/                            # 데이터 폴더
│   └── BTC-USD.csv                  # 비트코인 가격 데이터 (직접 다운로드 필요)
│
├── outputs/                         # 결과물 폴더
│   ├── models/                      # 학습된 모델
│   │   └── lstm_btc_model_best.pth  # 최적 모델 (학습 후 생성)
│   └── figures/                     # 생성된 그래프
│       ├── training_validation_loss.png  # 학습 과정 (학습 후 생성)
│       └── btc_prediction_test_*.png     # 예측 결과 (visualize.py 실행 후)
│
├── requirements.txt                 # 필요 패키지 목록
├── README.md                        # 프로젝트 문서
├── LICENSE                          # MIT 라이선스
├── .gitignore                       # Git 제외 파일 설정
└── .gitattributes                   # Git 파일 속성
```

## 🚀 사용 방법

### 1. 데이터 준비

Yahoo Finance에서 BTC-USD 데이터를 다운로드하여 `data/BTC-USD.csv`로 저장합니다.

**필수 컬럼**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

### 2. 모델 학습

```bash
cd src
python main.py
```

**학습 결과물**:
- `outputs/models/lstm_btc_model_best.pth`: 최적 성능 모델 가중치
- `outputs/figures/training_validation_loss.png`: 학습/검증 손실 그래프

**주요 하이퍼파라미터** (`main.py` 내에서 수정 가능):
```python
input_seq_length = 10      # 입력 시퀀스 길이 (과거 10일)
output_seq_length = 2      # 출력 시퀀스 길이 (미래 2일)
hidden_size = 128          # LSTM 은닉 유닛 수
num_layers = 3             # LSTM 레이어 수
learning_rate = 0.0001     # 학습률
num_epochs = 200           # 최대 에폭
batch_size = 32            # 배치 크기
patience = 30              # Early Stopping patience
```

### 3. 예측 시각화

```bash
cd src
python visualize.py
```

**생성 파일**:
- `outputs/figures/btc_prediction_test_*.png`: 테스트 데이터 예측 결과 시각화 (3x2 그리드)

각 그래프는 다음 정보를 포함합니다:
- 📈 과거 실제 가격 (파란선)
- 📉 미래 실제 가격 (초록선)
- 🔴 예측 가격 (빨간 점선)
- 📊 성능 지표 (RMSE, MAE, MAPE)

## 📈 모델 아키텍처

```
Input (batch, 10, 5)
    ↓
Encoder LSTM (3 layers, 128 hidden)
    ↓
Hidden State
    ↓
Decoder LSTM (3 layers, 128 hidden)
    ↓
Fully Connected Layer
    ↓
Output (batch, 2, 5) → Close 가격만 사용
```

### 주요 기법

1. **Percentage Change**: 가격의 절대값 대신 변화율 사용
2. **MinMax Scaling**: 0~1 범위로 정규화
3. **Gradient Clipping**: max_grad_norm=1.0
4. **Early Stopping**: patience=30
5. **Learning Rate Scheduling**: ReduceLROnPlateau

## 📊 성능 지표

모델 성능은 다음 지표로 평가됩니다:

- **RMSE** (Root Mean Squared Error): 예측 오차의 제곱근
- **MAE** (Mean Absolute Error): 평균 절대 오차
- **MAPE** (Mean Absolute Percentage Error): 평균 절대 백분율 오차

`visualize.py` 실행 시 콘솔에 전체 성능 요약이 출력됩니다.

## 🔧 커스터마이징

### 예측 기간 변경

`main.py`와 `visualize.py`의 다음 변수를 수정:

```python
input_seq_length = 20   # 과거 20일 사용
output_seq_length = 5   # 미래 5일 예측
```

### 모델 구조 변경

`MultivariateLSTM` 클래스의 파라미터 수정:

```python
hidden_size = 256      # 은닉 유닛 증가
num_layers = 4         # 레이어 추가
```

## ⚠️ 주의사항

- 이 모델은 **교육 및 연구 목적**으로 제작되었습니다
- 실제 투자 결정에 사용하지 마세요
- 과거 데이터 기반 예측이므로 미래 성능을 보장하지 않습니다

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

