# 🔧 상세 설치 가이드

## 시스템 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)
- (선택) CUDA 지원 GPU (빠른 학습을 원할 경우)

## 단계별 설치

### 1️⃣ Python 설치 확인

```bash
python --version
```

Python 3.8 이상이 설치되어 있어야 합니다.

### 2️⃣ 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/LSTM_BTC.git
cd LSTM_BTC
```

### 3️⃣ 가상환경 생성 및 활성화

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4️⃣ 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5️⃣ (선택) PyTorch GPU 버전 설치

CUDA를 사용할 수 있는 환경이라면:

```bash
# CUDA 11.8용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

더 자세한 내용은 [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)를 참고하세요.

### 6️⃣ 데이터 다운로드

Yahoo Finance에서 BTC-USD 데이터를 다운로드합니다:

**방법 1: 수동 다운로드**
1. [Yahoo Finance BTC-USD 페이지](https://finance.yahoo.com/quote/BTC-USD/history) 방문
2. 기간 설정 (예: 2020-01-01 ~ 현재)
3. "Download" 클릭
4. 다운로드한 파일을 `BTC-USD.csv`로 이름 변경하여 프로젝트 루트에 저장

**방법 2: Python 스크립트 (선택)**
```python
import yfinance as yf

# yfinance 먼저 설치: pip install yfinance
btc = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31')
btc.to_csv('BTC-USD.csv')
```

### 7️⃣ 설치 확인

```bash
python -c "import torch; import pandas; import sklearn; print('모든 패키지 설치 완료!')"
```

## 🎯 다음 단계

설치가 완료되었다면:

1. `python main.py` - 모델 학습 시작
2. `python visualize.py` - 예측 결과 시각화

## 🐛 문제 해결

### 문제: ModuleNotFoundError

**해결**: 가상환경이 활성화되어 있는지 확인하고, `pip install -r requirements.txt` 재실행

### 문제: CUDA 오류

**해결**: CPU 버전으로 실행됩니다. 문제가 없으면 그대로 진행하세요. GPU를 사용하려면 CUDA 버전 확인 후 올바른 PyTorch 설치

### 문제: 메모리 부족

**해결**: `main.py`에서 `batch_size`를 16 또는 8로 줄여보세요:
```python
batch_size = 16  # 기본값 32에서 줄임
```

## 📚 추가 자료

- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Pandas 문서](https://pandas.pydata.org/docs/)
- [Scikit-learn 가이드](https://scikit-learn.org/stable/user_guide.html)
