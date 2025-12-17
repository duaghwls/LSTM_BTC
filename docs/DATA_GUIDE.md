# 📊 데이터 형식 가이드

## BTC-USD.csv 데이터 형식

프로젝트에서 사용하는 비트코인 데이터는 다음과 같은 형식이어야 합니다.

### 필수 컬럼

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| Date | Date | 거래 날짜 (YYYY-MM-DD) | 2024-11-01 |
| Open | Float | 시가 (USD) | 68234.50 |
| High | Float | 고가 (USD) | 69123.75 |
| Low | Float | 저가 (USD) | 67890.25 |
| Close | Float | 종가 (USD) | 68950.00 |
| Volume | Integer | 거래량 | 25647382912 |

### 데이터 샘플

```csv
Date,Open,High,Low,Close,Volume
2024-11-01,68234.50,69123.75,67890.25,68950.00,25647382912
2024-11-02,68950.00,69500.25,68123.50,69234.75,27123456789
2024-11-03,69234.75,70100.00,68900.00,69850.50,28456789012
...
```

## 데이터 획득 방법

### 1. Yahoo Finance (권장)

**웹사이트에서 직접 다운로드:**
1. https://finance.yahoo.com/quote/BTC-USD/history 방문
2. Time Period: 원하는 기간 설정 (예: Jan 01, 2020 - Dec 31, 2024)
3. Show: Historical Prices
4. Frequency: Daily
5. "Download" 버튼 클릭
6. 다운로드한 파일을 `BTC-USD.csv`로 저장

**Python으로 자동 다운로드:**

```python
import yfinance as yf
import pandas as pd

# yfinance 설치: pip install yfinance

# 데이터 다운로드
btc = yf.download(
    'BTC-USD',
    start='2020-01-01',
    end='2024-12-31',
    interval='1d'
)

# CSV로 저장
btc.to_csv('BTC-USD.csv')
print("데이터 다운로드 완료!")
```

### 2. 기타 데이터 소스

- **CoinGecko API**: https://www.coingecko.com/en/api
- **Binance API**: https://binance-docs.github.io/apidocs/
- **Coinbase API**: https://docs.cloud.coinbase.com/

## 데이터 전처리

프로젝트는 자동으로 다음 전처리를 수행합니다:

1. **날짜 분할**: 2024-11-12 기준 Train/Test 분리
2. **Percentage Change**: 가격 변화율 계산
3. **MinMax Scaling**: 0~1 범위로 정규화
4. **시퀀스 생성**: 슬라이딩 윈도우 방식

## 주의사항

⚠️ **데이터 품질**:
- 결측치(NaN)가 없는지 확인
- 날짜가 연속적인지 확인
- 거래량이 0이 아닌지 확인

⚠️ **데이터 크기**:
- 최소 1년 이상의 데이터 권장
- Train/Test 비율 고려 (현재: 2024-11-12 기준 분할)

## 커스텀 데이터 사용

다른 암호화폐나 주식 데이터를 사용하려면:

1. 위 형식에 맞춰 CSV 파일 준비
2. `main.py`의 다음 라인 수정:
   ```python
   df = pd.read_csv("YOUR_DATA.csv")  # 파일명 변경
   ```
3. 날짜 분할 기준 수정 (필요시):
   ```python
   train_data = df[df["Date"] <= "2024-XX-XX"]  # 날짜 변경
   test_data = df[df["Date"] > "2024-XX-XX"]
   ```

## 문제 해결

### 문제: "FileNotFoundError: BTC-USD.csv"

**해결**: 
```bash
# 현재 디렉토리 확인
ls  # macOS/Linux
dir  # Windows

# BTC-USD.csv가 같은 폴더에 있는지 확인
```

### 문제: 날짜 파싱 오류

**해결**: CSV 파일의 Date 컬럼이 `YYYY-MM-DD` 형식인지 확인

### 문제: 데이터가 너무 적음

**해결**: 더 긴 기간의 데이터를 다운로드하거나, `input_seq_length`를 줄여보세요
