import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


# main.py와 동일한 모델 클래스
class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.input_size = input_size

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )

        # Output layer - 모든 feature 예측
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encoder
        encoder_output, (hidden, cell) = self.encoder_lstm(x)

        # Decoder
        decoder_input = x[:, -1:, :]
        decoder_outputs = []

        for _ in range(self.seq_length):
            decoder_output, (hidden, cell) = self.decoder_lstm(
                decoder_input, (hidden, cell)
            )
            output = self.fc(decoder_output)
            decoder_outputs.append(output)
            decoder_input = output

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs


# main.py와 동일한 시퀀스 생성 함수
def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(input_seq_length, len(data) - output_seq_length + 1):
        X.append(data[i - input_seq_length : i])
        y.append(data[i : i + output_seq_length, 0])
    return np.array(X), np.array(y)


# pct_change 역변환 함수
def inverse_pct_change(pct_values, initial_price):
    """
    pct_change로 변환된 값을 실제 가격으로 복원
    """
    prices = [initial_price]
    for pct in pct_values:
        next_price = prices[-1] * (1 + pct)
        prices.append(next_price)
    return np.array(prices[1:])


print("=" * 80)
print("BTC 가격 예측 시각화")
print("=" * 80)

# ========== 1. 데이터 로드 및 전처리 (main.py와 동일) ==========
df = pd.read_csv("../data/BTC-USD.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 원본 데이터 저장
original_data = df.copy()

train_data = df[df["Date"] <= "2024-11-12"]
test_data = df[df["Date"] > "2024-11-12"]

train_data = train_data.drop(columns=["Date"])
test_data = test_data.drop(columns=["Date"])

# differencing the data (main.py와 동일)
train_data_pct = train_data.copy()
test_data_pct = test_data.copy()
for col in train_data.columns:
    train_data_pct[col] = train_data[col].pct_change()
    test_data_pct[col] = test_data[col].pct_change()

train_data_pct = train_data_pct.dropna()
test_data_pct = test_data_pct.dropna()

# 정규화 적용 (main.py와 동일)
scaler = MinMaxScaler()
train_data_pct_scaled = scaler.fit_transform(train_data_pct)
test_data_pct_scaled = scaler.transform(test_data_pct)

# 윈도우 크기 설정 (main.py와 동일)
input_seq_length = 10  # 과거 10일 학습
output_seq_length = 2  # 미래 2일 예측

# 전체 데이터로 시퀀스 생성
all_data_pct_scaled = np.concatenate([train_data_pct_scaled, test_data_pct_scaled])
X, y = create_sequences(all_data_pct_scaled, input_seq_length, output_seq_length)

print(f"데이터 로드 완료")
print(f"  - 총 원본 데이터: {len(df)}개")
print(f"  - Train 데이터: {len(train_data_pct_scaled)}개")
print(f"  - Test 데이터: {len(test_data_pct_scaled)}개")
print(f"  - 생성된 시퀀스 수: {len(X)}개")
print(f"  - 입력 shape: {X.shape} (과거 {input_seq_length}일)")
print(f"  - 출력 shape: {y.shape} (미래 {output_seq_length}일)")

# ========== 2. 모델 로드 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X.shape[2]
hidden_size = 128
num_layers = 3

model = MultivariateLSTM(input_size, hidden_size, num_layers, output_seq_length).to(
    device
)
model.load_state_dict(torch.load("../outputs/models/lstm_btc_model_best.pth", map_location=device))
model.eval()

print(f"\n모델 로드 완료 (Device: {device})")

# ========== 3. 예측 수행 ==========
X_tensor = torch.FloatTensor(X).to(device)
with torch.no_grad():
    predictions = model(X_tensor)  # (num_sequences, output_seq_length, input_size)

predictions = predictions.cpu().numpy()
print(f"예측 완료: {predictions.shape}")

# ========== 4. 시각화 준비 ==========
# 인덱스 계산: pct_change로 1개 행이 줄어든 것을 고려
# original_data의 인덱스 = 1(pct_change 손실) + input_seq_length + seq_idx
pct_offset = 1  # pct_change dropna로 인한 offset

print("\n" + "=" * 80)
print(f"Test 데이터 예측 시각화")
print("=" * 80)

# Test 데이터 시작 시점의 시퀀스 인덱스 찾기
test_start_date = pd.to_datetime("2024-11-12")
test_start_seq_idx = None

for seq_idx in range(len(X)):
    original_idx = pct_offset + input_seq_length + seq_idx
    if original_idx < len(original_data):
        pred_start_date = original_data.iloc[original_idx]["Date"]
        if pd.to_datetime(pred_start_date) >= test_start_date:
            test_start_seq_idx = seq_idx
            break

if test_start_seq_idx is None:
    test_start_seq_idx = max(0, len(X) - 20)  # 최소 마지막 20개

# Test 데이터 기간의 모든 시퀀스 수 계산
num_test_sequences = len(X) - test_start_seq_idx
print(f"Test 데이터 시작 시퀀스 인덱스: {test_start_seq_idx}")
print(f"Test 데이터 기간 총 시퀀스 수: {num_test_sequences}")

# 최대 20개 시각화 (5x4 그리드)
max_plots = min(20, num_test_sequences)

# 간격 계산 (전체를 균등하게 샘플링)
if num_test_sequences <= max_plots:
    prediction_indices = list(range(test_start_seq_idx, len(X)))
else:
    step = num_test_sequences // max_plots
    prediction_indices = [test_start_seq_idx + i * step for i in range(max_plots)]

print(f"시각화할 예측 수: {len(prediction_indices)}")
print(f"선택된 시퀀스 인덱스 범위: {prediction_indices[0]} ~ {prediction_indices[-1]}")

# ========== 5. 그래프 생성 ==========
# 한 파일당 최대 6개의 그래프 (3x2 그리드)
plots_per_file = 6
n_files = (len(prediction_indices) + plots_per_file - 1) // plots_per_file

print(f"총 {n_files}개의 그래프 파일 생성 예정 (각 파일당 최대 {plots_per_file}개)")

all_metrics = []
saved_files = []

for file_idx in range(n_files):
    start_idx = file_idx * plots_per_file
    end_idx = min(start_idx + plots_per_file, len(prediction_indices))
    current_indices = prediction_indices[start_idx:end_idx]

    n_plots_in_file = len(current_indices)

    # 3x2 그리드
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()

    print(f"\n파일 {file_idx + 1}/{n_files} 생성 중... ({n_plots_in_file}개 그래프)")

    for plot_idx, seq_idx in enumerate(current_indices):
        ax = axes[plot_idx]

        # 예측값 추출 (Close 가격)
        prediction_scaled = predictions[seq_idx, :, 0]  # (output_seq_length,)

        # 역변환 1단계: MinMaxScaler inverse
        prediction_full = np.zeros((output_seq_length, input_size))
        prediction_full[:, 0] = prediction_scaled
        prediction_pct = scaler.inverse_transform(prediction_full)[:, 0]

        # 역변환 2단계: pct_change inverse
        # 예측 시작 시점의 원본 데이터 인덱스 계산
        original_idx = pct_offset + input_seq_length + seq_idx
        initial_price = original_data.iloc[original_idx]["Close"]

        # 예측 가격 계산
        predicted_prices = inverse_pct_change(prediction_pct, initial_price)

        # 실제 가격 추출
        actual_start_idx = original_idx + 1
        actual_end_idx = min(actual_start_idx + output_seq_length, len(original_data))
        actual_prices = original_data.iloc[actual_start_idx:actual_end_idx][
            "Close"
        ].values

        # 날짜 정보
        pred_start_date = original_data.iloc[original_idx]["Date"]
        prediction_dates = pd.date_range(
            start=pred_start_date, periods=output_seq_length + 1, freq="D"
        )[1:]

        # 과거 데이터 (60일)
        historical_period = 10
        hist_start_idx = max(0, original_idx - historical_period)
        hist_data = original_data.iloc[hist_start_idx : original_idx + 1]
        hist_dates = hist_data["Date"].values
        hist_prices = hist_data["Close"].values

        # 그래프 그리기
        ax.plot(
            hist_dates,
            hist_prices,
            "b-",
            linewidth=2.5,
            label="Historical Actual Price",
            marker="o",
            markersize=2,
            alpha=0.8,
        )

        if len(actual_prices) > 0:
            ax.plot(
                prediction_dates[: len(actual_prices)],
                actual_prices,
                "g-",
                linewidth=2.5,
                label="Future Actual Price",
                marker="s",
                markersize=4,
            )

        ax.plot(
            prediction_dates,
            predicted_prices,
            "r--",
            linewidth=2.5,
            label="Predicted Price",
            marker="^",
            markersize=4,
            alpha=0.8,
        )

        # 예측 시작점 강조
        ax.axvline(
            x=pred_start_date, color="black", linestyle=":", linewidth=2, alpha=0.6
        )
        ax.text(
            pred_start_date,
            ax.get_ylim()[1] * 0.98,
            "Prediction Start",
            rotation=90,
            verticalalignment="top",
            fontsize=9,
            color="black",
        )

        # 성능 지표 계산
        if len(actual_prices) > 0:
            pred_subset = predicted_prices[: len(actual_prices)]
            mse = np.mean((pred_subset - actual_prices) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_subset - actual_prices))
            mape = np.mean(np.abs((actual_prices - pred_subset) / actual_prices)) * 100

            all_metrics.append(
                {
                    "idx": seq_idx,
                    "date": pd.to_datetime(pred_start_date).strftime("%Y-%m-%d"),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                }
            )

            # 성능 지표 표시
            metrics_text = (
                f"RMSE: ${rmse:,.0f}\n" f"MAE: ${mae:,.0f}\n" f"MAPE: {mape:.2f}%"
            )
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                fontweight="bold",
            )

        # 축 설정
        ax.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax.set_ylabel("BTC Price (USD)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Prediction #{start_idx + plot_idx + 1} - Start: {pd.to_datetime(pred_start_date).strftime('%Y-%m-%d')} (Seq {seq_idx})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=45, labelsize=9)

        # y축 포맷
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # 남은 빈 서브플롯 제거
    for idx in range(n_plots_in_file, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    # 파일명 생성
    if n_files == 1:
        filename = "../outputs/figures/btc_prediction_test.png"
    else:
        filename = f"../outputs/figures/btc_prediction_test_{file_idx + 1:02d}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    saved_files.append(filename)
    print(f"  ✅ 저장 완료: {filename}")
    plt.close(fig)

print(f"\n✅ 총 {len(saved_files)}개 그래프 파일 저장 완료")

# ========== 6. 성능 지표 요약 ==========
if all_metrics:
    print("\n" + "=" * 80)
    print("예측 성능 요약")
    print("=" * 80)
    print(
        f"{'예측 #':<8} {'시작 날짜':<12} {'시퀀스':<8} {'RMSE':<15} {'MAE':<15} {'MAPE':<10}"
    )
    print("-" * 80)

    total_rmse, total_mae, total_mape = 0, 0, 0
    for i, m in enumerate(all_metrics):
        print(
            f"{i+1:<8} {m['date']:<12} {m['idx']:<8} "
            f"${m['rmse']:>12,.2f}  ${m['mae']:>12,.2f}  {m['mape']:>8.2f}%"
        )
        total_rmse += m["rmse"]
        total_mae += m["mae"]
        total_mape += m["mape"]

    print("-" * 80)
    n = len(all_metrics)
    print(
        f"{'평균':<8} {'':<12} {'':<8} "
        f"${total_rmse/n:>12,.2f}  ${total_mae/n:>12,.2f}  {total_mape/n:>8.2f}%"
    )
    print("=" * 80)

print("\n" + "=" * 80)
print("시각화 완료!")
print(f"생성된 파일 ({len(saved_files)}개):")
for i, filename in enumerate(saved_files, 1):
    print(f"  {i}. {filename}")
print("=" * 80)
