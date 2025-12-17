import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import copy

# tqdm이 있으면 사용, 없으면 기본 반복문 사용
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print(
        "tqdm이 설치되어 있지 않습니다. 'pip install tqdm'으로 설치하면 진행 상황 표시가 더 자세해집니다."
    )

df = pd.read_csv("../data/BTC-USD.csv")
df["Date"] = pd.to_datetime(df["Date"])

train_data = df[df["Date"] <= "2024-11-12"]
test_data = df[df["Date"] > "2024-11-12"]

train_data = train_data.drop(columns=["Date"])
test_data = test_data.drop(columns=["Date"])

# differencing the data
train_data_pct = train_data.copy()
test_data_pct = test_data.copy()
for col in train_data.columns:
    train_data_pct[col] = train_data[col].pct_change()
    test_data_pct[col] = test_data[col].pct_change()

train_data_pct = train_data_pct.dropna()
test_data_pct = test_data_pct.dropna()


# 정규화 적용
scaler = MinMaxScaler()
train_data_pct = scaler.fit_transform(train_data_pct)
test_data_pct = scaler.transform(test_data_pct)


# 시퀀스 생성 (many to many) - 학습/예측 윈도우 분리
def create_sequences(data, input_seq_length, output_seq_length):
    X, y = [], []
    for i in range(input_seq_length, len(data) - output_seq_length + 1):
        X.append(data[i - input_seq_length : i])  # 과거 input_seq_length일
        y.append(
            data[i : i + output_seq_length, 0]
        )  # 미래 output_seq_length일의 Close만
    return np.array(X), np.array(y)


input_seq_length = 10  # 과거 10일 학습
output_seq_length = 2  # 미래 2일 예측

train_X_pct, train_y_pct = create_sequences(
    train_data_pct, input_seq_length, output_seq_length
)
test_X_pct, test_y_pct = create_sequences(
    np.concatenate([train_data_pct, test_data_pct]), input_seq_length, output_seq_length
)

# Train/Validation 분할 (80/20)
train_X_final, val_X, train_y_final, val_y = train_test_split(
    train_X_pct, train_y_pct, test_size=0.2, shuffle=False
)


# Early Stopping 클래스
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  ✓ 검증 손실 초기화: {val_loss:.6f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  ⚠ EarlyStopping 카운터: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"  🛑 Early Stopping 발동! 최적 검증 손실: {self.best_loss:.6f}"
                    )
        else:
            improvement = self.best_loss - val_loss
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"  ✓ 검증 손실 개선: {improvement:.6f} → 최적 모델 저장")


# 다변량 LSTM 모델 정의
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

        # Decoder - 첫 번째 입력은 인코더의 마지막 출력
        decoder_input = x[:, -1:, :]  # (batch, 1, input_size)
        decoder_outputs = []

        for _ in range(self.seq_length):
            decoder_output, (hidden, cell) = self.decoder_lstm(
                decoder_input, (hidden, cell)
            )
            output = self.fc(
                decoder_output
            )  # (batch, 1, input_size) - 모든 feature 예측
            decoder_outputs.append(output)
            decoder_input = output  # 다음 입력으로 사용 (차원 일치!)

        decoder_outputs = torch.cat(
            decoder_outputs, dim=1
        )  # (batch, seq_length, input_size)
        return decoder_outputs


# 하이퍼파라미터 설정
input_size = train_X_pct.shape[2]  # 특성 개수
hidden_size = 128
num_layers = 3
learning_rate = 0.0001
num_epochs = 200
batch_size = 32
patience = 30  # Early Stopping patience
max_grad_norm = 1.0  # Gradient clipping

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultivariateLSTM(input_size, hidden_size, num_layers, output_seq_length).to(
    device
)

# 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# Early Stopping 초기화
early_stopping = EarlyStopping(patience=patience, min_delta=1e-6, verbose=True)

# 데이터를 텐서로 변환
train_X_tensor = torch.FloatTensor(train_X_final).to(device)
train_y_tensor = torch.FloatTensor(train_y_final).to(device)
val_X_tensor = torch.FloatTensor(val_X).to(device)
val_y_tensor = torch.FloatTensor(val_y).to(device)

# DataLoader 생성
train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataset = torch.utils.data.TensorDataset(val_X_tensor, val_y_tensor)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

# 학습
print("=" * 60)
print(f"학습 시작 - Device: {device}")
print(f"Train X shape: {train_X_final.shape}, Train y shape: {train_y_final.shape}")
print(f"Val X shape: {val_X.shape}, Val y shape: {val_y.shape}")
print(f"총 훈련 배치 수: {len(train_loader)}, 검증 배치 수: {len(val_loader)}")
print(f"Early Stopping Patience: {patience}, Max Gradient Norm: {max_grad_norm}")
print("=" * 60)

train_losses = []
val_losses = []
start_time = time.time()

for epoch in range(num_epochs):
    # ========== 훈련 단계 ==========
    model.train()
    epoch_loss = 0
    epoch_start_time = time.time()

    # 배치 진행 상황 표시
    if HAS_TQDM:
        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{num_epochs}] Train",
            ncols=100,
            leave=False,
        )
        batch_iter = pbar
    else:
        batch_iter = train_loader

    for batch_idx, (batch_X, batch_y) in enumerate(batch_iter):
        # Forward pass - Close 값만 손실 계산
        outputs = model(batch_X)  # (batch, seq_length, input_size)
        loss = criterion(outputs[:, :, 0], batch_y)  # Close만 손실 계산

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        epoch_loss += loss.item()

        # 배치별 손실 표시
        if HAS_TQDM:
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        elif (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
            # tqdm이 없으면 10%마다 진행 상황 출력
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(
                f"  배치 진행: {batch_idx+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {loss.item():.6f}",
                end="\r",
            )

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ========== 검증 단계 ==========
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs[:, :, 0], batch_y)  # Close만 손실 계산
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Learning Rate Scheduler
    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # LR 변경 감지 및 출력
    lr_changed = prev_lr != current_lr
    lr_info = ""
    if lr_changed:
        lr_info = f" | ⚡ LR 감소: {prev_lr:.6f} → {current_lr:.6f}"

    # 에폭별 상세 정보 출력
    epoch_time = time.time() - epoch_start_time
    elapsed_time = time.time() - start_time
    avg_time_per_epoch = elapsed_time / (epoch + 1)
    remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)

    print(
        f"Epoch [{epoch+1:3d}/{num_epochs}] | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"LR: {current_lr:.6f}{lr_info} | "
        f"시간: {epoch_time:.1f}초"
    )

    # Early Stopping 체크
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print(f"\n🛑 Early Stopping 발동! (Epoch {epoch+1})")
        # 최적 모델 복원
        model.load_state_dict(early_stopping.best_model)
        break

# 모델 저장
total_time = time.time() - start_time
actual_epochs = len(train_losses)

# 최적 모델 저장 (Early Stopping이 best model을 이미 로드함)
torch.save(model.state_dict(), "../outputs/models/lstm_btc_model_best.pth")

print("\n" + "=" * 60)
print("학습 완료!")
print(f"실행된 에폭: {actual_epochs}/{num_epochs}")
print(f"총 소요 시간: {total_time/60:.2f}분 ({total_time:.2f}초)")
print(f"평균 에폭당 시간: {total_time/actual_epochs:.2f}초")
print(f"최종 훈련 손실: {train_losses[-1]:.6f}")
print(f"최종 검증 손실: {val_losses[-1]:.6f}")
print(f"최고 성능 (최저 검증 손실): {early_stopping.best_loss:.6f}")
print(f"최적 모델 저장 완료: outputs/models/lstm_btc_model_best.pth")
print("=" * 60)

# 학습 및 검증 손실 시각화
plt.figure(figsize=(12, 6))

# 서브플롯 1: 훈련 vs 검증 손실
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 서브플롯 2: 손실 차이 (과적합 모니터링)
plt.subplot(1, 2, 2)
loss_diff = np.array(val_losses) - np.array(train_losses)
plt.plot(loss_diff, color="red", linewidth=2)
plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
plt.title(
    "Validation - Train Loss (Overfitting Monitor)", fontsize=14, fontweight="bold"
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss Difference", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../outputs/figures/training_validation_loss.png", dpi=100, bbox_inches="tight")
print("학습/검증 손실 그래프 저장 완료: outputs/figures/training_validation_loss.png")
plt.close()

# 추가: Learning Rate 변화 시각화 (선택적)
print("\n학습 요약:")
print(f"  - 초기 Learning Rate: {learning_rate}")
print(f"  - 최종 Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
print(
    f"  - Early Stopping Triggered: {'Yes (Epoch ' + str(actual_epochs) + ')' if actual_epochs < num_epochs else 'No'}"
)
