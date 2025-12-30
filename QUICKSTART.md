# Quick Start Guide - Run in Google Colab

在 Google Colab 中 5 分鐘開始訓練虛擬貨幣預測模型

## 最快方式：一行代碼啟動

### 步驟 1：打開 Google Colab

前往 [Google Colab](https://colab.research.google.com/)，建立新的 Notebook

### 步驟 2：複製貼上以下代碼

在 Colab 的代碼單元格中執行：

```python
# Clone repository and run training
!git clone https://github.com/caizongxun/crypto-online-learning.git
%cd crypto-online-learning
%run colab_online_learning_training.py
```

按 `Shift + Enter` 執行。

### 完成！

腳本會自動：
- 下載 6 個月 BTC 1 小時數據
- 創建 50+ 技術指標
- 訓練 LSTM 模型
- 模擬在線學習
- 保存結果和圖表

---

## 分步式執行（如果上方不行）

### Cell 1：安裝依賴

```python
!pip install -q torch torchvision torchaudio
!pip install -q yfinance pandas numpy scikit-learn matplotlib
!git clone https://github.com/caizongxun/crypto-online-learning.git
```

### Cell 2：導入模塊

```python
import sys
sys.path.insert(0, '/content/crypto-online-learning')

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from crypto_lstm_model import OnlineLearningPipeline
from data_utils import DataPipeline, MetricsCalculator

print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### Cell 3：下載數據

```python
# Download BTC data
print("Downloading BTC data...")
btc_data = yf.download('BTC-USD', period='6mo', interval='1h', progress=False)

print(f"\nData shape: {btc_data.shape}")
print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
print(f"\nFirst few rows:")
print(btc_data.head())
```

### Cell 4：特徵工程

```python
from data_utils import DataPipeline

# Create features
print("Creating technical indicators...")
data_pipeline = DataPipeline(sequence_length=60)
df = data_pipeline.create_features(btc_data.copy())

# Remove NaN rows
df = df.dropna()

print(f"\nTotal features: {len(df.columns)}")
print(f"Data shape after preprocessing: {df.shape}")
print(f"\nFeature list:")
for i, col in enumerate(df.columns[:10]):
    print(f"  {i+1}. {col}")
print(f"  ... and {len(df.columns) - 10} more")
```

### Cell 5：準備序列

```python
# Select features (exclude raw OHLCV)
feature_cols = [col for col in df.columns if col not in 
                ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

X = df[feature_cols].values
y = df['Close'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create sequences
X_seq, y_seq = data_pipeline.create_sequences(X, y)
print(f"\nSequences shape: {X_seq.shape}")
print(f"Targets shape: {y_seq.shape}")

# Split data
train_size = int(0.7 * len(X_seq))
val_size = int(0.15 * len(X_seq))
test_size = len(X_seq) - train_size - val_size

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]

X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"\nData split:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
```

### Cell 6：初始化管道

```python
from crypto_lstm_model import OnlineLearningPipeline

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = X_train.shape[2]

print(f"Input size: {input_size}")
print(f"Sequence length: 60")
print(f"Device: {device}")

# Initialize pipeline
pipeline = OnlineLearningPipeline(
    input_size=input_size,
    sequence_length=60,
    learning_rate=0.001,
    buffer_capacity=10000,
    device=device
)

# Initialize with training data
pipeline.initialize(X_train.reshape(-1, input_size))
print("\nPipeline initialized successfully")
```

### Cell 7：預訓練

```python
import random
from tqdm import tqdm

print("Pre-training on historical data...\n")

# Populate replay buffer
for i in range(len(X_train)):
    volatility = np.std(X_train[i])
    priority = 1.0 + volatility
    features = X_train[i].reshape(-1)
    target = y_train[i]
    pipeline.replay_buffer.add(features, float(target), priority=priority)

print(f"Replay buffer filled: {len(pipeline.replay_buffer)} samples\n")

# Training
batch_size = 32
num_epochs = 5
training_losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    
    for batch_idx in range(0, len(pipeline.replay_buffer), batch_size):
        batch = pipeline.replay_buffer.sample(batch_size, use_priority=True)
        loss = pipeline.trainer.train_step(batch)
        if loss is not None:
            epoch_losses.append(loss)
    
    avg_loss = np.mean(epoch_losses)
    training_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

print("\nPre-training completed!")
```

### Cell 8：驗證

```python
print("Validating model...\n")

pipeline.trainer.model.eval()
with torch.no_grad():
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_pred = pipeline.trainer.model(X_val_tensor).cpu().numpy()

y_val_pred = y_val_pred.flatten()

# Calculate metrics
from data_utils import MetricsCalculator
metrics = MetricsCalculator.calculate_all_metrics(y_val, y_val_pred)

print("Validation Metrics:")
for metric_name, value in metrics.items():
    if metric_name == 'mape':
        print(f"  {metric_name.upper()}: {value:.2f}%")
    else:
        print(f"  {metric_name.upper()}: {value:.4f}")
```

### Cell 9：在線學習模擬

```python
print("Simulating online learning on test set...\n")

online_losses = []
online_predictions = []
online_y_true = []

pipeline.trainer.model.train()

for i in range(len(X_test)):
    # Predict
    pred = pipeline.trainer.predict(X_test[i])
    online_predictions.append(pred)
    online_y_true.append(y_test[i])
    
    # Train
    features = X_test[i].reshape(-1)
    loss = pipeline.add_data(
        features=features,
        target=y_test[i],
        train=True,
        batch_size=16,
        use_priority=True
    )
    
    if loss is not None:
        online_losses.append(loss)
    
    if (i + 1) % 50 == 0:
        avg_loss = np.mean(online_losses[-50:]) if len(online_losses) > 0 else 0
        print(f"Step {i+1}/{len(X_test)} - Loss: {avg_loss:.6f}")

print(f"\nOnline learning completed!")
print(f"Total updates: {pipeline.update_count}")
print(f"Buffer size: {len(pipeline.replay_buffer)}")
```

### Cell 10：評估在線學習

```python
online_predictions = np.array(online_predictions)
online_y_true = np.array(online_y_true)

metrics_online = MetricsCalculator.calculate_all_metrics(online_y_true, online_predictions)

print("\nOnline Learning Metrics:")
for metric_name, value in metrics_online.items():
    if metric_name == 'mape':
        print(f"  {metric_name.upper()}: {value:.2f}%")
    else:
        print(f"  {metric_name.upper()}: {value:.4f}")
```

### Cell 11：可視化

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Training loss
axes[0].plot(training_losses, linewidth=2, color='blue')
axes[0].set_title('Training Loss per Epoch', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].grid(True, alpha=0.3)

# Validation
axes[1].plot(y_val, label='Actual', alpha=0.7, linewidth=1)
axes[1].plot(y_val_pred, label='Predicted', alpha=0.7, linewidth=1, linestyle='--')
axes[1].set_title('Validation: Actual vs Predicted', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('BTC Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Online learning
axes[2].plot(online_y_true, label='Actual', alpha=0.7, linewidth=1)
axes[2].plot(online_predictions, label='Predicted (Online)', alpha=0.7, linewidth=1, linestyle='--')
axes[2].set_title('Online Learning: Actual vs Predicted', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('BTC Price')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results.png', dpi=100, bbox_inches='tight')
plt.show()

print("Plot saved!")
```

### Cell 12：保存模型

```python
import os

os.makedirs('checkpoints', exist_ok=True)
pipeline.save('checkpoints/model.pt')

print("\nModel saved to 'checkpoints/model.pt'")
print("You can download it from Colab's Files panel")
```

---

## 下載結果

在 Colab 左側邊欄的「Files」標籤中：

1. 點擊「checkpoints」資料夾
2. 右鍵點擊「model.pt」
3. 選擇「Download」

或使用代碼：

```python
from google.colab import files
files.download('checkpoints/model.pt')
files.download('results.png')
```

---

## 常見問題

### Q: 為什麼訓練很慢？

A: 請檢查是否使用 GPU：

```python
import torch
print(torch.cuda.is_available())  # 應該返回 True
print(torch.cuda.get_device_name(0))
```

在 Colab 中啟用 GPU：菜單 → 代碼執行工具 → 變更執行時間類型 → GPU

### Q: 模型準度不高怎麼辦？

A: 嘗試：
- 增加訓練 epochs：`num_epochs = 10`
- 調整學習率：`learning_rate=0.01`
- 增加隱藏層大小：`hidden_size=256`

### Q: 如何使用自己的數據？

A: 替換下載數據的部分：

```python
# 使用自己的 CSV
df = pd.read_csv('my_data.csv', index_col='Date', parse_dates=True)
df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
```

### Q: 如何定期重訓練模型？

A: 使用 Colab 的定時執行功能或設置每週定時任務

---

## 下一步

1. **部署到生產環境**：使用 FastAPI + Kubernetes
2. **實時交易**：集成交易所 API
3. **模型監控**：使用 Prometheus + Grafana
4. **A/B 測試**：並行運行多個模型版本

更多詳細說明見 [README.md](README.md)

---

祝訓練愉快！
