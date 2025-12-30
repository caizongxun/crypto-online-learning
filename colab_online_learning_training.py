"""
Online Learning Training Script for Google Colab
Download BTC data, train LSTM model with experience replay
Supports continuous updates as new data arrives
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from IPython.display import clear_output
import json


print("=" * 80)
print("CRYPTO ONLINE LEARNING - LSTM Model Training")
print(f"Timestamp: {datetime.now()}")
print("=" * 80)


# ============================================================================
# STEP 1: Setup Colab Environment
# ============================================================================

print("\n[STEP 1] Setting up Colab environment...")

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Mount Google Drive (optional, for saving results)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    drive_available = True
    print("Google Drive mounted successfully")
except:
    drive_available = False
    print("Google Drive not available (running outside Colab)")


# ============================================================================
# STEP 2: Clone/Download Repository
# ============================================================================

print("\n[STEP 2] Downloading code from GitHub...")

os.system('cd /content && rm -rf crypto-online-learning && git clone https://github.com/caizongxun/crypto-online-learning.git')
os.chdir('/content/crypto-online-learning')
sys.path.insert(0, '/content/crypto-online-learning')

from crypto_lstm_model import (
    CryptoLSTMModel, ReplayBuffer, FeatureScaler, 
    OnlineLearningTrainer, OnlineLearningPipeline
)

print("Code downloaded and imported successfully")


# ============================================================================
# STEP 3: Download BTC Data
# ============================================================================

print("\n[STEP 3] Downloading BTC data...")

# Install yfinance if not available
os.system('pip install -q yfinance')

import yfinance as yf

# Download BTC 1-hour data for last 6 months
btc = yf.download('BTC-USD', period='6mo', interval='1h', progress=False)
print(f"Downloaded {len(btc)} candles")
print(f"Date range: {btc.index[0]} to {btc.index[-1]}")
print(f"\nFirst few rows:\n{btc.head()}")


# ============================================================================
# STEP 4: Feature Engineering
# ============================================================================

print("\n[STEP 4] Feature engineering...")

def create_technical_indicators(df):
    """
    Create technical indicators from OHLCV data
    """
    df = df.copy()
    
    # Price features
    df['close'] = df['Close']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['open'] = df['Open']
    df['volume'] = df['Volume']
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
    
    # Momentum
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # RSI (Relative Strength Index)
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_14'] = rsi(df['Close'], 14)
    df['rsi_7'] = rsi(df['Close'], 7)
    
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_middle'] = sma_20
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Volatility
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    return df

# Create features
df = create_technical_indicators(btc)
df = df.dropna()

print(f"Total features created: {len(df.columns)}")
print(f"Data shape: {df.shape}")
print(f"\nFeature names:")
print(df.columns.tolist())


# ============================================================================
# STEP 5: Prepare Data for LSTM
# ============================================================================

print("\n[STEP 5] Preparing sequences for LSTM...")

# Select features (exclude OHLCV raw data)
feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
X = df[feature_cols].values
y = df['Close'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Create sequences
sequence_length = 60  # Look back 60 timesteps

def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, sequence_length)
print(f"Sequences shape: {X_seq.shape}")
print(f"Targets shape: {y_seq.shape}")

# Split into train/val/test
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


# ============================================================================
# STEP 6: Initialize Online Learning Pipeline
# ============================================================================

print("\n[STEP 6] Initializing online learning pipeline...")

input_size = X_train.shape[2]  # Number of features
print(f"Input size (n_features): {input_size}")
print(f"Sequence length: {sequence_length}")

pipeline = OnlineLearningPipeline(
    input_size=input_size,
    sequence_length=sequence_length,
    learning_rate=0.001,
    buffer_capacity=10000,
    device=device
)

# Initialize with training data statistics
pipeline.initialize(X_train.reshape(-1, input_size))
print("Pipeline initialized")


# ============================================================================
# STEP 7: Pre-training on Historical Data
# ============================================================================

print("\n[STEP 7] Pre-training on historical data...")

# Populate replay buffer with training data
for i in range(len(X_train)):
    # Create priority based on volatility (harder samples get higher priority)
    volatility = np.std(X_train[i])
    priority = 1.0 + volatility  # Higher volatility = higher priority
    
    # Add flattened sequence to buffer
    features = X_train[i].reshape(-1)
    target = y_train[i]
    
    pipeline.replay_buffer.add(features, float(target), priority=priority)

print(f"Replay buffer filled with {len(pipeline.replay_buffer)} samples")

# Train on batches
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

print("\nPre-training completed")


# ============================================================================
# STEP 8: Validation
# ============================================================================

print("\n[STEP 8] Validating model...")

pipeline.trainer.model.eval()
with torch.no_grad():
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_pred = pipeline.trainer.model(X_val_tensor).cpu().numpy()

y_val_pred = y_val_pred.flatten()

# Calculate metrics
mae = np.mean(np.abs(y_val - y_val_pred))
mse = np.mean((y_val - y_val_pred)**2)
rmse = np.sqrt(mse)
corr = np.corrcoef(y_val, y_val_pred)[0, 1]

print(f"Validation Metrics:")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  Corr: {corr:.4f}")

# Direction accuracy
direction_true = np.diff(y_val) > 0
direction_pred = np.diff(y_val_pred) > 0
dir_acc = np.mean(direction_true == direction_pred)
print(f"  Direction Accuracy: {dir_acc:.4f}")


# ============================================================================
# STEP 9: Simulate Online Learning
# ============================================================================

print("\n[STEP 9] Simulating online learning on test set...")

online_losses = []
online_predictions = []
online_y_true = []

pipeline.trainer.model.train()

for i in range(len(X_test)):
    # Get prediction before training
    pred = pipeline.trainer.predict(X_test[i])
    online_predictions.append(pred)
    online_y_true.append(y_test[i])
    
    # Add to replay buffer and train
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
    
    # Print progress
    if (i + 1) % 50 == 0:
        avg_loss = np.mean(online_losses[-50:]) if len(online_losses) > 0 else 0
        print(f"Step {i+1}/{len(X_test)} - Loss: {avg_loss:.6f}")

print("\nOnline learning simulation completed")


# ============================================================================
# STEP 10: Evaluate Online Learning Performance
# ============================================================================

print("\n[STEP 10] Evaluating online learning performance...")

online_predictions = np.array(online_predictions)
online_y_true = np.array(online_y_true)

online_mae = np.mean(np.abs(online_y_true - online_predictions))
online_rmse = np.sqrt(np.mean((online_y_true - online_predictions)**2))
online_corr = np.corrcoef(online_y_true, online_predictions)[0, 1]

online_direction_true = np.diff(online_y_true) > 0
online_direction_pred = np.diff(online_predictions) > 0
online_dir_acc = np.mean(online_direction_true == online_direction_pred)

print(f"Online Learning Metrics:")
print(f"  MAE:  {online_mae:.6f}")
print(f"  RMSE: {online_rmse:.6f}")
print(f"  Corr: {online_corr:.4f}")
print(f"  Direction Accuracy: {online_dir_acc:.4f}")
print(f"  Updates: {pipeline.update_count}")
print(f"  Buffer Size: {len(pipeline.replay_buffer)}")


# ============================================================================
# STEP 11: Visualization
# ============================================================================

print("\n[STEP 11] Creating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Plot 1: Training loss
axes[0].plot(training_losses, label='Pre-training Loss', linewidth=2)
axes[0].set_title('Pre-training Loss per Epoch', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Validation predictions
axes[1].plot(y_val, label='Actual', alpha=0.7, linewidth=1)
axes[1].plot(y_val_pred, label='Predicted', alpha=0.7, linewidth=1)
axes[1].set_title('Validation Set: Actual vs Predicted', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('BTC Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Online learning predictions
axes[2].plot(online_y_true, label='Actual', alpha=0.7, linewidth=1)
axes[2].plot(online_predictions, label='Predicted (Online Learning)', alpha=0.7, linewidth=1)
axes[2].set_title('Online Learning: Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Sample')
axes[2].set_ylabel('BTC Price')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
print("Plot saved as 'training_results.png'")
plt.show()


# ============================================================================
# STEP 12: Save Model
# ============================================================================

print("\n[STEP 12] Saving model...")

model_dir = 'checkpoints'
os.makedirs(model_dir, exist_ok=True)

pipeline.save(f'{model_dir}/online_learning_model.pt')
print(f"Model saved to {model_dir}/online_learning_model.pt")

# Save training results
results = {
    'pre_training': {
        'epochs': num_epochs,
        'final_loss': float(training_losses[-1]),
    },
    'validation': {
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'direction_accuracy': float(dir_acc)
    },
    'online_learning': {
        'mae': float(online_mae),
        'rmse': float(online_rmse),
        'correlation': float(online_corr),
        'direction_accuracy': float(online_dir_acc),
        'updates': int(pipeline.update_count),
        'buffer_size': len(pipeline.replay_buffer)
    },
    'data_info': {
        'total_samples': len(btc),
        'sequence_length': sequence_length,
        'n_features': input_size,
        'date_range': [str(btc.index[0]), str(btc.index[-1])]
    }
}

with open(f'{model_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {model_dir}/results.json")


# ============================================================================
# STEP 13: Save to Google Drive (if available)
# ============================================================================

if drive_available:
    print("\n[STEP 13] Saving to Google Drive...")
    
    import shutil
    
    drive_path = '/content/drive/MyDrive/crypto_online_learning'
    os.makedirs(drive_path, exist_ok=True)
    
    shutil.copy('checkpoints/online_learning_model.pt', f'{drive_path}/online_learning_model.pt')
    shutil.copy('checkpoints/results.json', f'{drive_path}/results.json')
    shutil.copy('training_results.png', f'{drive_path}/training_results.png')
    
    print(f"Files saved to: {drive_path}")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nNext steps:")
print("1. Review the visualizations and results")
print("2. Adjust hyperparameters if needed")
print("3. Deploy model to production")
print("4. Monitor model performance in real-time")
print("5. Retrain periodically with new data")
print(f"\nModel location: {os.path.abspath(model_dir)}/online_learning_model.pt")
print(f"Results location: {os.path.abspath(model_dir)}/results.json")
print("\n" + "="*80)
