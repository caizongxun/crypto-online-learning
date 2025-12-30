# Cryptocurrency Price Prediction with Online Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A complete framework for training LSTM models with **online learning** and **experience replay** for cryptocurrency price prediction. Supports real-time model updates as new market data arrives.

## Features

- **LSTM-based Architecture**: Specialized for time series forecasting with temporal dependencies
- **Online Learning**: Update model incrementally as new data arrives (no need for full retraining)
- **Experience Replay Buffer**: Prevent catastrophic forgetting by mixing old and new training data
- **Prioritized Sampling**: Prioritize difficult samples for faster convergence
- **Dynamic Feature Scaling**: Min-Max scaling adapts to new data ranges
- **Colab Ready**: Complete training script optimized for Google Colab
- **Checkpoint Saving**: Save and load model states for resuming training

## Why Online Learning for Crypto Trading?

### Problem with Traditional Batch Learning
- Requires retraining entire model with all historical data
- Can't adapt to market regime changes quickly
- Long training time means predictions lag behind market
- Data scaling becomes inconsistent as market ranges expand

### Solution: Online Learning
- Update model weights incrementally with each new price bar
- Adapt to market changes within 1-4 hours instead of 24 hours
- Continuous operation without retraining interruptions
- Lower computational cost per update

## Quick Start with Colab

### Option 1: Direct Execution (Recommended)

Open this notebook in Colab:

```python
# Run in a Colab cell:
!git clone https://github.com/caizongxun/crypto-online-learning.git
%cd crypto-online-learning
%run colab_online_learning_training.py
```

That's it! The script will:
1. Download 6 months of BTC 1-hour data
2. Create 50+ technical indicators
3. Train LSTM model on historical data
4. Simulate online learning on test set
5. Save results and visualizations

### Option 2: Step-by-Step Notebook

Create a new Colab notebook and execute:

```python
# Cell 1: Setup
import torch
import numpy as np
import pandas as pd
import yfinance as yf

# Clone repo
!git clone https://github.com/caizongxun/crypto-online-learning.git
%cd crypto-online-learning
import sys
sys.path.insert(0, '.')

from crypto_lstm_model import OnlineLearningPipeline

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
```

```python
# Cell 2: Download data
btc_data = yf.download('BTC-USD', period='6mo', interval='1h')
print(f"Downloaded {len(btc_data)} candles")
```

```python
# Cell 3: Initialize and train
pipeline = OnlineLearningPipeline(
    input_size=50,
    sequence_length=60,
    learning_rate=0.001,
    buffer_capacity=10000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ... training code ...
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         OnlineLearningPipeline (Main Class)             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐   │
│  │   LSTM Model │  │ Replay Buffer│  │   Scaler   │   │
│  └──────────────┘  └──────────────┘  └────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  OnlineLearningTrainer (Handle updates)          │  │
│  │  - train_step()                                  │  │
│  │  - predict()                                     │  │
│  │  - save_checkpoint()                             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. CryptoLSTMModel
Multi-layer LSTM with dense output layers:
- Input: (batch_size, sequence_length, n_features)
- LSTM layers: 2 layers with 128 hidden units
- Output: (batch_size, 1) - next period price

#### 2. ReplayBuffer
Circular buffer storing recent experiences:
- Capacity: 10,000 samples (configurable)
- Sampling methods: Random or Prioritized
- Prevents catastrophic forgetting

#### 3. FeatureScaler
Dynamic min-max normalization:
- Tracks min/max values across batches
- Updates when new data arrives
- Supports MinMax and Standard scaling

#### 4. OnlineLearningTrainer
Handles model training and inference:
- Mini-batch gradient descent
- Gradient clipping to prevent explosion
- Checkpoint save/load functionality

## Configuration

### Model Hyperparameters

```python
pipeline = OnlineLearningPipeline(
    input_size=50,              # Number of features
    sequence_length=60,         # Lookback window (timesteps)
    learning_rate=0.001,        # Adam optimizer LR
    buffer_capacity=10000,      # Replay buffer size
    device='cuda'               # 'cuda' or 'cpu'
)
```

### Training Parameters

```python
# Pre-training
num_epochs = 5              # Full passes through training data
batch_size = 32             # Samples per gradient update
use_priority = True         # Prioritized experience replay

# Online learning
update_frequency = 'hourly' # How often to train on new data
batch_size_online = 16      # Smaller for online updates
```

## Usage Examples

### Example 1: Basic Training

```python
from crypto_lstm_model import OnlineLearningPipeline
import numpy as np

# Initialize
pipeline = OnlineLearningPipeline(input_size=50, sequence_length=60)

# Initialize with historical data
historical_data = np.random.randn(1000, 50)  # (samples, features)
pipeline.initialize(historical_data)

# Simulate new data arrival
for t in range(100):
    new_features = np.random.randn(50)
    target_price = 50000 + np.random.randn()
    
    # Add data and train
    loss = pipeline.add_data(
        features=new_features,
        target=target_price,
        train=True,
        batch_size=32
    )
    
    print(f"Step {t}: Loss = {loss:.6f}")
```

### Example 2: Make Predictions

```python
# Single prediction
new_features = np.random.randn(50)
prediction = pipeline.predict_next(new_features)
print(f"Next price prediction: {prediction}")

# Batch predictions
features_batch = np.random.randn(10, 50)
predictions = pipeline.trainer.predict_batch(features_batch)
print(f"Predictions shape: {predictions.shape}")
```

### Example 3: Save and Load

```python
# Save model
pipeline.save('my_model.pt')

# Load model
pipeline_loaded = OnlineLearningPipeline()
pipeline_loaded.load('my_model.pt')
```

## Technical Details

### Experience Replay Mechanism

Prevents catastrophic forgetting when learning from streaming data:

```
New Data Arrives
    ↓
[Add to Replay Buffer]
    ↓
[Sample Mini-batch from Buffer]
    ├─ 80% new data samples
    └─ 20% old data samples
    ↓
[Gradient Update]
    ↓
[Repeat every new data point]
```

**Benefits:**
- Maintains diversity in training data
- Old market patterns not forgotten
- Smoother training dynamics
- Better generalization to unseen data

### Prioritized Sampling

Priority based on:
- **TD-Error**: Samples with high prediction error sampled more often
- **Difficulty**: Samples with high volatility sampled more often
- **Recency**: Recent samples get higher priority

Implementation:
```python
priorities = [priority_1, priority_2, ..., priority_N]
probabilities = priorities / sum(priorities)
batch_indices = np.random.choice(N, size=batch_size, p=probabilities)
```

### Feature Scaling Update

Dynamic scaling adapts to market ranges:

```python
# Initial fit
scaler.fit(historical_data)  # Compute initial min/max

# Online update as new data arrives
for new_sample in stream:
    scaler.update(new_sample)  # Update min/max values
    scaled = scaler.transform(new_sample)
```

## Performance Metrics

The training script reports:

| Metric | Description | Good Range |
|--------|-------------|------------|
| MAE | Mean Absolute Error | Lower is better |
| RMSE | Root Mean Square Error | Lower is better |
| Correlation | Predicted vs Actual correlation | > 0.6 |
| Direction Accuracy | % correct up/down predictions | > 55% |
| Updates | Number of training steps | Monitored |
| Buffer Utilization | How full the replay buffer is | 80-100% |

## Troubleshooting

### GPU Memory Error
```python
# Reduce batch size
pipeline.add_data(..., batch_size=8)  # Instead of 32
```

### Loss Not Decreasing
```python
# Increase learning rate slightly
pipeline.trainer.optimizer.param_groups[0]['lr'] = 0.01
```

### Predictions Always Same Value
```python
# Check scaler is initialized
print(pipeline.scaler.is_fitted)  # Should be True

# Verify data passed to model
print(pipeline.replay_buffer.buffer[0])  # Check sample
```

## Advanced: Custom Implementation

For production use, implement custom data feed:

```python
class LiveDataFeed:
    """Custom data feed from exchange API"""
    
    def __init__(self, exchange_api):
        self.api = exchange_api
    
    def get_latest(self):
        """Get latest OHLCV and indicators"""
        ohlcv = self.api.fetch_ohlcv('BTC/USDT', timeframe='1h')
        features = self.compute_features(ohlcv)
        return features
    
    def compute_features(self, ohlcv):
        """Create 50 technical indicators"""
        # ... feature engineering code ...
        return features

# Use in training loop
data_feed = LiveDataFeed(exchange_api)

while True:
    features = data_feed.get_latest()
    loss = pipeline.add_data(features, train=True)
    time.sleep(3600)  # Update every hour
```

## Deployment Checklist

- [ ] Train model on 6+ months of historical data
- [ ] Validate on held-out test set
- [ ] Monitor direction accuracy > 55%
- [ ] Set up automatic model checkpoints
- [ ] Implement model performance monitoring
- [ ] Create rollback procedure for bad models
- [ ] Test in paper trading first
- [ ] Implement position sizing based on confidence
- [ ] Add circuit breakers for anomalies
- [ ] Schedule periodic retraining (weekly)

## References

### Key Papers
- **Online Learning**: "Online Learning and Optimization" (Hazan, 2016)
- **Experience Replay**: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- **Continual Learning**: "Continual learning with hypernetworks" (von Rutte et al., 2019)
- **LSTM for Time Series**: "LSTM: A Search Space Odyssey" (Greff et al., 2015)

### Recommended Reading
- Catastrophic Forgetting and Solutions [1]
- Deep RL for Trading [2]
- Time Series Forecasting Best Practices [3]

## License

MIT License - see LICENSE file

## Citation

If you use this code, please cite:

```bibtex
@software{crypto_online_learning_2025,
  title={Cryptocurrency Price Prediction with Online Learning},
  author={caizongxun},
  year={2025},
  url={https://github.com/caizongxun/crypto-online-learning}
}
```

## Contact & Support

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/caizongxun/crypto-online-learning/issues)
- GitHub Discussions: [Join discussion](https://github.com/caizongxun/crypto-online-learning/discussions)

## Disclaimer

This code is for educational purposes. Cryptocurrency trading involves significant risk. Always:
- Backtest thoroughly before live trading
- Start with small position sizes
- Use stop losses
- Monitor model performance continuously
- Update models based on market changes

Past performance does not guarantee future results.

---

**Last Updated**: December 30, 2025
