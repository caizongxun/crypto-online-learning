"""
Crypto Online Learning Training Script for Google Colab
Automated end-to-end training pipeline
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch

try:
    from google.colab import drive
    IN_COLAB = True
except:
    IN_COLAB = False


def setup_colab_environment():
    """Setup Google Colab environment"""
    print("[STEP 1] Setting up Colab environment...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if IN_COLAB:
        try:
            drive.mount('/content/drive')
            print("Google Drive mounted")
        except:
            print("Google Drive not available (might be a permission issue)")
    else:
        print("Google Drive not available (running outside Colab)")
    
    return device


def download_code_from_github():
    """Download and import code from GitHub"""
    print("\n[STEP 2] Downloading code from GitHub...")
    
    try:
        from crypto_lstm_model import OnlineLearningPipeline
        from data_utils import DataPipeline, MetricsCalculator
        print("Code downloaded and imported successfully")
        return True
    except ImportError as e:
        print(f"Error importing: {e}")
        return False


def download_btc_data():
    """Download BTC historical data"""
    print("\n[STEP 3] Downloading BTC data...")
    
    btc = yf.download('BTC-USD', period='6mo', interval='1h', progress=False)
    
    # Normalize all column names to lowercase
    btc.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in btc.columns]
    
    print(f"\nDownloaded {len(btc)} candles")
    print(f"Date range: {btc.index[0]} to {btc.index[-1]}")
    print(f"\nFirst few rows:")
    print(btc.head())
    print(f"\nColumn names: {list(btc.columns)}")
    
    return btc


def create_features(btc_data):
    """Create technical indicators"""
    print("\n[STEP 4] Feature engineering...")
    
    from data_utils import DataPipeline
    
    df = btc_data.copy()
    
    # CRITICAL: Ensure all columns are lowercase before any processing
    print(f"Input columns: {list(df.columns)}")
    df.columns = [str(col).lower() for col in df.columns]
    print(f"Normalized columns: {list(df.columns)}")
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = list(df.columns)
    print(f"Required: {required_cols}")
    print(f"Available: {available_cols}")
    
    pipeline = DataPipeline(sequence_length=60)
    
    # Create features - this function will normalize columns internally
    df = pipeline.create_features(df)
    
    # Remove NaN rows
    df = df.dropna()
    
    print(f"Total features created: {len(df.columns)}")
    print(f"Data shape: {df.shape}")
    print(f"\nFeature names:")
    print(df.columns.tolist())
    
    return df, pipeline


def prepare_sequences(df, pipeline):
    """Prepare sequences for LSTM"""
    print("\n[STEP 5] Preparing sequences for LSTM...")
    
    # Select numeric features only (exclude OHLCV)
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['close'].values.astype(np.float32)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Selected {len(feature_cols)} features (excluded OHLCV)")
    
    # Create sequences
    X_seq, y_seq = pipeline.create_sequences(X, y)
    print(f"Sequences shape: {X_seq.shape}")
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def initialize_pipeline(X_train, y_train, device):
    """Initialize online learning pipeline"""
    print("\n[STEP 6] Initializing online learning pipeline...")
    
    from crypto_lstm_model import OnlineLearningPipeline
    
    input_size = X_train.shape[2]  # Number of features
    
    print(f"Input size (n_features): {input_size}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Device: {device}")
    
    pipeline = OnlineLearningPipeline(
        input_size=input_size,
        sequence_length=60,
        learning_rate=0.001,
        buffer_capacity=10000,
        device=device
    )
    
    # Initialize with flattened training data AND targets
    X_flat = X_train.reshape(-1, input_size)
    pipeline.initialize(X_flat, y_train)  # CRITICAL: Pass targets!
    print("Pipeline initialized")
    
    return pipeline


def pretrain_model(pipeline, X_train, y_train):
    """Pre-train on historical data"""
    print("\n[STEP 7] Pre-training on historical data...")
    
    batch_size = 32
    num_epochs = 5
    training_losses = []
    
    # Add training data to replay buffer
    print("Replay buffer filled with {} samples".format(len(X_train)))
    
    # First, populate the replay buffer
    for i in range(len(X_train)):
        sequence = X_train[i]  # (seq_len, n_features)
        target = y_train[i]
        
        # Calculate priority
        volatility = np.std(sequence)
        priority = 1.0 + volatility
        
        # Add to replay buffer (unscaled target - pipeline will scale it)
        pipeline.replay_buffer.add(sequence, float(target), priority=priority)
    
    print(f"Buffer size: {len(pipeline.replay_buffer)}")
    
    # Training epochs
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Sample from replay buffer and train
        for batch_idx in range(0, max(1, len(pipeline.replay_buffer) // batch_size)):
            batch = pipeline.replay_buffer.sample(batch_size, use_priority=True)
            loss = pipeline.trainer.train_step(batch)
            
            if loss is not None:
                epoch_losses.append(loss)
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - No training samples")
    
    print("\nPre-training completed!")
    return training_losses


def validate_model(pipeline, X_val, y_val, device):
    """Validate on validation set"""
    print("\n[STEP 8] Validating model...")
    
    pipeline.trainer.model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_val)):
            sequence = X_val[i]
            pred = pipeline.predict_next(sequence)
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    from data_utils import MetricsCalculator
    metrics = MetricsCalculator.calculate_all_metrics(y_val, predictions)
    
    print("\nValidation Metrics:")
    print(f"  Actual range: [{y_val.min():.2f}, {y_val.max():.2f}]")
    print(f"  Predicted range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    for metric_name, value in metrics.items():
        if metric_name == 'mape':
            print(f"  {metric_name.upper()}: {value:.2f}%")
        else:
            print(f"  {metric_name.upper()}: {value:.4f}")
    
    return predictions, metrics


def online_learning_simulation(pipeline, X_test, y_test):
    """Simulate online learning on test set"""
    print("\n[STEP 9] Simulating online learning on test set...")
    
    online_losses = []
    online_predictions = []
    online_y_true = []
    
    pipeline.trainer.model.train()
    
    for i in range(len(X_test)):
        sequence = X_test[i]
        target = y_test[i]
        
        # Predict
        pred = pipeline.predict_next(sequence)
        online_predictions.append(pred)
        online_y_true.append(float(target))
        
        # Train on this new sample
        # Add to replay buffer and train
        volatility = np.std(sequence)
        priority = 1.0 + volatility
        pipeline.replay_buffer.add(sequence, float(target), priority=priority)
        
        # Train
        if len(pipeline.replay_buffer) >= 16:
            batch = pipeline.replay_buffer.sample(16, use_priority=True)
            loss = pipeline.trainer.train_step(batch)
            if loss is not None:
                online_losses.append(loss)
        
        if (i + 1) % max(1, len(X_test) // 10) == 0:
            avg_loss = np.mean(online_losses[-50:]) if len(online_losses) > 0 else 0
            print(f"Step {i+1}/{len(X_test)} - Loss: {avg_loss:.6f}")
    
    print(f"\nOnline learning completed!")
    print(f"Total updates: {len(online_losses)}")
    print(f"Buffer size: {len(pipeline.replay_buffer)}")
    
    return np.array(online_predictions), np.array(online_y_true), online_losses


def evaluate_online_learning(y_true, predictions):
    """Evaluate online learning results"""
    print("\n[STEP 10] Evaluating online learning...")
    
    from data_utils import MetricsCalculator
    metrics = MetricsCalculator.calculate_all_metrics(y_true, predictions)
    
    print("\nOnline Learning Metrics:")
    print(f"  Actual range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"  Predicted range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    for metric_name, value in metrics.items():
        if metric_name == 'mape':
            print(f"  {metric_name.upper()}: {value:.2f}%")
        else:
            print(f"  {metric_name.upper()}: {value:.4f}")
    
    return metrics


def save_results(pipeline, fig_path='training_results.png'):
    """Save pipeline and results"""
    print(f"\n[STEP 11] Saving results...")
    
    os.makedirs('checkpoints', exist_ok=True)
    pipeline.save('checkpoints/model.pt')
    print(f"Model saved to checkpoints/model.pt")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("CRYPTO ONLINE LEARNING - LSTM Model Training")
    print(f"Timestamp: {datetime.now()}")
    print("="*80)
    
    # Setup
    device = setup_colab_environment()
    download_code_from_github()
    
    # Data preparation
    btc_data = download_btc_data()
    df, data_pipeline = create_features(btc_data)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_sequences(df, data_pipeline)
    
    # Model initialization and training
    pipeline = initialize_pipeline(X_train, y_train, device)
    training_losses = pretrain_model(pipeline, X_train, y_train)
    
    # Validation
    val_predictions, val_metrics = validate_model(pipeline, X_val, y_val, device)
    
    # Online learning simulation
    online_predictions, online_y_true, online_losses = online_learning_simulation(pipeline, X_test, y_test)
    
    # Evaluation
    online_metrics = evaluate_online_learning(online_y_true, online_predictions)
    
    # Save results
    save_results(pipeline)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    # Generate visualizations
    print("\n[STEP 12] Generating visualizations...")
    try:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Training loss
        if training_losses:
            axes[0].plot(training_losses, linewidth=2, color='blue')
            axes[0].set_title('Training Loss per Epoch', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss (MSE)')
            axes[0].grid(True, alpha=0.3)
        
        # Validation
        axes[1].plot(y_val, label='Actual', alpha=0.7, linewidth=1)
        axes[1].plot(val_predictions, label='Predicted', alpha=0.7, linewidth=1, linestyle='--')
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
        plt.savefig('training_results.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("Results saved to training_results.png")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print("\nAll done! Check 'checkpoints/model.pt' and 'training_results.png' for results.")


if __name__ == '__main__':
    main()
