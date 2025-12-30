import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class CryptoLSTMModel(nn.Module):
    """
    LSTM-based model for cryptocurrency price prediction
    """
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dense layers for final prediction
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Dense layers
        hidden = self.relu(self.fc1(last_output))
        output = self.fc2(hidden)
        
        return output


class ReplayBuffer:
    """
    Experience Replay Buffer for Online Learning
    Stores (state, target) pairs and samples them for training
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # For prioritized replay
    
    def add(self, state, target, priority=1.0):
        """
        Add experience to buffer
        
        Args:
            state: input features (seq_len, input_size)
            target: target value (scalar)
            priority: importance weight (higher = more likely to be sampled)
        """
        self.buffer.append((state, target))
        self.priorities.append(priority)
    
    def sample(self, batch_size=32, use_priority=True):
        """
        Sample batch from buffer
        
        Args:
            batch_size: number of samples
            use_priority: if True, use prioritized sampling
        
        Returns:
            List of (state, target) tuples
        """
        if use_priority and len(self.buffer) > 0:
            # Prioritized sampling
            priorities = np.array(list(self.priorities), dtype=np.float32)
            priorities = priorities / priorities.sum()
            
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                p=priorities,
                replace=False
            )
            batch = [self.buffer[i] for i in indices]
        else:
            # Random sampling
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return len(self.buffer) == self.capacity


class FeatureScaler:
    """
    Dynamic feature scaling for online learning
    Updates min/max values as new data arrives
    """
    def __init__(self, feature_names=None, scaling_type='minmax'):
        """
        Args:
            feature_names: list of feature names
            scaling_type: 'minmax' or 'standard'
        """
        self.feature_names = feature_names
        self.scaling_type = scaling_type
        
        self.mins = None
        self.maxs = None
        self.means = None
        self.stds = None
        self.is_fitted = False
    
    def fit(self, X):
        """
        Initialize scaler with initial data
        
        Args:
            X: (n_samples, n_features)
        """
        if self.scaling_type == 'minmax':
            self.mins = np.min(X, axis=0)
            self.maxs = np.max(X, axis=0)
            # Avoid division by zero
            self.maxs = np.where(self.maxs == self.mins, 1.0, self.maxs)
        elif self.scaling_type == 'standard':
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
            # Avoid division by zero
            self.stds = np.where(self.stds == 0, 1.0, self.stds)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Scale features
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            X_scaled: scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X = np.array(X, dtype=np.float32)
        
        if self.scaling_type == 'minmax':
            X_scaled = (X - self.mins) / (self.maxs - self.mins)
        elif self.scaling_type == 'standard':
            X_scaled = (X - self.means) / self.stds
        
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def update(self, X):
        """
        Update scaler with new data (for online learning)
        Dynamically adjust min/max or mean/std
        
        Args:
            X: new data batch (n_samples, n_features)
        """
        if not self.is_fitted:
            self.fit(X)
            return
        
        X = np.array(X, dtype=np.float32)
        
        if self.scaling_type == 'minmax':
            # Update mins and maxs
            self.mins = np.minimum(self.mins, np.min(X, axis=0))
            self.maxs = np.maximum(self.maxs, np.max(X, axis=0))
            # Avoid division by zero
            self.maxs = np.where(self.maxs == self.mins, 1.0, self.maxs)
        elif self.scaling_type == 'standard':
            # Running mean and std update
            # Simplified: recalculate from recent window
            pass


class OnlineLearningTrainer:
    """
    Online learning trainer for LSTM model
    Handles model updates as new data arrives
    """
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        """
        Args:
            model: CryptoLSTMModel instance
            learning_rate: learning rate for optimizer
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.training_history = {
            'losses': [],
            'timestamps': [],
            'batch_sizes': []
        }
    
    def train_step(self, batch):
        """
        Single training step
        
        Args:
            batch: list of (state, target) tuples
        
        Returns:
            loss: scalar loss value
        """
        if len(batch) == 0:
            return None
        
        # Convert to tensors
        states = torch.stack([torch.FloatTensor(state) for state, _ in batch]).to(self.device)
        targets = torch.FloatTensor([target for _, target in batch]).unsqueeze(1).to(self.device)
        
        # Forward pass
        predictions = self.model(states)
        loss = self.loss_fn(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, state):
        """
        Make prediction on single state
        
        Args:
            state: (seq_len, input_size)
        
        Returns:
            prediction: scalar value
        """
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            prediction = self.model(state_tensor).cpu().numpy()[0][0]
        self.model.train()
        
        return prediction
    
    def predict_batch(self, states):
        """
        Make predictions on batch of states
        
        Args:
            states: list of (seq_len, input_size) arrays
        
        Returns:
            predictions: array of predictions
        """
        self.model.eval()
        with torch.no_grad():
            states_tensor = torch.stack([torch.FloatTensor(state) for state in states]).to(self.device)
            predictions = self.model(states_tensor).cpu().numpy()
        self.model.train()
        
        return predictions
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.training_history
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_history = checkpoint['history']


class OnlineLearningPipeline:
    """
    Complete pipeline for online learning
    Integrates model, scaler, replay buffer, and trainer
    """
    def __init__(self, input_size=50, sequence_length=60, learning_rate=0.001, 
                 buffer_capacity=10000, device='cpu'):
        """
        Args:
            input_size: number of features
            sequence_length: lookback window size
            learning_rate: learning rate
            buffer_capacity: replay buffer size
            device: 'cpu' or 'cuda'
        """
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.device = device
        
        # Initialize components
        self.model = CryptoLSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=1
        )
        
        self.trainer = OnlineLearningTrainer(self.model, learning_rate=learning_rate, device=device)
        self.scaler = FeatureScaler(scaling_type='minmax')
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        self.is_initialized = False
        self.update_count = 0
    
    def initialize(self, initial_data):
        """
        Initialize pipeline with historical data
        
        Args:
            initial_data: (n_samples, n_features) array
        """
        # Fit scaler
        self.scaler.fit(initial_data)
        
        self.is_initialized = True
        print(f"Pipeline initialized with {len(initial_data)} samples")
    
    def add_data(self, features, target, train=True, batch_size=32, use_priority=True):
        """
        Add new data and optionally train
        
        Args:
            features: (n_features,) array
            target: scalar value
            train: whether to train on this step
            batch_size: batch size for training
            use_priority: use prioritized replay
        
        Returns:
            loss: training loss (if train=True)
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        # Scale features
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        self.scaler.update(features)
        features_scaled = self.scaler.transform(features)[0]
        
        # Add to replay buffer
        # Note: In real use, you'd maintain a sliding window of features
        self.replay_buffer.add(features_scaled, float(target))
        
        # Train if requested
        loss = None
        if train and len(self.replay_buffer) >= batch_size:
            batch = self.replay_buffer.sample(batch_size, use_priority=use_priority)
            loss = self.trainer.train_step(batch)
            self.update_count += 1
        
        return loss
    
    def predict_next(self, features):
        """
        Predict next value given features
        
        Args:
            features: (n_features,) array
        
        Returns:
            prediction: scalar
        """
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        features_scaled = self.scaler.transform(features)[0]
        
        # For single prediction, we need to construct a sequence
        # This is a simplified version - in practice, maintain a state buffer
        state = np.expand_dims(features_scaled, axis=0)  # (1, input_size) -> (1, 1, input_size)
        
        prediction = self.trainer.predict(state)
        return prediction
    
    def save(self, filepath):
        """Save entire pipeline"""
        import pickle
        
        checkpoint = {
            'model_checkpoint': {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.trainer.optimizer.state_dict(),
            },
            'scaler_params': {
                'mins': self.scaler.mins,
                'maxs': self.scaler.maxs,
                'means': self.scaler.means,
                'stds': self.scaler.stds,
                'type': self.scaler.scaling_type
            },
            'config': {
                'input_size': self.input_size,
                'sequence_length': self.sequence_length,
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Pipeline saved to {filepath}")
    
    def load(self, filepath):
        """Load entire pipeline"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_checkpoint']['model_state'])
        self.trainer.optimizer.load_state_dict(checkpoint['model_checkpoint']['optimizer_state'])
        
        self.scaler.mins = checkpoint['scaler_params']['mins']
        self.scaler.maxs = checkpoint['scaler_params']['maxs']
        self.scaler.means = checkpoint['scaler_params']['means']
        self.scaler.stds = checkpoint['scaler_params']['stds']
        self.scaler.scaling_type = checkpoint['scaler_params']['type']
        self.scaler.is_fitted = True
        
        self.is_initialized = True
        print(f"Pipeline loaded from {filepath}")
