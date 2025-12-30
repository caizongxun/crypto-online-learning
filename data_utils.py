"""
Data utilities for feature engineering and real-time data processing
Supports technical indicators calculation and data pipeline management
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


class TechnicalIndicators:
    """Calculate technical indicators for crypto price prediction"""
    
    @staticmethod
    def moving_average(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return np.convolve(prices, np.ones(period)/period, mode='valid')
    
    @staticmethod
    def exponential_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i-1] * (1 - multiplier)
        
        return ema
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        delta = np.diff(prices)
        seed = delta[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(period, len(prices)):
            delta = prices[i] - prices[i-1]
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta
            
            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period
            
            rs = up / down if down != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.exponential_moving_average(prices, fast)
        ema_slow = TechnicalIndicators.exponential_moving_average(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.moving_average(prices, period)
        
        # Pad to match original length
        sma_padded = np.concatenate([prices[:period-1], sma])
        
        std = np.zeros_like(prices)
        for i in range(period, len(prices)):
            std[i] = np.std(prices[i-period:i])
        
        upper_band = sma_padded + (std * std_dev)
        lower_band = sma_padded - (std * std_dev)
        
        return upper_band, sma_padded, lower_band
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = np.zeros_like(close)
        atr[period] = tr[1:period+1].mean()
        
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        k = np.zeros_like(close)
        
        for i in range(period, len(close)):
            low_min = np.min(low[i-period:i])
            high_max = np.max(high[i-period:i])
            
            if high_max - low_min != 0:
                k[i] = 100 * (close[i] - low_min) / (high_max - low_min)
        
        # Smooth K
        k_smoothed = np.zeros_like(k)
        for i in range(smooth_k, len(k)):
            k_smoothed[i] = np.mean(k[i-smooth_k:i])
        
        # Calculate D (signal line)
        d = np.zeros_like(k_smoothed)
        for i in range(smooth_d, len(k_smoothed)):
            d[i] = np.mean(k_smoothed[i-smooth_d:i])
        
        return k_smoothed, d


class DataPipeline:
    """Data pipeline for feature engineering and preprocessing"""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
        
        Returns:
            DataFrame with original + technical indicator features
        """
        df = df.copy()
        
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = TechnicalIndicators.exponential_moving_average(
                df['Close'].values, period
            )
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
            df[f'roc_{period}'] = df['Close'].pct_change(period)
        
        # RSI
        df['rsi_7'] = TechnicalIndicators.rsi(df['Close'].values, 7)
        df['rsi_14'] = TechnicalIndicators.rsi(df['Close'].values, 14)
        
        # MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(df['Close'].values)
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(df['Close'].values)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = upper - lower
        df['bb_position'] = (df['Close'] - lower) / (upper - lower)
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(
            df['High'].values, df['Low'].values, df['Close'].values
        )
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # Stochastic
        k, d = TechnicalIndicators.stochastic(
            df['High'].values, df['Low'].values, df['Close'].values
        )
        df['stochastic_k'] = k
        df['stochastic_d'] = d
        
        # Price position
        df['price_position_20'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                                  (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
        
        return df
    
    def create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,) or None
        
        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,) or None
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            if y is not None:
                y_seq.append(y[i+self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq) if y is not None else None
    
    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Min-Max normalization
        
        Args:
            X: Feature matrix
        
        Returns:
            X_normalized, mins, maxs
        """
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        X_normalized = (X - mins) / (maxs - mins + 1e-8)
        
        return X_normalized, mins, maxs


class StreamDataBuffer:
    """Circular buffer for streaming data"""
    
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.buffer = []
    
    def add(self, item):
        """Add item to buffer"""
        self.buffer.append(item)
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)
    
    def get_array(self) -> np.ndarray:
        """Get buffer as numpy array"""
        if len(self.buffer) == 0:
            return np.array([])
        
        if isinstance(self.buffer[0], np.ndarray):
            return np.array(self.buffer)
        else:
            return np.array(self.buffer).reshape(-1, 1)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]


class MetricsCalculator:
    """Calculate performance metrics"""
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAE"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE"""
        return np.sqrt(MetricsCalculator.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MAPE"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Percentage of correct direction predictions"""
        if len(y_true) < 2:
            return 0.0
        
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        
        return np.mean(direction_true == direction_pred)
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient"""
        if len(y_true) < 2:
            return 0.0
        
        return np.corrcoef(y_true, y_pred)[0, 1]
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate all metrics at once"""
        return {
            'mae': MetricsCalculator.mean_absolute_error(y_true, y_pred),
            'mse': MetricsCalculator.mean_squared_error(y_true, y_pred),
            'rmse': MetricsCalculator.root_mean_squared_error(y_true, y_pred),
            'mape': MetricsCalculator.mean_absolute_percentage_error(y_true, y_pred),
            'direction_accuracy': MetricsCalculator.direction_accuracy(y_true, y_pred),
            'correlation': MetricsCalculator.correlation(y_true, y_pred)
        }
