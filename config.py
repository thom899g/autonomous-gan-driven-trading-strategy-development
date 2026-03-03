"""
Configuration and hyperparameter management for GAN-driven trading system.
Centralized configuration ensures reproducibility and easy experimentation.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime
import json

@dataclass
class GANConfig:
    """Configuration for GAN data generation"""
    # Architecture
    latent_dim: int = 100
    sequence_length: int = 60
    features_dim: int = 5  # OHLCV: Open, High, Low, Close, Volume
    generator_hidden_dim: int = 256
    discriminator_hidden_dim: int = 256
    num_lstm_layers: int = 2
    
    # Training
    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Validation
    validation_frequency: int = 50
    sample_frequency: int = 100
    
    # Data scaling
    use_minmax_scaling: bool = True
    scale_range: tuple = (-1, 1)

@dataclass
class RLConfig:
    """Configuration for reinforcement learning trading agent"""
    # Environment
    initial_balance: float = 10000.0
    max_position: float = 0.1  # Max 10% of portfolio per trade
    transaction_cost: float = 0.001  # 0.1% per transaction
    slippage: float = 0.0005  # 0.05% slippage
    
    # PPO Algorithm (chosen for stability)
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    
    # Training
    total_timesteps: int = 100000
    eval_frequency: int = 10000
    save_frequency: int = 50000
    
    # State features
    lookback_window: int = 30
    technical_indicators: List[str] = None
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = [
                "rsi", "macd", "bollinger_upper", "bollinger_lower", 
                "atr", "volume_sma"
            ]

@dataclass
class FirebaseConfig:
    """Firebase configuration for state management"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "trading-system")
    collection_name: str = "trading_states"
    model_bucket: str = "trading-models"
    
    # Real-time streaming
    use_realtime_db: bool = True
    real