"""
Configuration for Password Prediction Transformer.

This holds all hyperparameters in one place for easy experimentation.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    
    # Vocabulary
    vocab_size: int = 258  # 0-255 (bytes), 256 (END), 257 (START)
    
    # Sequence
    max_seq_len: int = 60  # Maximum password length
    
    # Architecture
    d_model: int = 256     # Embedding dimension
    n_heads: int = 8       # Number of attention heads
    n_layers: int = 6      # Number of transformer blocks
    d_ff: int = 1024       # Feed-forward dimension (4x d_model)
    dropout: float = 0.1   # Dropout probability
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_model > 0, "d_model must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Data
    train_file: str = 'data/expanded_train.txt'
    eval_file: str = 'data/expanded_eval.txt'
    
    # Training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    
    # Hardware
    device: str = 'cuda'  # Will fallback to CPU if CUDA unavailable
    mixed_precision: bool = True  # FP16 training
    num_workers: int = 4  # Data loading threads
    pin_memory: bool = True
    
    # Checkpointing
    save_every: int = 1  # Save every N epochs
    checkpoint_dir: str = 'checkpoints'
    
    # Logging
    log_every: int = 100  # Log every N steps
    eval_every: int = 1   # Evaluate every N epochs
    

@dataclass
class Config:
    """Complete configuration combining model and training."""
    model: ModelConfig
    training: TrainingConfig
    
    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            training=TrainingConfig()
        )
    
    @classmethod
    def smoke_test(cls):
        """Configuration for quick smoke test (tiny dataset)."""
        return cls(
            model=ModelConfig(
                d_model=128,  # Smaller for speed
                n_layers=2,
                n_heads=4
            ),
            training=TrainingConfig(
                batch_size=8,
                epochs=2,
                log_every=10
            )
        )
