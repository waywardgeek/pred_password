"""
100K Password Training Test - Validates full pipeline before GPU training.

Uses:
- 100K training passwords
- 10K eval passwords  
- Full model (256-dim, 6 layers)
- 5 epochs (~5-10 minutes on Mac CPU)

Success criteria:
- Loss decreases
- Accuracy improves  
- Can generate reasonable predictions
"""
import sys
sys.path.insert(0, 'src')

from config import Config, ModelConfig, TrainingConfig
from train import train


def test_100k():
    """Run 100K password training test."""
    print("=" * 70)
    print("100K PASSWORD TRAINING TEST")
    print("=" * 70)
    print("\nThis validates the full pipeline before moving to GPU:")
    print("  ✓ Full model architecture (4.9M params)")
    print("  ✓ Real data distribution (expanded by frequency)")
    print("  ✓ 100K training passwords")
    print("  ✓ 5 epochs")
    print("\nExpected time: 5-10 minutes on Mac CPU\n")
    
    # Full model, small dataset
    config = Config(
        model=ModelConfig(
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            max_seq_len=60,
            vocab_size=256
        ),
        training=TrainingConfig(
            train_file='data/train_100k.txt',
            eval_file='data/eval_10k.txt',
            batch_size=32,
            epochs=5,
            learning_rate=3e-4,
            log_every=50,
            eval_every=1,
            save_every=1,
            checkpoint_dir='checkpoints/test_100k',
            num_workers=0,  # CPU: single-threaded is faster
            device='cpu'  # Explicit CPU
        )
    )
    
    print("Configuration:")
    print(f"  Model: {config.model.d_model}-dim, {config.model.n_layers} layers, {config.model.n_heads} heads")
    print(f"  Training: {config.training.train_file}")
    print(f"  Eval: {config.training.eval_file}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Device: {config.training.device}")
    print()
    
    # Run training
    train(config)
    
    print("\n" + "=" * 70)
    print("✓ 100K TEST COMPLETE!")
    print("=" * 70)
    print("\nIf results look good, ready to transfer to GPU for full training!")
    print("  - Loss should decrease each epoch")
    print("  - Accuracy should improve (aim for >40% by epoch 5)")
    print("  - Checkpoints saved to checkpoints/test_100k/")


if __name__ == '__main__':
    test_100k()
