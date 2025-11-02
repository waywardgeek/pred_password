"""
Tiny Model Training - Fast experimentation.

Small model (~500K params) on 1M passwords for quick iterations.
"""
import sys
sys.path.insert(0, 'src')

from config import Config, ModelConfig, TrainingConfig
from train import train


def train_tiny():
    """Train tiny model on 1M passwords."""
    print("=" * 70)
    print("PASSWORD TRANSFORMER - TINY MODEL")
    print("=" * 70)
    print("\nDataset:")
    print("  Training:   1M passwords")
    print("  Evaluation: 100K passwords")
    print("\nModel:")
    print("  Architecture: 128-dim, 3 layers, 4 heads")
    print("  Parameters:   ~500K (10x smaller than full model)")
    print("\nHardware:")
    print("  GPU: RTX 4060")
    print("  Batch size: 64")
    print("\nExpected time: ~10 minutes per epoch\n")
    
    # Tiny model configuration
    config = Config(
        model=ModelConfig(
            d_model=128,      # Smaller embedding
            n_layers=3,       # Fewer layers
            n_heads=4,        # Fewer attention heads
            d_ff=512,         # Smaller feedforward
            dropout=0.1,
            max_seq_len=60,
            vocab_size=258    # 0-255 bytes + END (256) + START (257)
        ),
        training=TrainingConfig(
            train_file='data/train_1m.txt',
            eval_file='data/eval_100k.txt',
            batch_size=64,    # Larger batch since model is smaller
            epochs=5,         # Quick training
            learning_rate=3e-4,
            weight_decay=0.1,
            warmup_steps=500,
            gradient_clip=1.0,
            log_every=500,
            eval_every=1,
            save_every=1,
            checkpoint_dir='checkpoints/tiny_model',
            num_workers=4,
            device='cuda',
            pin_memory=True
        )
    )
    
    print("Starting training...\n")
    train(config)
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nModel saved to: checkpoints/tiny_model/best_model.pt")
    print("\nNext: Test password entropy with:")
    print("  ./venv/bin/python password_entropy.py <password>")


if __name__ == '__main__':
    train_tiny()
