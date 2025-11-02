"""
Full Dataset Training on GPU (RTX 4060).

Training on 14.7M passwords with full model.
"""
import sys
sys.path.insert(0, 'src')

from config import Config, ModelConfig, TrainingConfig
from train import train


def train_full():
    """Train on full expanded dataset with GPU."""
    print("=" * 70)
    print("PASSWORD TRANSFORMER - FULL TRAINING (GPU)")
    print("=" * 70)
    print("\nDataset:")
    print("  Training:   14.7M passwords (expanded by frequency)")
    print("  Evaluation: 1.6M passwords")
    print("\nModel:")
    print("  Architecture: 256-dim, 6 layers, 8 heads")
    print("  Parameters:   4.9M")
    print("\nHardware:")
    print("  GPU: RTX 4060 (8GB VRAM)")
    print("  Batch size: 32 (with mixed precision)")
    print("\nExpected time: ~2 hours for 10 epochs\n")
    
    # Full model, full dataset
    config = Config(
        model=ModelConfig(
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            max_seq_len=60,
            vocab_size=258  # 0-255 (bytes), 256 (END), 257 (START)
        ),
        training=TrainingConfig(
            train_file='data/expanded_train.txt',
            eval_file='data/expanded_eval.txt',
            batch_size=32,  # Good for 8GB VRAM
            epochs=10,
            learning_rate=3e-4,
            weight_decay=0.1,
            warmup_steps=1000,  # More warmup for large dataset
            gradient_clip=1.0,
            log_every=1000,  # Log every 1000 steps
            eval_every=1,
            save_every=1,
            checkpoint_dir='checkpoints/full_training',
            num_workers=4,  # 4 CPU threads for data loading
            device='cuda',  # GPU!
            pin_memory=True
        )
    )
    
    print("Starting training...\n")
    
    # Run training
    train(config)
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nBest model saved to: checkpoints/full_training/best_model.pt")


if __name__ == '__main__':
    train_full()
