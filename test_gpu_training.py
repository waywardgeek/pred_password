"""
Quick GPU test - verify PyTorch uses GPU correctly.
Uses 100K passwords, 2 epochs (~5 min).
"""
import sys
sys.path.insert(0, 'src')

from config import Config, ModelConfig, TrainingConfig
from train import train


def gpu_test():
    """Quick GPU verification test."""
    print("=" * 70)
    print("GPU TEST - Quick validation before full training")
    print("=" * 70)
    print("\nThis runs 2 epochs on 100K passwords to verify:")
    print("  ✓ GPU is being used")
    print("  ✓ Mixed precision works")
    print("  ✓ No memory errors")
    print("\nExpected time: ~5 minutes\n")
    
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
            epochs=2,
            learning_rate=3e-4,
            log_every=50,
            eval_every=1,
            save_every=1,
            checkpoint_dir='checkpoints/gpu_test',
            num_workers=4,
            device='cuda',  # Use GPU
            pin_memory=True
        )
    )
    
    print("Configuration:")
    print(f"  Device: {config.training.device}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Training: {config.training.train_file}")
    print()
    
    # Run training
    train(config)
    
    print("\n" + "=" * 70)
    print("✓ GPU TEST PASSED!")
    print("=" * 70)
    print("\nGPU is working. Ready for full training!")
    print("  Run: ./venv/bin/python train_full.py")


if __name__ == '__main__':
    gpu_test()
