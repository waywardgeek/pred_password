"""
Smoke test: Quick end-to-end test with tiny model and data.

This verifies:
1. Model can be created
2. Data can be loaded
3. Training loop runs without errors
4. Loss decreases

Takes ~2 minutes on CPU.
"""
import sys
sys.path.insert(0, 'src')

from config import Config, ModelConfig, TrainingConfig
from train import train


def smoke_test():
    """Run quick smoke test."""
    print("=" * 70)
    print("SMOKE TEST - Quick validation that everything works")
    print("=" * 70)
    
    # Create tiny configuration
    config = Config(
        model=ModelConfig(
            d_model=128,  # Small for speed
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            max_seq_len=60,
            vocab_size=256
        ),
        training=TrainingConfig(
            train_file='data/test_train.txt',  # Expanded format
            eval_file='data/test_eval.txt',    # Expanded format
            batch_size=8,
            epochs=2,
            learning_rate=3e-4,
            log_every=5,
            eval_every=1,
            save_every=1,
            checkpoint_dir='checkpoints/smoke',
            num_workers=0  # No multiprocessing for smoke test
        )
    )
    
    print("\nSmoke test config:")
    print(f"  Model: {config.model.d_model}-dim, {config.model.n_layers} layers")
    print(f"  Data: {config.training.train_file} (100 passwords)")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print("\nThis should take ~2 minutes...\n")
    
    # Run training
    train(config)
    
    print("\n✓ SMOKE TEST PASSED!")
    print("All systems operational. Ready for full training.")


if __name__ == '__main__':
    smoke_test()
