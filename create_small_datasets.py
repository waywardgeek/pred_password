"""
Create smaller datasets for faster training experiments.

Samples from expanded_train.txt and expanded_eval.txt.
"""
import random

random.seed(42)

print("Creating 1M training set...")
with open('data/expanded_train.txt', 'r', encoding='latin-1') as f:
    # Read all passwords
    all_passwords = f.readlines()
    print(f"  Source: {len(all_passwords):,} passwords")
    
    # Sample 1M
    sample = random.sample(all_passwords, min(1_000_000, len(all_passwords)))
    
    # Write
    with open('data/train_1m.txt', 'w', encoding='latin-1') as out:
        out.writelines(sample)
    print(f"  ✓ Created data/train_1m.txt: {len(sample):,} passwords")

print("\nCreating 100K eval set...")
with open('data/expanded_eval.txt', 'r', encoding='latin-1') as f:
    # Read all passwords
    all_passwords = f.readlines()
    print(f"  Source: {len(all_passwords):,} passwords")
    
    # Sample 100K
    sample = random.sample(all_passwords, min(100_000, len(all_passwords)))
    
    # Write
    with open('data/eval_100k.txt', 'w', encoding='latin-1') as out:
        out.writelines(sample)
    print(f"  ✓ Created data/eval_100k.txt: {len(sample):,} passwords")

print("\n✓ Datasets ready for tiny model training!")
