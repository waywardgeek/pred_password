"""
Expand rockyou-withcount.txt by frequency and split 90/10.

Reads rockyou-withcount.txt (format: "count password"), expands each
password by its frequency, randomizes the order, then splits into
90% training and 10% evaluation.

This ensures popular passwords appear in BOTH train and eval sets.
"""
import random


def expand_and_split(
    input_file='rockyou-withcount.txt',
    train_output='data/expanded_train.txt',
    eval_output='data/expanded_eval.txt',
    train_ratio=0.9,
    seed=42
):
    """
    Expand passwords by frequency, shuffle, and split.
    
    Args:
        input_file: RockYou file with "count password" format
        train_output: Output file for training set
        eval_output: Output file for evaluation set
        train_ratio: Fraction for training (default 0.9)
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("EXPANDING ROCKYOU DATASET BY FREQUENCY")
    print("=" * 70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {train_output} (train)")
    print(f"        {eval_output} (eval)")
    print(f"Split:  {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
    print(f"Seed:   {seed}\n")
    
    print("Reading and expanding passwords...")
    
    # Read and expand passwords
    expanded = []
    total_unique = 0
    total_accounts = 0
    
    with open(input_file, 'r', encoding='latin-1', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split(' ', 1)
            
            if len(parts) != 2:
                continue
            
            try:
                count = int(parts[0])
                password = parts[1]
                
                # Only passwords 8-60 characters
                if 8 <= len(password) <= 60:
                    # Repeat password 'count' times
                    for _ in range(count):
                        expanded.append(password + '\n')
                    
                    total_unique += 1
                    total_accounts += count
            
            except (ValueError, IndexError):
                continue
            
            # Progress every 1M lines
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} lines...")
                print(f"    {total_unique:,} unique passwords")
                print(f"    {total_accounts:,} total accounts")
    
    print(f"\n✓ Read complete!")
    print(f"  Unique passwords: {total_unique:,}")
    print(f"  Total accounts:   {total_accounts:,}")
    print(f"  Avg frequency:    {total_accounts/total_unique:.1f}")
    
    # Shuffle
    print(f"\nShuffling {total_accounts:,} passwords (seed={seed})...")
    random.seed(seed)
    random.shuffle(expanded)
    print("✓ Shuffled")
    
    # Split
    split_point = int(len(expanded) * train_ratio)
    train_set = expanded[:split_point]
    eval_set = expanded[split_point:]
    
    print(f"\n✓ Split:")
    print(f"  Train: {len(train_set):,} passwords ({len(train_set)/len(expanded)*100:.1f}%)")
    print(f"  Eval:  {len(eval_set):,} passwords ({len(eval_set)/len(expanded)*100:.1f}%)")
    
    # Write files
    print(f"\nWriting {train_output}...")
    with open(train_output, 'w', encoding='latin-1') as f:
        f.writelines(train_set)
    print(f"✓ Written: {len(train_set):,} passwords")
    
    print(f"\nWriting {eval_output}...")
    with open(eval_output, 'w', encoding='latin-1') as f:
        f.writelines(eval_set)
    print(f"✓ Written: {len(eval_set):,} passwords")
    
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Verify popular passwords appear in BOTH sets
    test_passwords = ['password', '123456789', 'iloveyou']
    
    for pwd in test_passwords:
        train_count = sum(1 for p in train_set if p.strip() == pwd)
        eval_count = sum(1 for p in eval_set if p.strip() == pwd)
        total_count = train_count + eval_count
        
        if total_count > 0:
            print(f"\n'{pwd}':")
            print(f"  Train: {train_count:,} ({train_count/total_count*100:.1f}%)")
            print(f"  Eval:  {eval_count:,} ({eval_count/total_count*100:.1f}%)")
            
            if train_count > 0 and eval_count > 0:
                print(f"  ✓ Present in BOTH sets (correct!)")
            else:
                print(f"  ⚠ Only in one set (unusual for common password)")
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE!")
    print("=" * 70)
    print(f"Training set:   {train_output}")
    print(f"Evaluation set: {eval_output}")


if __name__ == '__main__':
    import sys
    import os
    
    # Check if rockyou-withcount.txt exists
    if not os.path.exists('rockyou-withcount.txt'):
        print("Error: rockyou-withcount.txt not found!")
        print("Please ensure it's in the current directory.")
        sys.exit(1)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Run expansion
    expand_and_split(
        input_file='rockyou-withcount.txt',
        train_output='data/expanded_train.txt',
        eval_output='data/expanded_eval.txt',
        train_ratio=0.9,
        seed=42
    )
    
    print("\nReady for training!")
    print("  Train: ./venv/bin/python src/train.py")
