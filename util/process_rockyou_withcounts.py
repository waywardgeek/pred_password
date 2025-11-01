#!/usr/bin/env python3
"""
Process RockYou dataset WITH FREQUENCY COUNTS (OPTIMIZED):
1. Filter passwords >= 8 characters
2. Keep frequency counts
3. Split into train (90%) and eval (10%)
"""
import random

print("Reading rockyou-withcount.txt and filtering...")
passwords_with_counts = []
total_accounts = 0
line_count = 0

with open('rockyou-withcount.txt', 'r', encoding='latin-1', errors='ignore') as f:
    for line in f:
        line_count += 1
        if line_count % 1000000 == 0:
            print(f"  Processed {line_count:,} lines...")
            
        line = line.strip()
        if not line:
            continue
            
        # Parse "count password" format - optimize by finding first space
        space_idx = line.find(' ')
        if space_idx == -1:
            continue
            
        try:
            count = int(line[:space_idx])
            password = line[space_idx+1:]
            
            # Filter for >= 8 characters
            if len(password) >= 8:
                passwords_with_counts.append(f"{count} {password}\n")
                total_accounts += count
        except ValueError:
            continue

print(f"\n✓ Filtered to {len(passwords_with_counts):,} unique passwords (>= 8 chars)")
print(f"✓ Total accounts represented: {total_accounts:,}")

# Shuffle randomly
print("\nShuffling...")
random.seed(42)
random.shuffle(passwords_with_counts)

# Split 90/10
print("Splitting...")
split_point = int(len(passwords_with_counts) * 0.9)
train_set = passwords_with_counts[:split_point]
eval_set = passwords_with_counts[split_point:]

print(f"\nTraining set: {len(train_set):,} unique passwords ({len(train_set)/len(passwords_with_counts)*100:.1f}%)")
print(f"Evaluation set: {len(eval_set):,} unique passwords ({len(eval_set)/len(passwords_with_counts)*100:.1f}%)")

# Write training set
print("\nWriting min8_train.data...")
with open('min8_train.data', 'w', encoding='latin-1') as f:
    f.writelines(train_set)

# Write evaluation set
print("Writing min8_eval.data...")
with open('min8_eval.data', 'w', encoding='latin-1') as f:
    f.writelines(eval_set)

print("\n✓ Done!")
print(f"  min8_train.data: {len(train_set):,} unique passwords")
print(f"  min8_eval.data:  {len(eval_set):,} unique passwords")

# Show top 10 most common passwords in training set
print("\n=== Top 10 most common passwords in training set (>= 8 chars) ===")
train_sorted = sorted(train_set, key=lambda x: int(x.split(' ', 1)[0]), reverse=True)
for i, line in enumerate(train_sorted[:10], 1):
    count, pwd = line.strip().split(' ', 1)
    print(f"{i:2d}. {int(count):,} - {pwd}")
