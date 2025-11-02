"""
Dataset for password sequences (expanded format - no frequency counts).

This version expects passwords already expanded by frequency and
split into train/eval. Each line is just a password (no counts).
"""
import torch
from torch.utils.data import Dataset
from typing import Dict


class PasswordDataset(Dataset):
    """
    Loads passwords from file (one password per line, already expanded).
    
    Example file:
        password
        password
        123456789
        iloveyou
        ...
    
    Returns batches ready for training:
        - input: password bytes
        - target: shifted left by 1 (next character prediction)
        - length: actual password length (for masking)
    """
    
    def __init__(self, data_file: str, max_len: int = 60):
        """
        Args:
            data_file: Path to password file (one per line, expanded)
            max_len: Maximum password length (longer ones truncated)
        """
        self.max_len = max_len
        self.passwords = []
        
        print(f"Loading passwords from {data_file}...")
        
        # Load passwords (already expanded, no counts)
        with open(data_file, 'r', encoding='latin-1', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                password = line.strip()
                
                # Filter: only passwords between 8 and max_len (leaving room for START + END tokens)
                if 8 <= len(password) <= max_len - 2:
                    # Convert to byte representation
                    # Format: [START=257, char1, char2, ..., charN, END=256]
                    byte_seq = [257] + [ord(c) for c in password] + [256]
                    self.passwords.append(byte_seq)
                
                # Progress every 1M lines
                if line_num % 1000000 == 0:
                    print(f"  Processed {line_num:,} lines...")
        
        print(f"✓ Loaded {len(self.passwords):,} passwords")
        print(f"  Avg length: {sum(len(p) for p in self.passwords) / len(self.passwords):.1f}")
    
    def __len__(self) -> int:
        return len(self.passwords)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one password sample.
        
        Returns:
            dict with:
                - input: (max_len,) password bytes, right-padded with 0
                - target: (max_len,) next char prediction, padded with 0
                - length: actual password length (scalar)
        """
        password = self.passwords[idx]
        length = len(password)
        
        # Create input/target pairs for next-character prediction
        # Input:  [p, a, s, s, w, o, r, d]
        # Target: [a, s, s, w, o, r, d, 0]  (0 = EOS token)
        
        input_seq = password[:]
        target_seq = password[1:] + [0]  # Shift left, append EOS
        
        # Pad to max_len with 0
        input_seq = input_seq + [0] * (self.max_len - len(input_seq))
        target_seq = target_seq + [0] * (self.max_len - len(target_seq))
        
        return {
            'input': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_seq, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }


def test_dataset():
    """Test dataset on a small file."""
    print("Testing PasswordDataset (expanded format)...")
    
    # Create a tiny test file (expanded, no counts)
    test_file = 'test_passwords_expanded.txt'
    with open(test_file, 'w') as f:
        f.write("password\n")
        f.write("password\n")  # Repeated (expanded by frequency)
        f.write("12345678\n")
        f.write("iloveyou\n")
    
    # Load dataset
    dataset = PasswordDataset(test_file, max_len=20)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Target shape: {sample['target'].shape}")
    print(f"  Length: {sample['length'].item()}")
    
    # Decode to verify
    input_bytes = sample['input'][:sample['length']].tolist()
    target_bytes = sample['target'][:sample['length']].tolist()
    
    input_str = ''.join(chr(b) for b in input_bytes)
    target_str = ''.join(chr(b) if b != 0 else '<EOS>' for b in target_bytes)
    
    print(f"  Input text: '{input_str}'")
    print(f"  Target text: '{target_str}'")
    
    # Verify target is shifted
    assert input_str[1:] == target_str[:-5], "Target should be input shifted left"
    
    # Verify second sample is also "password" (same frequency=2)
    sample2 = dataset[1]
    input_bytes2 = sample2['input'][:sample2['length']].tolist()
    input_str2 = ''.join(chr(b) for b in input_bytes2)
    print(f"\nSecond sample: '{input_str2}' (same as first due to frequency)")
    
    print("\n✓ Dataset test passed!")
    
    # Clean up
    import os
    os.remove(test_file)


if __name__ == '__main__':
    test_dataset()
