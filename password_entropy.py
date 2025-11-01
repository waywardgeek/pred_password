"""
Password Entropy Calculator using trained transformer model.

Computes entropy (in bits) of a password based on character-by-character
probabilities from the model: -sum(log2(p_i))

Lower entropy = more predictable = weaker password
Higher entropy = less predictable = stronger password
"""
import sys
sys.path.insert(0, 'src')

import torch
import math
from pathlib import Path
from model import PasswordTransformer
from config import ModelConfig


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    config = checkpoint.get('config')
    if config is None:
        # Detect model type by checking parameter count
        param_count = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        
        if param_count > 1_000_000:  # Full model
            print("  Detected: Full model (4.9M params)")
            config = ModelConfig(
                d_model=256,
                n_layers=6,
                n_heads=8,
                d_ff=1024,
                dropout=0.1,
                max_seq_len=60,
                vocab_size=258  # 0-255 bytes, 256 END, 257 START
            )
        else:  # Tiny model
            print("  Detected: Tiny model (668K params)")
            config = ModelConfig(
                d_model=128,
                n_layers=3,
                n_heads=4,
                d_ff=512,
                dropout=0.1,
                max_seq_len=60,
                vocab_size=258  # 0-255 for bytes, 256 for END, 257 for START
            )
    
    # Create model
    model = PasswordTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


def compute_entropy(model, password, device='cuda'):
    """
    Compute password entropy: -sum(log2(p_i)) for each character.
    
    Returns:
        total_entropy: Total bits of entropy
        per_char_entropy: List of entropy contribution per character
        per_char_probs: List of probabilities for each character
    """
    # Convert password to bytes
    password_bytes = password.encode('utf-8', errors='ignore')
    if len(password_bytes) > 60:
        print(f"⚠ Password too long ({len(password_bytes)} bytes), truncating to 60")
        password_bytes = password_bytes[:60]
    
    # Prepend START token (257) to the input
    # Model format: [START, char1, char2, ..., charN]
    input_ids = torch.tensor([[257] + list(password_bytes)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Get model predictions
        logits = model(input_ids)  # [1, seq_len+1, vocab_size]
        
        # Get probabilities (softmax over vocab)
        probs = torch.softmax(logits, dim=-1)  # [1, seq_len+1, vocab_size]
        
        # Compute entropy for each character
        per_char_entropy = []
        per_char_probs = []
        
        for i in range(len(password_bytes)):
            # Position i in password corresponds to position i in model output
            # (position 0 predicts first char after START token)
            target_byte = password_bytes[i]
            predicted_prob = probs[0, i, target_byte].item()
            per_char_probs.append(predicted_prob)
            
            # Entropy contribution: -log2(p)
            if predicted_prob > 0:
                entropy_bits = -math.log2(predicted_prob)
            else:
                entropy_bits = float('inf')  # Zero probability
            
            per_char_entropy.append(entropy_bits)
        
        # Add entropy for TERMINATION (end of password)
        # After the last character, what's the probability the password ends?
        # We use END token (byte 256) since that's how passwords
        # are stored in the training data (each password ends with 256)
        termination_byte = 256  # END token
        if len(password_bytes) > 0:
            # Get probability of END after the last character
            # Position len(password_bytes) predicts what comes after last char
            termination_prob = probs[0, len(password_bytes), termination_byte].item()
            per_char_probs.append(termination_prob)
            
            if termination_prob > 0:
                termination_entropy = -math.log2(termination_prob)
            else:
                termination_entropy = float('inf')
            
            per_char_entropy.append(termination_entropy)
        
        total_entropy = sum(per_char_entropy)
    
    return total_entropy, per_char_entropy, per_char_probs


def analyze_password(model, password, device='cuda'):
    """Analyze password and print detailed entropy breakdown."""
    print("\n" + "=" * 70)
    print(f"PASSWORD ENTROPY ANALYSIS: '{password}'")
    print("=" * 70)
    
    total_entropy, per_char_entropy, per_char_probs = compute_entropy(model, password, device)
    
    print(f"\nTotal Entropy: {total_entropy:.2f} bits")
    print(f"Average per character: {total_entropy/len(password):.2f} bits/char")
    
    # Strength assessment
    print("\nStrength Assessment:")
    if total_entropy < 20:
        print("  🔴 VERY WEAK - Highly predictable")
    elif total_entropy < 30:
        print("  🟠 WEAK - Somewhat predictable")
    elif total_entropy < 40:
        print("  🟡 MODERATE - Average strength")
    elif total_entropy < 50:
        print("  🟢 STRONG - Good entropy")
    else:
        print("  🟢 VERY STRONG - High entropy")
    
    print("\nPer-Character Breakdown:")
    print(f"{'Pos':<5} {'Char':<8} {'Probability':<15} {'Entropy (bits)':<15}")
    print("-" * 52)
    
    password_bytes = password.encode('utf-8', errors='ignore')
    for i, (byte_val, prob, entropy) in enumerate(zip(password_bytes, per_char_probs, per_char_entropy)):
        char = chr(byte_val) if 32 <= byte_val < 127 else f'\\x{byte_val:02x}'
        print(f"{i:<5} {char:<8} {prob:<15.6f} {entropy:<15.2f}")
    
    # Show termination probability (last element in lists)
    if len(per_char_probs) > len(password_bytes):
        term_prob = per_char_probs[-1]
        term_entropy = per_char_entropy[-1]
        print(f"{len(password_bytes):<5} {'<END>':<8} {term_prob:<15.6f} {term_entropy:<15.2f}")
    
    print("\n" + "=" * 70)


def main():
    """CLI for password entropy analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute password entropy using trained transformer')
    parser.add_argument('password', nargs='?', help='Password to analyze')
    parser.add_argument('--model', default='checkpoints/tiny_model/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--full', action='store_true',
                       help='Use full model (4.9M params) instead of tiny (668K params)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--batch', action='store_true',
                       help='Read passwords from stdin (one per line)')
    
    args = parser.parse_args()
    
    # Handle --full flag
    if args.full:
        args.model = 'checkpoints/full_training/best_model.pt'
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("\nTrain a model first:")
        print("  ./venv/bin/python train_tiny.py")
        return 1
    
    model = load_model(args.model, device)
    
    if args.batch:
        # Batch mode: read from stdin
        print("\n" + "=" * 70)
        print("BATCH MODE - Enter passwords (one per line, Ctrl+D to finish):")
        print("=" * 70)
        
        for line in sys.stdin:
            password = line.strip()
            if password:
                total_entropy, _, _ = compute_entropy(model, password, device)
                print(f"{password:<30} {total_entropy:>8.2f} bits")
    
    elif args.password:
        # Single password mode
        analyze_password(model, args.password, device)
    
    else:
        # Interactive mode
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE - Enter passwords to analyze (Ctrl+C to quit)")
        print("=" * 70)
        
        try:
            while True:
                password = input("\nPassword: ").strip()
                if password:
                    analyze_password(model, password, device)
        except (KeyboardInterrupt, EOFError):
            print("\n\n✓ Done!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
