"""
Arithmetic Coding for Password Validation.

Uses model probabilities to compress/decompress passwords.
Validates entropy calculations by comparing compressed size to calculated entropy.
"""
import sys
sys.path.insert(0, 'src')

import torch
import math
from model import PasswordTransformer
from config import ModelConfig
from pathlib import Path


def load_model(model_path, device='cpu'):
    """Load trained model."""
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    
    # Detect model size and create appropriate config
    param_count = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    
    if param_count > 1_000_000:  # Full model
        config = ModelConfig(
            d_model=256, n_layers=6, n_heads=8, d_ff=1024,
            dropout=0.1, max_seq_len=60, vocab_size=258
        )
    else:  # Tiny model
        config = ModelConfig(
            d_model=128, n_layers=3, n_heads=4, d_ff=512,
            dropout=0.1, max_seq_len=60, vocab_size=258
        )
    
    model = PasswordTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def get_symbol_probabilities(model, context, device='cpu'):
    """
    Get probability distribution for next symbol given context.
    
    Args:
        model: Trained transformer model
        context: List of byte values (including START=257)
        device: 'cpu' or 'cuda'
    
    Returns:
        torch.Tensor: Probability distribution over 258 symbols (0-257)
    """
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1)
        # Return probabilities for next symbol after last position
        return probs[0, -1]


def arithmetic_encode(password, model, device='cpu'):
    """
    Encode password using arithmetic coding with model probabilities.
    
    Returns:
        (encoded_value, num_bits): Encoded as integer and number of bits used
    """
    # Start with START token
    context = [257]  # START token
    
    # Password bytes + END token
    password_bytes = [ord(c) for c in password] + [256]
    
    # Arithmetic coding state
    low = 0.0
    high = 1.0
    
    # Track actual bits needed (for validation)
    total_bits = 0.0
    
    for symbol in password_bytes:
        # Get probability distribution given current context
        probs = get_symbol_probabilities(model, context, device)
        
        # Build cumulative probability distribution
        cumsum = torch.cumsum(probs, dim=0)
        cumsum = torch.cat([torch.tensor([0.0], device=device), cumsum])
        
        # Get range for this symbol
        symbol_low = cumsum[symbol].item()
        symbol_high = cumsum[symbol + 1].item()
        
        # Update interval
        range_size = high - low
        high = low + range_size * symbol_high
        low = low + range_size * symbol_low
        
        # Track bits (information content of this symbol)
        prob = (symbol_high - symbol_low)
        total_bits += -math.log2(prob) if prob > 0 else float('inf')
        
        # Add symbol to context for next prediction
        context.append(symbol)
    
    # Final encoded value is any number in [low, high)
    # We'll use the midpoint
    encoded_value = (low + high) / 2.0
    
    return encoded_value, total_bits


def arithmetic_decode(encoded_value, password_length, model, device='cpu'):
    """
    Decode password from arithmetic coded value.
    
    Args:
        encoded_value: Float in [0, 1) representing encoded password
        password_length: Expected password length (without START/END)
        model: Trained transformer model
        device: 'cpu' or 'cuda'
    
    Returns:
        str: Decoded password
    """
    # Start with START token
    context = [257]  # START token
    
    decoded_symbols = []
    
    # Decode until we hit END token or reach max length
    max_iterations = password_length + 1  # +1 for END token
    
    for _ in range(max_iterations):
        # Get probability distribution given current context
        probs = get_symbol_probabilities(model, context, device)
        
        # Build cumulative probability distribution
        cumsum = torch.cumsum(probs, dim=0)
        cumsum = torch.cat([torch.tensor([0.0], device=device), cumsum])
        
        # Find which symbol the encoded value falls into
        for symbol in range(258):
            symbol_low = cumsum[symbol].item()
            symbol_high = cumsum[symbol + 1].item()
            
            if symbol_low <= encoded_value < symbol_high:
                # Found the symbol!
                if symbol == 256:  # END token
                    # Done decoding
                    password = ''.join(chr(b) for b in decoded_symbols)
                    return password
                elif symbol == 257:  # START token (shouldn't happen)
                    raise ValueError("Unexpected START token during decoding")
                else:
                    decoded_symbols.append(symbol)
                    context.append(symbol)
                
                # Update encoded_value to be relative position within this symbol's range
                range_size = symbol_high - symbol_low
                encoded_value = (encoded_value - symbol_low) / range_size
                break
    
    # If we got here without END token, return what we have
    password = ''.join(chr(b) for b in decoded_symbols)
    return password


def test_password(password, model, device='cpu'):
    """Test arithmetic coding on a password."""
    print(f"\n{'='*70}")
    print(f"Testing: '{password}'")
    print(f"{'='*70}")
    
    # Encode
    encoded_value, compression_bits = arithmetic_encode(password, model, device)
    print(f"\nArithmetic Coding:")
    print(f"  Encoded value: {encoded_value:.50f}")
    print(f"  Compression size: {compression_bits:.2f} bits")
    
    # Decode
    decoded_password = arithmetic_decode(encoded_value, len(password), model, device)
    print(f"\nDecoding:")
    print(f"  Original:  '{password}'")
    print(f"  Decoded:   '{decoded_password}'")
    print(f"  Match: {'✓' if password == decoded_password else '✗ MISMATCH!'}")
    
    # Compare to entropy calculation
    from password_entropy import compute_entropy
    total_entropy, per_char_entropy, per_char_probs = compute_entropy(model, password, device)
    
    print(f"\nEntropy Calculation:")
    print(f"  Calculated entropy: {total_entropy:.2f} bits")
    print(f"  Compression size:   {compression_bits:.2f} bits")
    print(f"  Difference:         {abs(total_entropy - compression_bits):.2f} bits")
    print(f"  Match: {'✓' if abs(total_entropy - compression_bits) < 0.01 else '✗ DISCREPANCY!'}")
    
    return password == decoded_password and abs(total_entropy - compression_bits) < 0.01


def main():
    """Test arithmetic coding on several passwords."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test arithmetic coding with password model')
    parser.add_argument('--model', default='checkpoints/full_training/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('passwords', nargs='*', 
                       help='Passwords to test (default: test suite)')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return 1
    
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded model with {param_count:,} parameters")
    
    # Test passwords
    if args.passwords:
        test_cases = args.passwords
    else:
        # Default test suite
        test_cases = [
            'password',
            'password123',
            'd0ntb33vil',
            'Tr0ub4dor&3',
            'CorrectHorseBatteryStaple'
        ]
    
    print(f"\n{'='*70}")
    print(f"ARITHMETIC CODING VALIDATION")
    print(f"{'='*70}")
    
    all_passed = True
    for password in test_cases:
        passed = test_password(password, model, device)
        if not passed:
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Entropy calculations are validated by arithmetic coding!")
    else:
        print("✗ SOME TESTS FAILED")
        print("There may be a bug in entropy calculation or arithmetic coding.")
    print(f"{'='*70}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
