#!/usr/bin/env python3
"""
Quick test to verify PyTorch GPU setup.
"""
import torch
import sys

print("=" * 70)
print("PyTorch GPU Test")
print("=" * 70)

# Basic info
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print()

# CUDA check
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print()
    
    # Memory info
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {total_mem:.2f} GB")
    print()
    
    # Quick tensor test
    print("Testing GPU computation...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"✓ Matrix multiplication successful!")
    print(f"  Result shape: {z.shape}")
    print(f"  Device: {z.device}")
    print()
    
    print("=" * 70)
    print("✓ GPU setup verified! Ready for training.")
    print("=" * 70)
else:
    print("✗ CUDA not available!")
    sys.exit(1)
