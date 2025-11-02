# Password Prediction Transformer

A transformer-based model for estimating password strength through learned entropy calculation.

## Overview

This project trains a transformer model on the RockyYou password dataset to predict password character sequences. By calculating the negative log probability of each character (Shannon entropy), we can estimate password strength more accurately than traditional rule-based systems.

**Key Features:**
- ✅ Learns real-world password patterns from 14.7M examples
- ✅ Validates XKCD's "correct horse battery staple" hypothesis
- ✅ Validated with arithmetic coding (compression = entropy)
- ✅ Recognizes leet-speak substitutions and common patterns

## Quick Start

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch tqdm

# Analyze a password (requires trained model)
python password_entropy.py --full "password123"
```

## Model Downloads

**Pre-trained models are available in GitHub Releases:**

- **Full Model** (4.9M params, 38.4% accuracy): `best_model.pt` (57MB)
- **Tiny Model** (668K params, 29.7% accuracy): `tiny_model.pt` (8MB)

Download and place in:
- Full: `checkpoints/full_training/best_model.pt`
- Tiny: `checkpoints/tiny_model/best_model.pt`

**Why not in the repo?**
Models are large binary files (561MB total with all checkpoints). GitHub Releases keeps the repo lightweight while providing easy model downloads.

## Training Your Own

```bash
# Prepare data (split RockyYou into train/eval)
python prepare_data.py

# Train tiny model (8 minutes on RTX 4060)
python train_tiny.py

# Train full model (20 hours on RTX 4060)
python train_full.py
```

## Results

Example entropy estimates (full model):

| Password | Entropy | Strength |
|----------|---------|----------|
| `password` | 8 bits | 🔴 VERY WEAK |
| `password123` | 14 bits | 🔴 VERY WEAK |
| `Tr0ub4dor&3` | 73 bits | 🟢 VERY STRONG |
| `CorrectHorseBatteryStaple` | 103 bits | 🟢 VERY STRONG |

**XKCD was right!** Random words beat complex substitutions.

## Architecture

- **Transformer**: Decoder-only with causal attention
- **Full Model**: 6 layers, 256-dim, 8 heads, 4.9M params
- **Tiny Model**: 3 layers, 128-dim, 4 heads, 668K params
- **Vocab**: 258 tokens (0-255 bytes, 256=END, 257=START)

## Validation

Entropy calculations are validated with arithmetic coding:
```bash
python arithmetic_coder.py --full
```

Compression size matches calculated entropy to 0.00 bits precision! ✅

## Files

- `password_entropy.py` - Analyze password strength
- `arithmetic_coder.py` - Validate entropy with compression
- `train_full.py` / `train_tiny.py` - Training scripts
- `src/model.py` - Transformer architecture
- `src/train.py` - Training loop
- `src/dataset.py` - Password dataset loader

## License

Apache 2.0 - See LICENSE file

## Citation

If you use this work, please cite:
```
Password Prediction Transformer
https://github.com/[your-username]/pred_password
```

## Acknowledgments

- RockyYou dataset (leaked 2009, now used for security research)
- XKCD comic #936 for password wisdom
- CodeRhapsody for development assistance
