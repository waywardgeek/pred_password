# Password Prediction Transformer - Design Document

## Project Overview

Build a GPT-style transformer model to predict the next character in a password sequence, trained on the RockYou dataset (16.4M accounts, 9.6M unique passwords ≥8 characters).

**Goal**: Compare against hand-crafted predictor and CMU study results, while exploring homeostatic neuron mechanisms.

---

## 1. Input Representation

### Character Encoding
- **Byte-level representation**: Each character is treated as a single byte (0-255)
- **Vocabulary size**: 256 (full 8-bit range)
- **Special tokens**:
  - `0`: `<EOS>` (End of Sequence / End of Password)
  - `1-255`: Standard byte values (ASCII + extended)
  
**Why byte-level?**
- Natural UTF-8 support (multi-byte sequences learned automatically)
- No need for character-level vocabulary management
- Handles any 8-bit character naturally
- Zero is never used in passwords (null terminator), perfect for padding/EOS

### Sequence Format
```
Input:     [112, 97, 115, 115, 119, 111, 114, 100]  # "password"
Target:    [97, 115, 115, 119, 111, 114, 100, 0]    # predict next char, then EOS
```

### Maximum Sequence Length
- **Max length**: 60 characters
- **Padding**: Right-padded with `0` for sequences < 60
- **Attention mask**: Mask out padding positions

**Distribution in dataset**:
- Most passwords: 8-12 characters
- Long tail: up to 60+ characters
- 60 chars covers >99% of real passwords

---

## 2. Model Architecture (GPT-style)

### Overview: Decoder-Only Transformer

```
Input (byte sequence)
    ↓
Embedding Layer (256 → d_model)
    ↓
Positional Encoding (learned, up to 60 positions)
    ↓
Transformer Decoder Stack (N layers)
    ├── Multi-Head Self-Attention (causal masking)
    ├── Feed-Forward Network
    └── Layer Normalization + Residual
    ↓
Output Projection (d_model → 256)
    ↓
Softmax → Next character probability distribution
```

### Hyperparameters (Starting Point)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` | 256 | Embedding dimension - moderate for character-level |
| `n_heads` | 8 | Multi-head attention (32 dims per head) |
| `n_layers` | 6 | Standard GPT-small depth |
| `d_ff` | 1024 | Feed-forward dimension (4x d_model) |
| `dropout` | 0.1 | Standard regularization |
| `max_seq_len` | 60 | Maximum password length |
| `vocab_size` | 256 | Full byte range |

**Parameter count**: ~11M parameters (small by LLM standards, appropriate for this task)

---

## 3. Detailed Architecture Components

### 3.1 Embedding Layer

```python
class ByteEmbedding(nn.Module):
    """
    Maps byte values (0-255) to dense d_model vectors.
    """
    def __init__(self, vocab_size=256, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)  # Scale like GPT-2
    
    def forward(self, x):
        # x: (batch, seq_len) with values 0-255
        return self.embedding(x) * self.scale  # (batch, seq_len, d_model)
```

**Why scale?** Following GPT-2/GPT-3 convention - helps gradient flow in early layers.

### 3.2 Positional Encoding

**Learned vs Sinusoidal**: Use **learned** positional embeddings.

```python
class LearnedPositionalEncoding(nn.Module):
    """
    Learned position embeddings for each position 0-59.
    """
    def __init__(self, max_len=60, d_model=256):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)  # Broadcast add
```

**Why learned?** 
- Maximum 60 positions (small, learnable)
- Can capture password-specific positional patterns (e.g., numbers at end)
- More flexible than sinusoidal for short sequences

### 3.3 Transformer Decoder Block

Standard transformer block with **causal masking** (can only attend to previous positions):

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention (causal)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Smoother than ReLU, used in GPT-2
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm architecture like GPT-2)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask):
        # Pre-norm architecture (more stable training)
        # Self-attention with residual
        normed = self.ln1(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        normed = self.ln2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x
```

**Key design choices**:
- **Pre-norm**: Layer norm before each sub-layer (better gradient flow)
- **GELU activation**: Smoother than ReLU, standard in modern transformers
- **Causal masking**: Position `i` can only attend to positions `0...i`

### 3.4 Causal Mask Generation

```python
def generate_causal_mask(seq_len, device):
    """
    Create causal mask: upper triangular matrix of -inf.
    Position i can attend to positions 0..i (lower triangle + diagonal).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # (seq_len, seq_len)
```

**Attention pattern** (seq_len=5):
```
     0    1    2    3    4
0 [ OK  -inf -inf -inf -inf]  ← Position 0 sees only itself
1 [ OK   OK  -inf -inf -inf]  ← Position 1 sees 0,1
2 [ OK   OK   OK  -inf -inf]  ← Position 2 sees 0,1,2
3 [ OK   OK   OK   OK  -inf]
4 [ OK   OK   OK   OK   OK ]  ← Position 4 sees all
```

### 3.5 Output Layer

```python
class OutputProjection(nn.Module):
    """
    Project d_model back to vocabulary logits.
    """
    def __init__(self, d_model=256, vocab_size=256):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.proj(x)  # (batch, seq_len, vocab_size)
```

**No weight tying**: Unlike some LMs, we don't tie input embedding and output projection weights (byte-level is different from word-level).

---

## 4. Complete Model Class

```python
class PasswordTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=60,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input processing
        self.byte_embedding = ByteEmbedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(max_seq_len, d_model)
        
        # Transformer stack
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = OutputProjection(d_model, vocab_size)
        
        # Final layer norm (GPT-2 style)
        self.ln_f = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch, seq_len) byte values 0-255
            padding_mask: (batch, seq_len) True for padding positions
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Create causal mask
        causal_mask = generate_causal_mask(seq_len, x.device)
        
        # Combine with padding mask if provided
        if padding_mask is not None:
            # Expand padding mask for attention
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)
            causal_mask = causal_mask.unsqueeze(0) + padding_mask.masked_fill(padding_mask, float('-inf'))
        
        # Embedding + positional encoding
        x = self.byte_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        return logits  # (batch, seq_len, vocab_size)
```

---

## 5. Training Pipeline

### 5.1 Dataset Class

```python
class PasswordDataset(Dataset):
    """
    Dataset for password sequences with frequency-weighted sampling.
    """
    def __init__(self, data_file, max_len=60):
        self.max_len = max_len
        self.passwords = []
        self.weights = []
        
        # Load passwords with frequency counts
        with open(data_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    count = int(parts[0])
                    password = parts[1]
                    
                    # Filter valid passwords
                    if 8 <= len(password) <= max_len:
                        # Convert to byte representation
                        byte_seq = [ord(c) for c in password]
                        self.passwords.append(byte_seq)
                        self.weights.append(count)  # Frequency for sampling
        
        # Normalize weights for sampling
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        print(f"Loaded {len(self.passwords):,} passwords")
    
    def __len__(self):
        return len(self.passwords)
    
    def __getitem__(self, idx):
        password = self.passwords[idx]
        
        # Create input/target pairs
        # Input: password bytes
        # Target: shift left by 1, append EOS (0)
        input_seq = password[:]
        target_seq = password[1:] + [0]  # Next char prediction + EOS
        
        # Pad to max_len
        input_seq = input_seq + [0] * (self.max_len - len(input_seq))
        target_seq = target_seq + [0] * (self.max_len - len(target_seq))
        
        return {
            'input': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_seq, dtype=torch.long),
            'length': len(password)
        }
```

### 5.2 Weighted Sampling

**Key insight**: Common passwords should be seen more during training!

```python
from torch.utils.data import WeightedRandomSampler

# Create sampler based on frequency
sampler = WeightedRandomSampler(
    weights=dataset.weights,
    num_samples=len(dataset),
    replacement=True  # Allow sampling same password multiple times
)

train_loader = DataLoader(
    dataset,
    batch_size=128,
    sampler=sampler,  # Frequency-weighted sampling
    num_workers=4,
    pin_memory=True
)
```

**Why weighted sampling?**
- `123456789` (76,789 accounts) should be ~76,000x more likely than rare passwords
- Matches real-world distribution
- Model learns common patterns first

### 5.3 Loss Function

**Cross-Entropy Loss** with **masking for padding**:

```python
def compute_loss(logits, targets, lengths):
    """
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) ground truth next characters
        lengths: (batch,) actual password lengths
    
    Returns:
        loss: scalar
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross-entropy
    logits_flat = logits.view(-1, vocab_size)  # (batch*seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch*seq_len,)
    
    # Compute loss (ignore_index=0 for padding)
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=0,  # Don't compute loss on padding
        reduction='mean'
    )
    
    return loss
```

**Alternative**: Per-position loss weighting (early positions matter more?). Start with uniform.

### 5.4 Optimization

**AdamW optimizer** (Adam with weight decay):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,          # Standard transformer LR
    betas=(0.9, 0.95),  # Slightly higher beta2 for stability
    eps=1e-8,
    weight_decay=0.1   # Regularization
)
```

**Learning rate schedule** - Warmup + Cosine Decay:

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=3e-4,
    total_steps=num_training_steps,
    pct_start=0.05,  # 5% warmup
    anneal_strategy='cos',
    cycle_momentum=False
)
```

**Why OneCycleLR?**
- Warmup prevents early instability
- Cosine decay gradually reduces LR
- Single cycle for convergence

### 5.5 Training Loop (Simplified)

```python
def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader):
        inputs = batch['input'].to(device)      # (batch, 60)
        targets = batch['target'].to(device)    # (batch, 60)
        lengths = batch['length']
        
        # Forward pass
        logits = model(inputs)  # (batch, 60, 256)
        
        # Compute loss
        loss = compute_loss(logits, targets, lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)
```

**Gradient clipping**: Prevents exploding gradients, essential for stable transformer training.

---

## 6. Evaluation Metrics

### 6.1 Perplexity
Standard language modeling metric:

```python
perplexity = torch.exp(loss)
```

**Lower is better**. Measures how "surprised" the model is by the data.

### 6.2 Next-Character Accuracy

```python
def compute_accuracy(logits, targets, lengths):
    """
    Accuracy: % of correctly predicted next characters.
    """
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)
    
    # Mask padding
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()
```

### 6.3 Top-K Accuracy

"Is the correct character in the top-K predictions?"

```python
def top_k_accuracy(logits, targets, k=5):
    """
    % of times correct character is in top-K predictions.
    """
    top_k_preds = logits.topk(k, dim=-1).indices  # (batch, seq_len, k)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    
    correct = (top_k_preds == targets_expanded).any(dim=-1)
    return correct.float().mean().item()
```

### 6.4 Position-Specific Accuracy

Track accuracy by position in password (interesting for pattern analysis):

```python
position_accuracies = defaultdict(list)
for pos in range(seq_len):
    mask = (torch.arange(seq_len) == pos)
    pos_correct = (predictions[:, pos] == targets[:, pos]).float().mean()
    position_accuracies[pos].append(pos_correct)
```

**Hypothesis**: Accuracy might be higher for later positions (more context).

---

## 7. Inference / Generation

### 7.1 Greedy Decoding

Given a prefix, predict the rest:

```python
def predict_next_chars(model, prefix, max_new_chars=10):
    """
    Given prefix (e.g., "passw"), predict next characters.
    """
    model.eval()
    
    # Convert prefix to bytes
    input_seq = [ord(c) for c in prefix]
    
    with torch.no_grad():
        for _ in range(max_new_chars):
            # Pad to max_len
            padded = input_seq + [0] * (60 - len(input_seq))
            x = torch.tensor([padded], dtype=torch.long)
            
            # Forward pass
            logits = model(x)  # (1, 60, 256)
            
            # Get prediction for last non-padding position
            next_pos = len(input_seq)
            next_logits = logits[0, next_pos - 1, :]  # (256,)
            
            # Greedy: take argmax
            next_byte = next_logits.argmax().item()
            
            # Stop if EOS
            if next_byte == 0:
                break
            
            input_seq.append(next_byte)
    
    # Convert back to string
    return ''.join(chr(b) for b in input_seq)
```

### 7.2 Sampling Strategies

**Temperature sampling**:
```python
# Higher temperature = more random
probs = F.softmax(next_logits / temperature, dim=-1)
next_byte = torch.multinomial(probs, 1).item()
```

**Top-K sampling**:
```python
# Only sample from top-K most likely
top_k_logits, top_k_indices = next_logits.topk(k=10)
top_k_probs = F.softmax(top_k_logits, dim=-1)
next_byte = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
```

---

## 8. Training Plan

### Phase 1: Baseline Training (2-3 days on GPU)

**Goal**: Establish baseline performance.

- **Epochs**: 10
- **Batch size**: 128
- **Learning rate**: 3e-4
- **Hardware**: Single GPU (A100 or V100 recommended)
- **Time**: ~4-6 hours per epoch on 8.6M passwords

**Expected results**:
- Perplexity: 3-5 (reasonable for password prediction)
- Top-1 accuracy: 40-50%
- Top-5 accuracy: 70-80%

### Phase 2: Hyperparameter Tuning

Experiment with:
- `d_model`: 128, 256, 512
- `n_layers`: 4, 6, 8
- `n_heads`: 4, 8, 16
- Learning rate: 1e-4, 3e-4, 1e-3
- Batch size: 64, 128, 256

**Use eval set to select best model**.

### Phase 3: Homeostatic Mechanisms (Research)

**Hypothesis**: Can homeostatic neurons improve learning?

Implement:
1. Track neuron activation statistics
2. Adjust biases to maintain target activation levels
3. Compare vs. baseline

**Evaluation**: Does it improve perplexity, accuracy, or training stability?

---

## 9. Laptop Development & Validation (RTX 4070 - 8GB VRAM)

### 9.1 Hardware Constraints

**Your Setup**: Alienware laptop with NVIDIA RTX 4070 (8GB VRAM)

**What's possible**:
- ✅ Full model architecture (4.9M params uses ~20MB)
- ✅ Training with reasonable batch sizes (16-32)
- ✅ Mixed precision training (FP16) for 2x speedup
- ✅ Complete validation before scaling to bigger GPUs

**Memory breakdown** (batch_size=16):
```
Model parameters (FP32):     ~20 MB
Optimizer state (AdamW):     ~60 MB (2x params for momentum)
Activations (batch=16):      ~100 MB
Gradients:                   ~20 MB
PyTorch overhead:            ~500 MB
-------------------------
Total:                       ~700 MB  ← Well within 8GB!
```

**Strategy**: Start small, validate thoroughly, then scale up at work.

---

### 9.2 Phase 0: Smoke Test (5 minutes)

**Goal**: Verify everything works end-to-end with tiny data.

**Dataset**: First 1,000 passwords from training set
```bash
head -1000 min8_train.data > min8_train_tiny.data
head -100 min8_eval.data > min8_eval_tiny.data
```

**Model config** (same as production):
```python
config = {
    'd_model': 256,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 1024,
    'dropout': 0.1,
    'max_seq_len': 60,
    'vocab_size': 256
}
```

**Training**:
- Batch size: 16
- Epochs: 2
- Time: ~2 minutes
- **Goal**: No crashes, loss decreases

**Success criteria**:
- ✅ Model loads and runs
- ✅ Loss decreases from epoch 1 to 2
- ✅ Can generate predictions (even if nonsense)
- ✅ No memory errors

**Expected output**:
```
Epoch 1: Loss=5.2, Top-1 Acc=15%
Epoch 2: Loss=4.8, Top-1 Acc=22%
✓ Smoke test passed!
```

---

### 9.3 Phase 1: Small-Scale Training (1-2 hours)

**Goal**: Validate architecture and training pipeline on representative subset.

**Dataset**: 100K passwords (~1% of full dataset)
```python
# Randomly sample 100K from training set
import random
random.seed(42)

with open('min8_train.data', 'r') as f:
    all_lines = f.readlines()

sampled = random.sample(all_lines, 100_000)

with open('min8_train_100k.data', 'w') as f:
    f.writelines(sampled)
```

**Why 100K?**
- Large enough to see real patterns
- Small enough to iterate quickly
- Still includes common passwords (weighted sampling)

**Training config**:
```python
config = {
    # Model (same as production)
    'd_model': 256,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 1024,
    'dropout': 0.1,
    
    # Training
    'batch_size': 32,        # Comfortable for 8GB
    'epochs': 10,
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'gradient_clip': 1.0,
    
    # Memory optimization
    'mixed_precision': True,  # FP16 for 2x speedup
    'gradient_accumulation': 1  # No accumulation needed yet
}
```

**Expected time**: 
- Steps per epoch: 100K / 32 = 3,125
- Time per step: ~30ms (with FP16)
- Per epoch: 94 seconds
- **10 epochs: ~16 minutes**

**Success criteria**:
- ✅ Perplexity < 10 (shows learning)
- ✅ Top-1 accuracy > 30%
- ✅ Top-5 accuracy > 60%
- ✅ Can predict common patterns (e.g., "password" → "password1")
- ✅ No overfitting (train/eval gap < 20%)

**Validation checks**:
```python
# Test on common passwords
test_passwords = ['password', 'iloveyou', '12345678']
for pwd in test_passwords:
    prediction = model.predict_next_chars(pwd[:4], max_new=4)
    print(f"{pwd[:4]}... → {prediction}")
    # Should get something reasonable!
```

**What to observe**:
1. **Loss curve**: Should decrease smoothly
2. **Accuracy**: Should improve each epoch
3. **Common passwords**: Should predict well (high frequency)
4. **Rare passwords**: Might be bad (not enough data)
5. **Generation quality**: Run inference, see if patterns make sense

---

### 9.4 Memory Optimization Techniques

If you hit memory issues, try these in order:

#### 1. Mixed Precision Training (Free 2x speedup!)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():  # FP16 for forward/backward
        logits = model(inputs)
        loss = compute_loss(logits, targets)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Why?** FP16 uses half the memory, nearly same accuracy.

#### 2. Reduce Batch Size
```python
batch_size = 16  # Instead of 32
# Still trains fine, just slower
```

#### 3. Gradient Accumulation
```python
# Simulate batch_size=64 with 4 steps of size 16
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = compute_loss(...) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Trade**: Same effective batch size, ~4x slower.

#### 4. Gradient Checkpointing (Last resort)
```python
# Trade compute for memory (recompute activations during backward)
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(x):
    for layer in self.layers:
        x = checkpoint(layer, x)  # Saves memory, costs time
    return x
```

**Trade**: 30% slower, saves ~40% memory.

---

### 9.5 Laptop Training Best Practices

**Before training**:
```bash
# Check GPU
nvidia-smi

# Close other GPU apps (Chrome sometimes uses GPU!)
# Monitor during training
watch -n 1 nvidia-smi
```

**During training**:
- Monitor GPU memory usage
- Watch temperature (throttles if too hot)
- Save checkpoints every epoch
- Log to tensorboard for visualization

**Power settings**:
```bash
# Set to performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 125  # Set power limit (adjust for your laptop)
```

**Cooling**: 
- Use laptop on hard surface (not bed!)
- Consider cooling pad for long runs
- Watch temps: >85°C = throttling risk

---

### 9.6 Validation Experiments (2-3 days on laptop)

**Goal**: Validate design choices before big training run.

#### Experiment 1: Model Size Sweep
Train on 100K passwords, vary model size:

| Config | d_model | n_layers | Params | Time/epoch | Top-1 Acc |
|--------|---------|----------|--------|------------|-----------|
| Tiny   | 128     | 4        | 1.2M   | 60s        | ? |
| Small  | 256     | 6        | 4.9M   | 94s        | ? |
| Medium | 384     | 8        | 12M    | 180s       | ? |

**Question**: Is bigger always better? Or do we overfit on 100K?

#### Experiment 2: Frequency Weighting
Compare weighted vs. uniform sampling:

| Sampling | Top-1 (common) | Top-1 (rare) | Perplexity |
|----------|----------------|--------------|------------|
| Uniform  | ?              | ?            | ? |
| Weighted | ?              | ?            | ? |

**Hypothesis**: Weighted should do better on common passwords.

#### Experiment 3: Sequence Length
Most passwords are 8-12 chars. Do we need max_len=60?

| max_len | Memory | Time/epoch | Accuracy |
|---------|--------|------------|----------|
| 20      | Lower  | Faster     | ? |
| 40      | Medium | Medium     | ? |
| 60      | Higher | Slower     | ? |

**Trade-off**: Shorter = faster, but truncates long passwords.

#### Experiment 4: Learning Rate
Find optimal learning rate on 100K subset:

| LR    | Converges? | Final Loss | Notes |
|-------|------------|------------|-------|
| 1e-4  | ?          | ?          | Too slow? |
| 3e-4  | ?          | ?          | Baseline |
| 1e-3  | ?          | ?          | Unstable? |

---

### 9.7 Decision Point: Scale Up or Iterate?

**After laptop experiments, decide**:

### ✅ Ready to scale if:
- Small model (100K) achieves Top-1 > 30%
- Common passwords predict well
- No signs of major bugs
- Architecture seems sound

### 🔄 Iterate more if:
- Loss doesn't decrease
- Predictions are random garbage
- Memory/training issues
- Unclear if design is good

---

### 9.8 Scaling to Full Dataset (At Work)

**Once validated on laptop**, scale to full 8.6M passwords:

**Hardware options at Google**:
1. **Single A100 (80GB)**: 
   - Batch size: 256
   - 10 epochs: ~6 hours
   - Cost: Moderate

2. **Multi-GPU (4x A100)**:
   - Batch size: 1024 (256 per GPU)
   - 10 epochs: ~2 hours
   - Cost: Higher, but fast iteration

3. **TPU v4 Pod**:
   - Massive parallelism
   - 10 epochs: ~30 minutes
   - Cost: Check quotas

**Transfer learning**:
```python
# Load checkpoint from laptop
checkpoint = torch.load('laptop_model_100k.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training on full dataset
# (Already learned basic patterns, just needs more data)
```

---

### 9.9 Laptop Development Workflow Summary

```
Day 1: Implementation
├── Implement model.py (PasswordTransformer)
├── Implement dataset.py (PasswordDataset)
├── Implement train.py (training loop)
└── Run smoke test (1K passwords, 5 min)
    ✓ Verify no crashes

Day 2: Small-Scale Validation
├── Create 100K subset
├── Train baseline (10 epochs, 16 min)
├── Evaluate metrics
└── Test inference/generation
    ✓ Verify reasonable results

Day 3: Experiments
├── Model size sweep (3 runs, ~1 hour)
├── Sampling strategy (2 runs, ~30 min)
├── Learning rate sweep (3 runs, ~1 hour)
└── Analyze results
    ✓ Select best config

Day 4: Final Validation
├── Train best config on 100K (20 epochs)
├── Thorough evaluation
├── Document findings
└── Prepare for scale-up
    ✓ Ready for full training

Then: Scale to full dataset at work!
```

---

### 9.10 Laptop-Friendly Configuration File

**`config_laptop.yaml`**:
```yaml
# Model architecture (same as production)
model:
  d_model: 256
  n_layers: 6
  n_heads: 8
  d_ff: 1024
  dropout: 0.1
  max_seq_len: 60
  vocab_size: 256

# Training (optimized for 8GB VRAM)
training:
  batch_size: 32
  epochs: 10
  learning_rate: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 500
  gradient_clip: 1.0
  mixed_precision: true
  
  # Dataset
  train_file: 'min8_train_100k.data'  # Start small!
  eval_file: 'min8_eval.data'
  weighted_sampling: true
  
  # Checkpointing
  save_every: 1  # Save every epoch
  checkpoint_dir: 'checkpoints/'
  
  # Logging
  log_every: 100  # steps
  eval_every: 1   # epochs

# Hardware
device: 'cuda'
num_workers: 4  # Data loading threads
pin_memory: true
```

**`config_production.yaml`** (for Google GPUs):
```yaml
# Same model, but scale up training
training:
  batch_size: 256           # 8x larger
  train_file: 'min8_train.data'  # Full dataset
  epochs: 20                # More epochs
  # ... rest same
```

---

### 9.11 Quick Reference: Laptop Commands

**Setup**:
```bash
# On laptop
cd ~/password_prediction
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm pyyaml tensorboard

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**Smoke test**:
```bash
head -1000 min8_train.data > min8_train_tiny.data
python train.py --config config_smoke.yaml --quick-test
```

**Small-scale training**:
```bash
python create_subset.py --input min8_train.data --output min8_train_100k.data --size 100000
python train.py --config config_laptop.yaml
```

**Monitor**:
```bash
# Terminal 1: Training
python train.py --config config_laptop.yaml

# Terminal 2: GPU monitor
watch -n 1 nvidia-smi

# Terminal 3: Tensorboard
tensorboard --logdir runs/
```

**Evaluate**:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --data min8_eval.data
```

**Generate samples**:
```bash
python generate.py --checkpoint checkpoints/best_model.pt --prefix "passw" --num-samples 10
```

---

## 10. Implementation Checklist

### Core Model
- [ ] `ByteEmbedding` class
- [ ] `LearnedPositionalEncoding` class
- [ ] `TransformerBlock` class
- [ ] `PasswordTransformer` main model
- [ ] Causal mask generation
- [ ] Weight initialization

### Data Pipeline
- [ ] `PasswordDataset` class
- [ ] Weighted sampler based on frequency
- [ ] Data loading with batching
- [ ] Validation/test split handling

### Training
- [ ] Loss function with padding masking
- [ ] AdamW optimizer setup
- [ ] Learning rate scheduler
- [ ] Gradient clipping
- [ ] Training loop with progress tracking
- [ ] Checkpoint saving/loading

### Evaluation
- [ ] Perplexity computation
- [ ] Top-1 accuracy
- [ ] Top-K accuracy
- [ ] Position-specific metrics
- [ ] Validation loop

### Inference
- [ ] Greedy decoding
- [ ] Temperature sampling
- [ ] Top-K sampling
- [ ] Prefix-based prediction

### Utilities
- [ ] Configuration file (YAML or dataclass)
- [ ] Logging (tensorboard or wandb)
- [ ] Model summary (param count)
- [ ] Training time estimation

---

## 11. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | GPT-style decoder | Autoregressive next-char prediction |
| **Tokenization** | Byte-level (0-255) | No vocab management, UTF-8 compatible |
| **Max length** | 60 characters | Covers >99% of passwords |
| **Embedding dim** | 256 | Reasonable for character-level |
| **Layers** | 6 | Standard GPT-small depth |
| **Heads** | 8 | 32 dims per head (d_model/n_heads) |
| **FFN dim** | 1024 | 4x d_model (standard) |
| **Positional encoding** | Learned | Better for short sequences |
| **Normalization** | Pre-norm | More stable training |
| **Activation** | GELU | Smoother than ReLU |
| **Sampling** | Weighted by frequency | Matches real-world distribution |
| **Optimizer** | AdamW | Standard for transformers |
| **LR schedule** | Warmup + cosine | Stable convergence |
| **Loss** | Cross-entropy | Standard for classification |

---

## 12. Comparison to Hand-Crafted Predictor

**After training, compare**:

1. **Perplexity**: Lower = better password modeling
2. **Top-K accuracy**: How often is correct char in top-K?
3. **Common password prediction**: Accuracy on top-100 passwords
4. **Rare password prediction**: Does ML generalize better?
5. **Pattern recognition**: Does it learn "password" + number patterns?

**Hypothesis**: Transformer should beat hand-crafted on:
- Rare passwords (generalization from patterns)
- Long-range dependencies (attention mechanism)
- Implicit pattern learning (no manual rules)

**Hand-crafted might win on**:
- Very common passwords (if rules are explicit)
- Low-data regime (ML needs lots of data)

---

## 13. Next Steps

1. **Review this design** - discuss any changes
2. **Implement core model** - PyTorch classes
3. **Test on small subset** - verify shapes, training works
4. **Full training** - run on complete dataset
5. **Evaluate** - compare to baselines
6. **Experiment** - homeostatic mechanisms, architecture variations

---

## Appendix A: Model Size Calculation

```
Embedding layer:       256 * 256 = 65,536
Positional encoding:   60 * 256 = 15,360
Transformer blocks (6x):
  - Attention (Q,K,V):   256 * 256 * 3 = 196,608 per block
  - Attention out:       256 * 256 = 65,536 per block
  - FFN layer 1:         256 * 1024 = 262,144 per block
  - FFN layer 2:         1024 * 256 = 262,144 per block
  - Layer norms:         ~1,000 per block
  Total per block:       ~787,000
  Total 6 blocks:        ~4,722,000
Output projection:     256 * 256 = 65,536

TOTAL: ~4.9M parameters (trainable)
```

**Memory footprint** (float32):
- Parameters: ~20 MB
- Activations (batch=128): ~300 MB
- Gradients: ~20 MB
- **Total: ~350 MB** (easily fits on modern GPUs)

---

## Appendix B: Training Time Estimation

**Dataset**: 8.6M passwords
**Batch size**: 128
**Steps per epoch**: 8,600,000 / 128 = 67,188 steps
**Time per step**: ~50ms (A100 GPU)

**Per epoch**: 67,188 * 0.05s = 3,359s = **56 minutes**
**10 epochs**: **~9.3 hours**

**Recommendations**:
- Use mixed precision (float16) for 2x speedup
- Multi-GPU training if available
- Save checkpoints every epoch

---

## Appendix C: Files to Create

```
password_prediction/
├── model.py                 # PasswordTransformer class
├── dataset.py               # PasswordDataset class
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── generate.py              # Inference/generation script
├── config.py                # Configuration dataclass
├── utils.py                 # Helper functions
└── requirements.txt         # Dependencies
```

---

**Ready to implement?** Let me know if you want to discuss any design choices first!
