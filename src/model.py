"""
Password Prediction Transformer Model.

GPT-style decoder-only transformer for next-character prediction.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import ModelConfig


class ByteEmbedding(nn.Module):
    """
    Converts byte values (0-255) to dense d_model vectors.
    
    Input:  (batch, seq_len) with values 0-255
    Output: (batch, seq_len, d_model)
    """
    
    def __init__(self, vocab_size: int = 256, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)  # Scale like GPT-2 for better gradient flow
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) byte values 0-255
        
        Returns:
            embeddings: (batch, seq_len, d_model) scaled embeddings
        """
        return self.embedding(x) * self.scale


class LearnedPositionalEncoding(nn.Module):
    """
    Learned position embeddings for each position 0 to max_len-1.
    
    Why learned? Max 60 positions is small, and passwords have specific
    positional patterns (e.g., numbers often at end) that learning can capture.
    """
    
    def __init__(self, max_len: int = 60, d_model: int = 256):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) embedded input
        
        Returns:
            x + positional encoding: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)
        
        # Get position embeddings and add to input
        pos_emb = self.pos_embedding(positions)  # (batch, seq_len, d_model)
        return x + pos_emb


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask: position i can only attend to positions 0..i.
    
    This is what makes it "autoregressive" - the model can't cheat by
    looking at future characters when predicting the next one.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        mask: (seq_len, seq_len) with -inf in upper triangle, 0 elsewhere
        
    Example (seq_len=4):
        [[  0, -inf, -inf, -inf],   ← Position 0 sees only itself
         [  0,   0, -inf, -inf],    ← Position 1 sees 0,1
         [  0,   0,   0, -inf],     ← Position 2 sees 0,1,2
         [  0,   0,   0,   0]]      ← Position 3 sees 0,1,2,3
    """
    # Create upper triangular matrix of 1s (diagonal=1 means start 1 above diagonal)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    
    # Replace 1s with -inf (will become 0 after softmax)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    return mask


class TransformerBlock(nn.Module):
    """
    One transformer decoder block with:
    1. Multi-head self-attention (causal)
    2. Feed-forward network
    3. Layer normalization (pre-norm style)
    4. Residual connections
    """
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, 
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) format
        )
        
        # Feed-forward network: d_model -> d_ff -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Smoother than ReLU, used in GPT-2
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm: normalize before each operation)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input
            causal_mask: (seq_len, seq_len) attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm: normalize BEFORE attention
        normed = self.ln1(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            normed, normed, normed,  # Query, Key, Value all same (self-attention)
            attn_mask=causal_mask,
            need_weights=False  # Don't need attention weights
        )
        x = x + self.dropout(attn_out)  # Residual connection
        
        # Pre-norm: normalize BEFORE feed-forward
        normed = self.ln2(x)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(normed)
        x = x + ffn_out  # Residual connection
        
        return x


class PasswordTransformer(nn.Module):
    """
    Complete GPT-style transformer for password prediction.
    
    Architecture:
        Input bytes → Embedding → Positional Encoding → 
        N × TransformerBlock → LayerNorm → Output Projection → Logits
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        
        # Input processing
        self.byte_embedding = ByteEmbedding(config.vocab_size, config.d_model)
        self.pos_encoding = LearnedPositionalEncoding(config.max_seq_len, config.d_model)
        
        # Transformer stack
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output projection: d_model -> vocab_size
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        # Final layer norm (GPT-2 style)
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: (batch, seq_len) byte values 0-255
            padding_mask: (batch, seq_len) True for padding positions (optional)
        
        Returns:
            logits: (batch, seq_len, vocab_size) unnormalized log probabilities
        """
        batch_size, seq_len = x.shape
        
        # Create causal mask: position i can only see positions 0..i
        causal_mask = generate_causal_mask(seq_len, x.device)
        
        # Embedding + positional encoding
        x = self.byte_embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)    # Add position info
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Quick test to verify model shapes."""
    print("Testing PasswordTransformer...")
    
    # Create small model
    config = ModelConfig(d_model=128, n_layers=2, n_heads=4, d_ff=512)
    model = PasswordTransformer(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create dummy input: batch of 2 passwords, each 10 bytes
    batch_size = 2
    seq_len = 10
    x = torch.randint(1, 256, (batch_size, seq_len))  # Avoid 0 (padding)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Verify shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Shape test passed!")
    
    # Test with different sequence length
    x2 = torch.randint(1, 256, (batch_size, 20))
    logits2 = model(x2)
    assert logits2.shape == (batch_size, 20, config.vocab_size)
    print("✓ Variable length test passed!")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_model()
