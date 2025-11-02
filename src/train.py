"""
Training script for Password Prediction Transformer.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os
from pathlib import Path

from model import PasswordTransformer
from dataset import PasswordDataset
from config import Config


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, 
                 lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss, ignoring padding positions.
    
    Args:
        logits: (batch, seq_len, vocab_size) predictions
        targets: (batch, seq_len) ground truth
        lengths: (batch,) actual lengths for masking
    
    Returns:
        loss: scalar
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross-entropy
    logits_flat = logits.view(-1, vocab_size)  # (batch*seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch*seq_len,)
    
    # Compute loss (ignore_index=0 masks padding)
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=0,  # Don't compute loss on padding
        reduction='mean'
    )
    
    return loss


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                     lengths: torch.Tensor) -> float:
    """
    Compute per-character accuracy, ignoring padding.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        lengths: (batch,)
    
    Returns:
        accuracy: fraction of correct predictions
    """
    # Get predictions (argmax over vocab)
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)
    
    # Create mask for non-padding positions
    batch_size, seq_len = targets.shape
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    # Count correct predictions in non-padding positions
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def train_epoch(model: PasswordTransformer, 
                train_loader: DataLoader,
                optimizer: AdamW,
                scheduler: OneCycleLR,
                device: torch.device,
                epoch: int,
                log_every: int = 100) -> dict:
    """
    Train for one epoch.
    
    Returns:
        metrics: dict with 'loss' and 'accuracy'
    """
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        lengths = batch['length']
        
        # Forward pass
        logits = model(inputs)
        
        # Compute loss and accuracy
        loss = compute_loss(logits, targets, lengths)
        acc = compute_accuracy(logits, targets, lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for transformers!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        # Update progress bar
        if batch_idx % log_every == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.3f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


@torch.no_grad()
def evaluate(model: PasswordTransformer,
             eval_loader: DataLoader,
             device: torch.device) -> dict:
    """
    Evaluate on validation set.
    
    Returns:
        metrics: dict with 'loss' and 'accuracy'
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch in tqdm(eval_loader, desc="Evaluating"):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        lengths = batch['length']
        
        # Forward pass (no gradients)
        logits = model(inputs)
        
        # Compute metrics
        loss = compute_loss(logits, targets, lengths)
        acc = compute_accuracy(logits, targets, lengths)
        
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


def save_checkpoint(model: PasswordTransformer, 
                    optimizer: AdamW,
                    epoch: int,
                    metrics: dict,
                    checkpoint_dir: str,
                    is_best: bool = False):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save epoch checkpoint
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved!")


def train(config: Config):
    """
    Main training function.
    """
    print("=" * 70)
    print("PASSWORD PREDICTION TRANSFORMER - TRAINING")
    print("=" * 70)
    
    # Set device with better error handling
    if config.training.device == 'cuda' and torch.cuda.is_available():
        try:
            # Test CUDA actually works
            test_tensor = torch.tensor([1.0]).cuda()
            device = torch.device('cuda')
            print(f"\nDevice: cuda (GPU: {torch.cuda.get_device_name(0)})")
        except RuntimeError as e:
            print(f"\n⚠ CUDA error: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
            print(f"Device: cpu")
    else:
        device = torch.device('cpu')
        print(f"\nDevice: cpu")
    
    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    train_dataset = PasswordDataset(
        config.training.train_file,
        max_len=config.model.max_seq_len
    )
    
    eval_dataset = PasswordDataset(
        config.training.eval_file,
        max_len=config.model.max_seq_len
    )
    
    # Create data loader (shuffle since dataset is already expanded by frequency)
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,  # Shuffle expanded dataset
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    # Create model
    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)
    
    model = PasswordTransformer(config.model).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    print(f"  d_model: {config.model.d_model}")
    print(f"  n_layers: {config.model.n_layers}")
    print(f"  n_heads: {config.model.n_heads}")
    print(f"  d_ff: {config.model.d_ff}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95)
    )
    
    num_training_steps = len(train_loader) * config.training.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=num_training_steps,
        pct_start=0.05,  # 5% warmup
        anneal_strategy='cos'
    )
    
    print(f"\nOptimizer: AdamW (lr={config.training.learning_rate:.2e})")
    print(f"Scheduler: OneCycleLR with {config.training.warmup_steps} warmup steps")
    print(f"Total training steps: {num_training_steps:,}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_eval_loss = float('inf')
    
    for epoch in range(1, config.training.epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, config.training.log_every
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.3f}")
        
        # Evaluate
        if epoch % config.training.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, device)
            print(f"Eval  - Loss: {eval_metrics['loss']:.4f}, "
                  f"Acc: {eval_metrics['accuracy']:.3f}")
            
            # Save checkpoint
            is_best = eval_metrics['loss'] < best_eval_loss
            if is_best:
                best_eval_loss = eval_metrics['loss']
            
            if epoch % config.training.save_every == 0:
                save_checkpoint(
                    model, optimizer, epoch,
                    {'train': train_metrics, 'eval': eval_metrics},
                    config.training.checkpoint_dir,
                    is_best=is_best
                )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best eval loss: {best_eval_loss:.4f}")


if __name__ == '__main__':
    # Use smoke test config for quick validation
    config = Config.smoke_test()
    train(config)
