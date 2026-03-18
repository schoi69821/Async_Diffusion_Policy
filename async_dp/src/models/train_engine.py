"""
Diffusion Policy Training Engine
Complete training loop with validation, early stopping, and checkpointing
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
import time
from datetime import datetime

from src.models.diffusion_net import DiffusionPolicy
from src.models.scheduler import get_scheduler
from config.settings import Config


class TrainingConfig:
    """Training hyperparameters"""
    # Basic
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6

    # Scheduler
    LR_SCHEDULER = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    LR_WARMUP_EPOCHS = 5
    LR_MIN = 1e-6

    # Early Stopping
    EARLY_STOP_PATIENCE = 15
    EARLY_STOP_MIN_DELTA = 1e-5

    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 10
    KEEP_LAST_N_CHECKPOINTS = 3

    # Validation
    VAL_SPLIT = 0.1  # 10% for validation
    VAL_EVERY_N_EPOCHS = 1

    # Training
    GRADIENT_CLIP_NORM = 1.0
    NUM_WORKERS = 0  # DataLoader workers (0 for Windows compatibility)

    # Diffusion
    NUM_TRAIN_TIMESTEPS = 100


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class TrainingLogger:
    """Simple training logger"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
        }

    def log(self, epoch, train_loss, val_loss, lr, epoch_time):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)

        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_best_epoch(self):
        if not self.history['val_loss']:
            return 0
        return int(min(range(len(self.history['val_loss'])),
                      key=lambda i: self.history['val_loss'][i]))


def create_lr_scheduler(optimizer, config, num_batches_per_epoch):
    """Create learning rate scheduler"""
    if config.LR_SCHEDULER == 'cosine':
        # Cosine annealing with warm restarts
        total_steps = config.EPOCHS * num_batches_per_epoch
        warmup_steps = config.LR_WARMUP_EPOCHS * num_batches_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return config.LR_MIN / config.LEARNING_RATE + \
                   (1 - config.LR_MIN / config.LEARNING_RATE) * \
                   (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.LR_SCHEDULER == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )

    elif config.LR_SCHEDULER == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

    return None


def train_one_epoch(model, loader, optimizer, scheduler, noise_scheduler, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        # Get data
        obs = batch['qpos'].to(device)  # (B, obs_horizon, dim)
        action = batch['action'].to(device)  # (B, pred_horizon, dim)

        # Use first observation frame as conditioning
        obs_cond = obs[:, 0, :]  # (B, dim)

        # Sample noise
        noise = torch.randn_like(action)

        # Sample random timesteps
        timesteps = torch.randint(
            0, config.NUM_TRAIN_TIMESTEPS,
            (action.shape[0],),
            device=device
        ).long()

        # Add noise to actions (forward diffusion)
        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

        # Predict noise
        noise_pred = model(noisy_action, timesteps, obs_cond)

        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.GRADIENT_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.GRADIENT_CLIP_NORM
            )

        optimizer.step()

        # Update LR scheduler (step-wise)
        if scheduler is not None and config.LR_SCHEDULER == 'cosine':
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, noise_scheduler, device, config):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        obs = batch['qpos'].to(device)
        action = batch['action'].to(device)
        obs_cond = obs[:, 0, :]

        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0, config.NUM_TRAIN_TIMESTEPS,
            (action.shape[0],),
            device=device
        ).long()

        noisy_action = noise_scheduler.add_noise(action, noise, timesteps)
        noise_pred = model(noisy_action, timesteps, obs_cond)

        loss = nn.functional.mse_loss(noise_pred, noise)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, is_best=False):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))


def train_loop(dataset, save_path, config=None, resume_from=None):
    """
    Complete training loop

    Args:
        dataset: AlohaDiffusionDataset instance
        save_path: Path to save best model
        config: TrainingConfig instance (uses default if None)
        resume_from: Path to checkpoint to resume from

    Returns:
        dict: Training history
    """
    if config is None:
        config = TrainingConfig()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    print(f"[Train] Dataset size: {len(dataset)}")

    # Train/Validation split
    val_size = int(len(dataset) * config.VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[Train] Train samples: {train_size}, Val samples: {val_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Create model
    model = DiffusionPolicy(Config.ACTION_DIM, Config.OBS_DIM).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {num_params:,}")

    # Create noise scheduler
    noise_scheduler = get_scheduler('ddpm', config.NUM_TRAIN_TIMESTEPS)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Create LR scheduler
    lr_scheduler = create_lr_scheduler(optimizer, config, len(train_loader))

    # Setup logging
    log_dir = os.path.dirname(save_path)
    os.makedirs(log_dir, exist_ok=True)
    logger = TrainingLogger(log_dir)

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        min_delta=config.EARLY_STOP_MIN_DELTA,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from and os.path.exists(resume_from):
        print(f"[Train] Resuming from {resume_from}")
        start_epoch, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, lr_scheduler
        )
        start_epoch += 1
        print(f"[Train] Resumed at epoch {start_epoch}, best loss: {best_val_loss:.4f}")

    # Training loop
    print(f"\n[Train] Starting training for {config.EPOCHS} epochs")
    print("=" * 60)

    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler,
            noise_scheduler, device, config
        )

        # Validate
        val_loss = train_loss  # Default if no validation
        if epoch % config.VAL_EVERY_N_EPOCHS == 0:
            val_loss = validate(model, val_loader, noise_scheduler, device, config)

        # Update LR scheduler (epoch-wise for plateau)
        current_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and config.LR_SCHEDULER == 'plateau':
            lr_scheduler.step(val_loss)
        elif lr_scheduler and config.LR_SCHEDULER == 'step':
            lr_scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        logger.log(epoch, train_loss, val_loss, current_lr, epoch_time)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  [*] New best model! Val loss: {val_loss:.4f}")

        # Save checkpoint
        if epoch % config.SAVE_EVERY_N_EPOCHS == 0 or is_best:
            ckpt_path = save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, lr_scheduler,
                epoch, val_loss, ckpt_path, is_best
            )

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\n[Train] Early stopping at epoch {epoch+1}")
            break

    print("=" * 60)
    print(f"[Train] Training complete!")
    print(f"[Train] Best validation loss: {best_val_loss:.4f}")

    # Save final model (just weights for inference)
    torch.save(model.state_dict(), save_path)
    print(f"[Train] Model saved to {save_path}")

    return logger.history


# Convenience function for quick training
def quick_train(data_dir, epochs=10, batch_size=16):
    """Quick training with minimal configuration"""
    from src.utils.dataset import AlohaDiffusionDataset

    dataset = AlohaDiffusionDataset(data_dir)

    config = TrainingConfig()
    config.EPOCHS = epochs
    config.BATCH_SIZE = batch_size
    config.EARLY_STOP_PATIENCE = epochs  # Disable early stopping

    return train_loop(dataset, Config.CKPT_PATH, config)


if __name__ == "__main__":
    # Test with dummy data
    from src.utils.dataset import AlohaDiffusionDataset

    print("Testing training loop with dummy data...")
    dataset = AlohaDiffusionDataset(Config.DATA_DIR)  # Will use dummy mode

    config = TrainingConfig()
    config.EPOCHS = 3
    config.BATCH_SIZE = 8
    config.EARLY_STOP_PATIENCE = 5

    history = train_loop(dataset, Config.CKPT_PATH, config)
    print(f"\nTraining history: {len(history['train_loss'])} epochs recorded")
