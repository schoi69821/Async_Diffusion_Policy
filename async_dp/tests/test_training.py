"""
Tests for Training Engine
"""
import torch
import numpy as np
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_engine import (
    TrainingConfig, EarlyStopping, TrainingLogger,
    train_one_epoch, validate, train_loop
)
from src.utils.dataset import AlohaDiffusionDataset
from config.settings import Config


def test_early_stopping():
    """Test early stopping logic"""
    es = EarlyStopping(patience=3, min_delta=0.01)

    # Improving losses
    assert es(1.0) == False
    assert es(0.9) == False
    assert es(0.8) == False

    # Stagnating losses (should trigger after patience)
    assert es(0.8) == False  # counter = 1
    assert es(0.79) == False  # counter = 2 (within min_delta)
    assert es(0.79) == True   # counter = 3, should stop

    print("[Test] Early stopping: PASSED")


def test_training_config():
    """Test training configuration defaults"""
    config = TrainingConfig()

    assert config.EPOCHS == 100
    assert config.BATCH_SIZE == 32
    assert config.LEARNING_RATE == 1e-4
    assert config.VAL_SPLIT == 0.1
    assert config.GRADIENT_CLIP_NORM == 1.0

    print("[Test] Training config: PASSED")


def test_training_logger():
    """Test training logger"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(tmpdir)

        logger.log(0, 1.0, 0.9, 1e-4, 10.0)
        logger.log(1, 0.8, 0.7, 1e-4, 10.0)
        logger.log(2, 0.6, 0.5, 1e-5, 10.0)

        assert len(logger.history['train_loss']) == 3
        assert logger.get_best_epoch() == 2  # Lowest val_loss

        # Check file was saved
        assert os.path.exists(os.path.join(tmpdir, 'training_log.json'))

    print("[Test] Training logger: PASSED")


def test_mini_training_loop():
    """Test minimal training loop with dummy data"""
    # Create dummy dataset
    dataset = AlohaDiffusionDataset(Config.DATA_DIR)  # Uses dummy mode

    # Very short training
    config = TrainingConfig()
    config.EPOCHS = 2
    config.BATCH_SIZE = 4
    config.EARLY_STOP_PATIENCE = 10
    config.SAVE_EVERY_N_EPOCHS = 100  # Don't save during test

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_model.pth')

        history = train_loop(dataset, save_path, config)

        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
        assert os.path.exists(save_path)

    print("[Test] Mini training loop: PASSED")


if __name__ == "__main__":
    test_early_stopping()
    test_training_config()
    test_training_logger()
    test_mini_training_loop()
    print("\nAll training tests passed!")
