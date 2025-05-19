import torch
from torch.utils.data import DataLoader
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
import logging
import time
import platform
import psutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepfakeTraining")

# Set OpenMP environment variables
os.environ["OMP_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["MKL_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # Intel CPU optimization

from data.dataset import DeepfakeDataset
from models.fair_deepfake_detector import FairDeepfakeDetector
from models.losses import DAWFDDLoss
from utils.trainer import FairTrainer
from utils.metrics import FairnessMetrics

def log_system_info():
    """Log system information for debugging"""
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    logger.info(f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
    
    # PyTorch config
    logger.info(f"PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"PyTorch OpenMP enabled: {torch.backends.openmp.is_available()}")
    logger.info(f"PyTorch MKL enabled: {torch.backends.mkl.is_available()}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Environment variables
    logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    logger.info(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    logger.info(f"KMP_AFFINITY: {os.environ.get('KMP_AFFINITY', 'Not set')}")

def main(args):
    start_time = time.time()
    logger.info(f"Starting training with config: {args.config}")
    
    # Log system information
    log_system_info()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration: {config}")
    
    # Set PyTorch thread settings
    if config['training'].get('use_openmp', False):
        num_threads = config['training'].get('num_threads', 8)
        torch.set_num_threads(num_threads)
        logger.info(f"Using OpenMP with {num_threads} threads")
    
    # Create save directory if it doesn't exist
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    logger.info(f"Save directory: {config['training']['save_dir']}")
        
    # Create datasets with preprocessing
    logger.info("Creating datasets and preprocessing videos...")
    data_start = time.time()
    
    train_dataset = DeepfakeDataset(
        config['data']['data_path'],
        mode='train',
        annotations_path=config['data']['annotations_path'],
        preprocess=True,  # Enable preprocessing
        cache_dir=config['data'].get('cache_dir', None)  # Optional cache directory path
    )
    logger.info(f"Train dataset created with {len(train_dataset)} samples")
    
    val_dataset = DeepfakeDataset(
        config['data']['data_path'],
        mode='val',
        annotations_path=config['data']['annotations_path'],
        preprocess=True,  # Enable preprocessing
        cache_dir=config['data'].get('cache_dir', None)  # Optional cache directory path
    )
    logger.info(f"Validation dataset created with {len(val_dataset)} samples")
    
    logger.info(f"Dataset creation and preprocessing took {time.time() - data_start:.2f} seconds")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    logger.info(f"Initializing model with backbone: {config['model']['backbone']}")
    model_start = time.time()
    model = FairDeepfakeDetector(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes']
    )
    logger.info(f"Model initialization took {time.time() - model_start:.2f} seconds")
    
    # Initialize loss and trainer
    logger.info("Initializing trainer...")
    loss_fn = DAWFDDLoss(lambda_fair=config['training']['lambda_fair'])
    trainer = FairTrainer(
        model=model,
        loss_fn=loss_fn,
        device=config['training']['device'],
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    best_val_auc = 0
    logger.info(f"Starting training for {config['training']['num_epochs']} epochs")


    epochs = []
    train_losses = []
    train_cls_losses = []
    train_fair_losses = []
    val_aucs = []
    val_accuracies = []
    fairness_gaps = []
    group_aucs = {}
    
    for epoch in range(config['training']['num_epochs']):
        epoch_start = time.time()
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']} started")
        
        # Train epoch
        train_metrics = trainer.train_epoch(train_loader)
        
        # Evaluate
        logger.info("Evaluating on validation set...")
        val_results = trainer.evaluate(val_loader)
        val_metrics = FairnessMetrics.compute_group_metrics(
            val_results['predictions'],
            val_results['labels'],
            val_results['demographics']
        )
        
        # Save best model
        if val_metrics['overall']['auc'] > best_val_auc:
            best_val_auc = val_metrics['overall']['auc']
            model_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved with AUC: {best_val_auc:.4f} at {model_path}")
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds")
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"Val AUC: {val_metrics['overall']['auc']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['overall']['accuracy']:.4f}")
        logger.info(f"Val Fairness Gap: {val_metrics['fairness']['demographic_parity']:.4f}")

        epochs.append(epoch + 1)
        train_losses.append(train_metrics['total_loss'])
        train_cls_losses.append(train_metrics['cls_loss'])
        train_fair_losses.append(train_metrics['fair_loss'])
        val_aucs.append(val_metrics['overall']['auc'])
        val_accuracies.append(val_metrics['overall']['accuracy'])
        fairness_gaps.append(val_metrics['fairness']['demographic_parity'])

        for group, group_metrics in val_metrics.items():
            if group.startswith('group_'):
                if group not in group_aucs:
                    group_aucs[group] = []
                group_aucs[group].append(group_metrics['auc'])
        
        # Log group metrics
        for group, group_metrics in val_metrics.items():
            if group.startswith('group_'):
                logger.info(f"Group {group}: AUC={group_metrics['auc']:.4f}, Acc={group_metrics['accuracy']:.4f}")
    
    # Save final model regardless of performance
    final_model_name = f"resnet_model_epochs_{config['training']['num_epochs']}.pth"
    final_model_path = os.path.join(config['training']['save_dir'], final_model_name)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved at {final_model_path}")

    logger.info("Generating training metric plots...")
    plots_dir = os.path.join(config['training']['save_dir'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Training Losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Total Loss')
    plt.plot(epochs, train_cls_losses, 'g--', label='Classification Loss')
    plt.plot(epochs, train_fair_losses, 'r-.', label='Fairness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(plots_dir, 'training_losses.png')
    plt.savefig(loss_plot_path)
    plt.close()
    logger.info(f"Saved training losses plot to {loss_plot_path}")

    # Plot 2: Validation Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_aucs, 'b-', label='AUC')
    plt.plot(epochs, val_accuracies, 'g--', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    val_plot_path = os.path.join(plots_dir, 'validation_metrics.png')
    plt.savefig(val_plot_path)
    plt.close()
    logger.info(f"Saved validation metrics plot to {val_plot_path}")
    
    # Plot 3: Fairness Gap
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fairness_gaps, 'r-')
    plt.xlabel('Epoch')
    plt.ylabel('Demographic Parity Gap')
    plt.title('Fairness Gap (Lower is Better)')
    plt.grid(True)
    fairness_plot_path = os.path.join(plots_dir, 'fairness_gap.png')
    plt.savefig(fairness_plot_path)
    plt.close()
    logger.info(f"Saved fairness gap plot to {fairness_plot_path}")

    # Plot 4: Group AUCs
    plt.figure(figsize=(10, 6))
    for group, aucs in group_aucs.items():
        plt.plot(epochs, aucs, label=f'{group}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Group-specific AUC')
    plt.legend()
    plt.grid(True)
    group_plot_path = os.path.join(plots_dir, 'group_aucs.png')
    plt.savefig(group_plot_path)
    plt.close()
    logger.info(f"Saved group AUCs plot to {group_plot_path}")
    
    # Log total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")
    logger.info(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    
    main(args)