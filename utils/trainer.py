import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepfakeTrainer")

class FairTrainer:
    def __init__(self, model, loss_fn, device='cpu', lr=1e-4):
        self.model = model
        self.model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        
        # Log model architecture and parameters
        logger.info(f"Model architecture: {type(model).__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"Using device: {device}")
        logger.info(f"OpenMP threads: {torch.get_num_threads()}")
        
        # Convert lr to float if it's a string
        if isinstance(lr, str):
            lr = float(lr)
            logger.info(f"Converted learning rate from string to float: {lr}")
        
        # Log optimizer settings
        logger.info(f"Learning rate: {lr}")
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_fair_loss = 0
        
        # Log memory usage before training
        if torch.cuda.is_available():
            logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        start_time = time.time()
        logger.info(f"Starting epoch with {len(dataloader)} batches")
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Log batch information
            if batch_idx == 0:
                logger.info(f"Batch keys: {batch.keys()}")
                logger.info(f"Image shape: {batch['image'].shape}")
                logger.info(f"Label shape: {batch['label'].shape}")
                logger.info(f"Demographics shape: {batch['demographics'].shape}")
            
            try:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                demographics = batch['demographics'].to(self.device)
                
                # Log data statistics occasionally
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}: Images min/max: {images.min().item():.2f}/{images.max().item():.2f}")
                    logger.info(f"Batch {batch_idx}: Labels: {labels.tolist()}")
                    logger.info(f"Batch {batch_idx}: Demographics: {demographics.tolist()}")
                
                # Forward pass
                outputs = self.model(images)
                logger.debug(f"Batch {batch_idx}: Output shape: {outputs.shape}")
                
                # Compute loss
                loss, cls_loss, fair_loss = self.loss_fn(outputs, labels, demographics)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Log gradients occasionally
                if batch_idx % 20 == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            logger.debug(f"Batch {batch_idx}: Grad {name} - mean: {param.grad.mean().item():.6f}, std: {param.grad.std().item():.6f}")
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_fair_loss += fair_loss.item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'cls_loss': cls_loss.item(),
                    'fair_loss': fair_loss.item()
                })
                
                # Log batch results
                if batch_idx % 5 == 0:
                    logger.info(f"Batch {batch_idx}: loss={loss.item():.4f}, cls_loss={cls_loss.item():.4f}, fair_loss={fair_loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Update learning rate
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(f"Epoch completed in {epoch_time:.2f} seconds")
        logger.info(f"Average loss: {total_loss / len(dataloader):.4f}")
        logger.info(f"Average cls_loss: {total_cls_loss / len(dataloader):.4f}")
        logger.info(f"Average fair_loss: {total_fair_loss / len(dataloader):.4f}")
        
        return {
            'total_loss': total_loss / len(dataloader),
            'cls_loss': total_cls_loss / len(dataloader),
            'fair_loss': total_fair_loss / len(dataloader)
        }
        
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_demographics = []
        
        logger.info(f"Starting evaluation with {len(dataloader)} batches")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                images = batch['image'].to(self.device)
                labels = batch['label']
                demographics = batch['demographics']
                
                # Log batch info occasionally
                if batch_idx % 10 == 0:
                    logger.info(f"Eval batch {batch_idx}: Images shape: {images.shape}")
                
                outputs = self.model(images)
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_demographics.extend(demographics.numpy())
                
                # Log predictions occasionally
                if batch_idx % 10 == 0:
                    logger.info(f"Eval batch {batch_idx}: Predictions: {preds[:5]}")
                    logger.info(f"Eval batch {batch_idx}: Labels: {labels[:5].numpy()}")
            
            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Log evaluation results
        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Total samples evaluated: {len(all_preds)}")
        
        # Log class distribution
        if all_labels:
            unique, counts = np.unique(all_labels, return_counts=True)
            logger.info(f"Label distribution: {dict(zip(unique, counts))}")
        
        # Log demographic distribution
        if all_demographics:
            unique, counts = np.unique(all_demographics, return_counts=True)
            logger.info(f"Demographic distribution: {dict(zip(unique, counts))}")
        
        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'demographics': np.array(all_demographics)
        }