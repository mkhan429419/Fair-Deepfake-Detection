import torch
import yaml
import numpy as np
import os
from torch.utils.data import DataLoader

# Set OpenMP environment variables
os.environ["OMP_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["MKL_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # Intel CPU optimization

from data.dataset import DeepfakeDataset
from models.fair_deepfake_detector import FairDeepfakeDetector
from utils.metrics import FairnessMetrics

def evaluate(config_path, model_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set PyTorch thread settings
    if config['training'].get('use_openmp', False):
        num_threads = config['training'].get('num_threads', 8)
        torch.set_num_threads(num_threads)
        print(f"Using OpenMP with {num_threads} threads")
    
    # Create test dataset and loader
    test_dataset = DeepfakeDataset(
        config['data']['test_path'],
        mode='test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    model = FairDeepfakeDetector(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes']
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.to(config['training']['device'])
    model.eval()
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_demographics = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(config['training']['device'])
            labels = batch['label']
            demographics = batch['demographics']
            
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_demographics.extend(demographics.numpy())
            
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_demographics = np.array(all_demographics)
    
    # Compute metrics
    metrics = FairnessMetrics.compute_group_metrics(
        all_preds,
        all_labels,
        all_demographics
    )
    
    # Print results
    print("Overall Metrics:")
    print(f"AUC: {metrics['overall']['auc']:.4f}")
    print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
    print("\nGroup Metrics:")
    for group, group_metrics in metrics.items():
        if group.startswith('group_'):
            print(f"{group}:")
            print(f"  AUC: {group_metrics['auc']:.4f}")
            print(f"  Accuracy: {group_metrics['accuracy']:.4f}")
    print("\nFairness Metrics:")
    print(f"Max Accuracy Disparity: {metrics['fairness']['max_accuracy_disparity']:.4f}")
    print(f"Max AUC Disparity: {metrics['fairness']['max_auc_disparity']:.4f}")

if __name__ == "__main__":
    evaluate('config/config.yaml', 'checkpoints/best_model.pth')