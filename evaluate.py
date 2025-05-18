import torch
import yaml
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
from tabulate import tabulate
import argparse

# Set OpenMP environment variables
os.environ["OMP_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["MKL_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # Intel CPU optimization

# Import your models
from models.fair_deepfake_detector import FairDeepfakeDetector
from utils.metrics import FairnessMetrics
from data.dataset import DeepfakeDataset  # Import the same dataset class used in training
from utils.trainer import FairTrainer  # Import the trainer for consistent evaluation

class SimpleDeepfakeDataset(Dataset):
    """Simplified dataset that directly loads videos from real/fake folders"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Find videos in real folder - label 0
        real_dir = os.path.join(data_path, 'real')
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
                    self.samples.append({
                        'path': os.path.join(real_dir, filename),
                        'label': 0,
                        'demographics': 0  # Default demographic
                    })
            print(f"Found {len(self.samples)} real videos")
                    
        # Find videos in fake folder - label 1
        fake_dir = os.path.join(data_path, 'fake')
        fake_count = 0
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.endswith(('.mp4', '.avi', '.mov', '.webm')):
                    self.samples.append({
                        'path': os.path.join(fake_dir, filename),
                        'label': 1,
                        'demographics': 0  # Default demographic
                    })
                    fake_count += 1
            print(f"Found {fake_count} fake videos")
        
        print(f"Total samples: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self._extract_frames(sample['path'], 6)  # Extract 6 frames
        
        if frames is None or len(frames) == 0:
            # Create blank image if no frames
            frame_tensor = torch.zeros((6, 3, 224, 224))
        else:
            # Process frames
            processed_frames = []
            for frame in frames:
                frame = cv2.resize(frame, (224, 224))
                # Convert to PyTorch tensor and normalize
                frame = frame / 255.0
                frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
                processed_frames.append(frame)
                
            # Ensure we have exactly 6 frames
            if len(processed_frames) < 6:
                last_frame = processed_frames[-1] if processed_frames else np.zeros((3, 224, 224))
                processed_frames.extend([last_frame] * (6 - len(processed_frames)))
                
            # Convert to tensor
            frame_tensor = torch.FloatTensor(np.array(processed_frames))
            
        return {
            'image': frame_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'demographics': torch.tensor(sample['demographics'], dtype=torch.long)
        }
    
    def _extract_frames(self, video_path, num_frames=6):
        """Extract evenly spaced frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return None
                
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count <= 0:
                print(f"Invalid frame count for video: {video_path}")
                return None
                
            # Calculate frame indices to extract
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=np.int32)
            
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {str(e)}")
            return None

def evaluate(config_path, model_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set PyTorch thread settings
    if config['training'].get('use_openmp', False):
        num_threads = config['training'].get('num_threads', 8)
        torch.set_num_threads(num_threads)
        print(f"Using OpenMP with {num_threads} threads")
    
    # Print directory information
    test_path = config['data'].get('test_path', 'data/test')
    print(f"Test path: {test_path}")
    
    # Check if test directory exists
    if not os.path.exists(test_path):
        print(f"ERROR: Test directory {test_path} does not exist!")
        return
    
    # Create dataset using the same DeepfakeDataset class as in training
    print("Creating test dataset...")
    test_dataset = DeepfakeDataset(
        test_path,
        mode='test',
        annotations_path=config['data'].get('annotations_path'),
        preprocess=True,
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    if len(test_dataset) == 0:
        print("ERROR: No samples found for evaluation. Please check your data paths.")
        return
    
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
    try:
        model.load_state_dict(torch.load(model_path, map_location=config['training']['device']))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
        
    model.to(config['training']['device'])
    model.eval()
    
    # Initialize the trainer for consistent evaluation
    from models.losses import DAWFDDLoss
    loss_fn = DAWFDDLoss(lambda_fair=config['training']['lambda_fair'])
    trainer = FairTrainer(
        model=model,
        loss_fn=loss_fn,
        device=config['training']['device'],
        lr=config['training']['learning_rate']
    )
    
    # Use the trainer's evaluate method for consistency with training
    print("Evaluating model...")
    val_results = trainer.evaluate(test_loader)
    
    # Compute metrics using the same method as in training
    metrics = FairnessMetrics.compute_group_metrics(
        val_results['predictions'],
        val_results['labels'],
        val_results['demographics']
    )
    
    # Print results
    print("\nOverall Metrics:")
    print(f"AUC: {metrics['overall']['auc']:.4f}")
    print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    print(f"F1: {metrics['overall']['f1']:.4f}")
    
    print("\nGroup Metrics:")
    for group, group_metrics in metrics.items():
        if group.startswith('group_'):
            print(f"{group}:")
            print(f"  AUC: {group_metrics['auc']:.4f}")
            print(f"  Accuracy: {group_metrics['accuracy']:.4f}")
            print(f"  Count: {group_metrics['count']}")
    
    # Fix the fairness metrics keys
    print("\nFairness Metrics:")
    fairness = metrics.get('fairness', {})
    print(f"Demographic Parity: {fairness.get('demographic_parity', float('nan')):.4f}")
    print(f"Max AUC Disparity: {fairness.get('max_auc_disparity', float('nan')):.4f}")
    print(f"Equal Opportunity: {fairness.get('equal_opportunity', float('nan')):.4f}")
    print(f"Equalized Odds: {fairness.get('equalized_odds', float('nan')):.4f}")
    
    # Print individual file predictions
    print("\nIndividual Predictions:")
    print("------------------------------------------------------------")
    print(f"{'Filename':<30} | {'True Label':<10} | {'Prediction':<10} | Probability")
    print("------------------------------------------------------------")
    
    for i, sample in enumerate(test_dataset.samples):
        if i < len(val_results['predictions']):
            # Get filename - handle different dataset structures
            if 'path' in sample:
                filename = os.path.basename(sample['path'])
            elif 'video_path' in sample:
                filename = os.path.basename(sample['video_path'])
            else:
                filename = f"sample_{i}"  # Fallback if no path is available
            
            label = "FAKE" if sample['label'] == 1 else "REAL"
            pred = "FAKE" if val_results['predictions'][i] > 0.5 else "REAL"
            print(f"{filename:<30} | {label:<10} | {pred:<10} | {val_results['predictions'][i]:.4f}")

def evaluate_all_models(config_path):
    """Evaluate all models in the checkpoints directory"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the checkpoint directory
    checkpoint_dir = config['training'].get('save_dir', 'checkpoints')
    print(f"Looking for models in: {checkpoint_dir}")
    
    # Find all .pth files in the checkpoint directory
    model_paths = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not model_paths:
        print(f"No model files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(model_paths)} model files to evaluate")
    
    # Create dataset using the same DeepfakeDataset class as in training
    test_path = config['data'].get('test_path', 'data/test')
    print("Creating test dataset...")
    test_dataset = DeepfakeDataset(
        test_path,
        mode='test',
        annotations_path=config['data'].get('annotations_path'),
        preprocess=True,
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    if len(test_dataset) == 0:
        print("ERROR: No samples found for evaluation. Please check your data paths.")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize the loss function for the trainer
    from models.losses import DAWFDDLoss
    loss_fn = DAWFDDLoss(lambda_fair=config['training']['lambda_fair'])
    
    # Results to store
    all_results = []
    
    # Evaluate each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*80}")
        
        # Initialize model - use the same backbone as in training
        model = FairDeepfakeDetector(
            backbone=config['model']['backbone'],
            num_classes=config['model']['num_classes']
        )
        
        # Load trained weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=config['training']['device']))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            continue
            
        model.to(config['training']['device'])
        model.eval()
        
        # Initialize trainer for consistent evaluation
        trainer = FairTrainer(
            model=model,
            loss_fn=loss_fn,
            device=config['training']['device'],
            lr=config['training']['learning_rate']
        )
        
        # Use the trainer's evaluate method
        print("Evaluating model...")
        val_results = trainer.evaluate(test_loader)
        
        # Compute metrics using the same method as in training
        metrics = FairnessMetrics.compute_group_metrics(
            val_results['predictions'],
            val_results['labels'],
            val_results['demographics']
        )
        
        # Store results
        model_results = {
            'model_name': model_name,
            'auc': metrics['overall']['auc'],
            'accuracy': metrics['overall']['accuracy'],
            'precision': metrics['overall']['precision'],
            'recall': metrics['overall']['recall'],
            'f1': metrics['overall']['f1'],
            'demographic_parity': metrics.get('fairness', {}).get('demographic_parity', float('nan')),
            'max_auc_disparity': metrics.get('fairness', {}).get('max_auc_disparity', float('nan'))
        }
        all_results.append(model_results)
        
        # Print results for this model
        print("\nOverall Metrics:")
        print(f"AUC: {metrics['overall']['auc']:.4f}")
        print(f"Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Precision: {metrics['overall']['precision']:.4f}")
        print(f"Recall: {metrics['overall']['recall']:.4f}")
        print(f"F1: {metrics['overall']['f1']:.4f}")
        
        # Print individual file predictions for this model
        print("\nIndividual Predictions:")
        print("------------------------------------------------------------")
        print(f"{'Filename':<30} | {'True Label':<10} | {'Prediction':<10} | Probability")
        print("------------------------------------------------------------")
        
        for i, sample in enumerate(test_dataset.samples):
            if i < len(val_results['predictions']):
                # Get filename - handle different dataset structures
                if 'path' in sample:
                    filename = os.path.basename(sample['path'])
                elif 'video_path' in sample:
                    filename = os.path.basename(sample['video_path'])
                else:
                    filename = f"sample_{i}"  # Fallback if no path is available
            
                label = "FAKE" if sample['label'] == 1 else "REAL"
                pred = "FAKE" if val_results['predictions'][i] > 0.5 else "REAL"
                print(f"{filename:<30} | {label:<10} | {pred:<10} | {val_results['predictions'][i]:.4f}")
    
    # Compare all models
    print("\n\nModel Comparison:")
    
    # Sort results by AUC
    all_results = sorted(all_results, key=lambda x: x['auc'], reverse=True)
    
    # Create table
    table_headers = ["Model", "AUC", "Accuracy", "F1", "Demo. Parity", "AUC Disparity"]
    table_data = []
    
    for result in all_results:
        table_data.append([
            result['model_name'],
            f"{result['auc']:.4f}",
            f"{result['accuracy']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['demographic_parity']:.4f}",
            f"{result['max_auc_disparity']:.4f}"
        ])
    
    # Print table
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    
    # Identify best model
    best_model = all_results[0]['model_name']
    print(f"\nBest Model (by AUC): {best_model} with AUC: {all_results[0]['auc']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, help='Path to specific model file (if not provided, evaluates all models)')
    args = parser.parse_args()
    
    if args.model:
        # Evaluate a specific model
        evaluate(args.config, args.model)
    else:
        # Evaluate all models in the checkpoint directory
        evaluate_all_models(args.config)