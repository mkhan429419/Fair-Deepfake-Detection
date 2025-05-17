import torch
import yaml
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set OpenMP environment variables
os.environ["OMP_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["MKL_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # Intel CPU optimization

# Import your models
from models.fair_deepfake_detector import FairDeepfakeDetector
from utils.metrics import FairnessMetrics

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
        
    # Print subdirectory information
    subdirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    print(f"Found subdirectories: {subdirs}")
    
    real_path = os.path.join(test_path, 'real')
    fake_path = os.path.join(test_path, 'fake')
    
    # Check real directory
    if os.path.exists(real_path):
        real_videos = [f for f in os.listdir(real_path) if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
        print(f"Found {len(real_videos)} videos in real directory")
    else:
        print("Real directory does not exist!")
        
    # Check fake directory
    if os.path.exists(fake_path):
        fake_videos = [f for f in os.listdir(fake_path) if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
        print(f"Found {len(fake_videos)} videos in fake directory")
    else:
        print("Fake directory does not exist!")
    
    # Create dataset and dataloader
    test_dataset = SimpleDeepfakeDataset(test_path)
    
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
    
    # Evaluation
    all_preds = []
    all_labels = []
    all_demographics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(config['training']['device'])
            labels = batch['label']
            demographics = batch['demographics']
            
            # Print shapes for debugging
            if batch_idx == 0:
                print(f"Input shape: {images.shape}")
            
            # Reshape if needed - handle both [B, F, C, H, W] and [B, C, H, W] formats
            if images.dim() == 5:  # [B, F, C, H, W]
                outputs = model(images)
            else:  # [B, C, H, W]
                outputs = model(images.unsqueeze(1))
                
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_demographics.extend(demographics.numpy())
            
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_demographics = np.array(all_demographics)
    
    print(f"Evaluation complete on {len(all_preds)} samples")
    print(f"Labels distribution: {np.bincount(all_labels)}")
    
    # Compute metrics
    metrics = FairnessMetrics.compute_group_metrics(
        all_preds,
        all_labels,
        all_demographics
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
        if i < len(all_preds):
            filename = os.path.basename(sample['path'])
            label = "FAKE" if sample['label'] == 1 else "REAL"
            pred = "FAKE" if all_preds[i] > 0.5 else "REAL"
            print(f"{filename:<30} | {label:<10} | {pred:<10} | {all_preds[i]:.4f}")

if __name__ == "__main__":
    evaluate('config/config.yaml', 'checkpoints/best_model.pth')