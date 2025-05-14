import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel
import os

# Set OpenMP environment variables for Intel CPU optimization
os.environ["OMP_NUM_THREADS"] = "8"  # Set to number of physical cores
os.environ["MKL_NUM_THREADS"] = "8"  # Set to number of physical cores
torch.set_num_threads(8)  # Set PyTorch to use 8 threads

class ContextAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, q, k, v):
        attn = torch.matmul(self.query(q), self.key(k).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, self.value(v)), attn

class FairDeepfakeDetector(nn.Module):
    def __init__(self, backbone='resnet', num_classes=2):
        super().__init__()
        self.backbone_type = backbone
        
        if backbone == 'inception':
            self.backbone = models.inception_v3(pretrained=True, aux_logits=False)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        elif backbone == 'resnet':
            # ResNet is more efficient on CPU with OpenMP
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        elif backbone == 'mobilenet':
            # MobileNet is very efficient on CPU
            self.backbone = models.mobilenet_v2(pretrained=True)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        else:  # Default to ViT
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
            feature_dim = 768
        
        # Temporal attention for video frames
        self.temporal_attention = nn.MultiheadAttention(feature_dim, 4, batch_first=True)
        
        # Context attention module
        self.context_attention = ContextAttention(feature_dim)
        
        # Fairness-aware classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, face_mask=None):
        batch_size = x.size(0)
        
        if x.dim() == 5:  # [batch, frames, channels, height, width]
            num_frames = x.size(1)
            # Reshape to process all frames
            x = x.view(-1, x.size(2), x.size(3), x.size(4))
            
            # Extract features for each frame
            if self.backbone_type in ['inception', 'resnet', 'mobilenet']:
                frame_features = self.backbone(x)
            else:  # ViT
                outputs = self.backbone(x)
                frame_features = outputs.last_hidden_state[:, 0]
                
            # Reshape back to [batch, frames, features]
            frame_features = frame_features.view(batch_size, num_frames, -1)
            
            # Apply temporal attention
            features, _ = self.temporal_attention(
                frame_features, frame_features, frame_features
            )
            
            # Pool across frames
            features = features.mean(dim=1)
        else:
            # Single image processing
            if self.backbone_type in ['inception', 'resnet', 'mobilenet']:
                features = self.backbone(x)
            else:  # ViT
                outputs = self.backbone(x)
                features = outputs.last_hidden_state[:, 0]
            
        if face_mask is not None:
            # Apply face-context attention
            context_features = features * (1 - face_mask)
            face_features = features * face_mask
            attended_features, _ = self.context_attention(
                face_features.unsqueeze(1),
                context_features.unsqueeze(1),
                context_features.unsqueeze(1)
            )
            features = features + attended_features.squeeze(1)
            
        logits = self.classifier(features)
        return logits