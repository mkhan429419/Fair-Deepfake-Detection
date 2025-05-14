import torch
import torch.nn as nn
import torch.nn.functional as F

class DAWFDDLoss(nn.Module):
    def __init__(self, lambda_fair=1.0):
        super().__init__()
        self.lambda_fair = lambda_fair
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, labels, demographics):
        # Standard classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Compute group-wise losses
        unique_groups = torch.unique(demographics)
        group_losses = []
        
        for group in unique_groups:
            group_mask = (demographics == group)
            if not group_mask.any():
                continue
                
            group_probs = F.softmax(logits[group_mask], dim=1)
            group_labels = labels[group_mask]
            group_loss = F.cross_entropy(group_probs, group_labels)
            group_losses.append(group_loss)
            
        # Compute fairness loss
        if len(group_losses) > 1:
            group_losses = torch.stack(group_losses)
            fairness_loss = torch.var(group_losses)
        else:
            fairness_loss = torch.tensor(0.0).to(logits.device)
            
        total_loss = cls_loss.mean() + self.lambda_fair * fairness_loss
        return total_loss, cls_loss.mean(), fairness_loss

class FairAlignLoss(nn.Module):
    def __init__(self, lambda_align=1.0):
        super().__init__()
        self.lambda_align = lambda_align
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, features, logits, labels, demographics):
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Compute alignment loss
        unique_groups = torch.unique(demographics)
        align_loss = torch.tensor(0.0).to(features.device)
        
        if len(unique_groups) > 1:
            # Compute pairwise MMD between group distributions
            for i in range(len(unique_groups)):
                for j in range(i + 1, len(unique_groups)):
                    group1_mask = (demographics == unique_groups[i])
                    group2_mask = (demographics == unique_groups[j])
                    
                    group1_features = features[group1_mask]
                    group2_features = features[group2_mask]
                    
                    align_loss += self._mmd_loss(group1_features, group2_features)
                    
        total_loss = cls_loss + self.lambda_align * align_loss
        return total_loss, cls_loss, align_loss
        
    def _mmd_loss(self, x, y):
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        
        rx = (xx.diagonal().unsqueeze(0).expand_as(xx))
        ry = (yy.diagonal().unsqueeze(0).expand_as(yy))
        
        K = torch.exp(- 0.5 * (rx.t() + rx - 2*xx))
        L = torch.exp(- 0.5 * (ry.t() + ry - 2*yy))
        P = torch.exp(- 0.5 * (rx.t() + ry - 2*xy))
        
        beta = (1./(x.size(0)*x.size(0)))
        gamma = (1./(y.size(0)*y.size(0)))
        alpha = (1./(x.size(0)*y.size(0)))
        
        return beta * K.sum() + gamma * L.sum() - 2. * alpha * P.sum()