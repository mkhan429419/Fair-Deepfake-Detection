import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score

class FairnessMetrics:
    @staticmethod
    def compute_group_metrics(predictions, labels, demographics):
        """Compute metrics for each demographic group"""
        group_metrics = {}
        
        # Convert inputs to numpy arrays if they are tensors
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()
        if hasattr(demographics, 'cpu'):
            demographics = demographics.cpu().numpy()
        
        # Overall metrics
        accuracy = accuracy_score(labels, np.round(predictions))
        
        # Handle case where only one class is present
        try:
            auc = roc_auc_score(labels, predictions)
        except ValueError:
            # If only one class is present, AUC is undefined
            auc = float('nan')  # or some default value like 0.5
        
        precision = precision_score(labels, np.round(predictions), zero_division=0)
        recall = recall_score(labels, np.round(predictions), zero_division=0)
        f1 = f1_score(labels, np.round(predictions), zero_division=0)
        
        group_metrics['overall'] = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': len(labels)
        }
        
        # Group-specific metrics
        unique_groups = np.unique(demographics)
        group_accuracies = []
        group_aucs = []
        
        for group in unique_groups:
            group_indices = demographics == group
            group_preds = predictions[group_indices]
            group_labels = labels[group_indices]
            
            if len(group_preds) == 0:
                continue
            
            group_accuracy = accuracy_score(group_labels, np.round(group_preds))
            group_accuracies.append(group_accuracy)
            
            # Handle case where only one class is present in the group
            try:
                group_auc = roc_auc_score(group_labels, group_preds)
                group_aucs.append(group_auc)
            except ValueError:
                # If only one class is present, AUC is undefined
                group_auc = float('nan')  # or some default value like 0.5
            
            group_precision = precision_score(group_labels, np.round(group_preds), zero_division=0)
            group_recall = recall_score(group_labels, np.round(group_preds), zero_division=0)
            group_f1 = f1_score(group_labels, np.round(group_preds), zero_division=0)
            
            group_metrics[f'group_{group}'] = {
                'accuracy': group_accuracy,
                'auc': group_auc,
                'precision': group_precision,
                'recall': group_recall,
                'f1': group_f1,
                'count': len(group_labels)
            }
        
        # Compute fairness metrics
        fairness_metrics = {}
        
        # Demographic parity (difference in accuracy between groups)
        if len(group_accuracies) >= 2:
            fairness_metrics['demographic_parity'] = max(group_accuracies) - min(group_accuracies)
        else:
            fairness_metrics['demographic_parity'] = 0.0
        
        # Equal opportunity (difference in TPR between groups)
        try:
            equal_opportunity = FairnessMetrics._compute_equal_opportunity(predictions, labels, demographics)
            fairness_metrics['equal_opportunity'] = equal_opportunity
        except Exception as e:
            fairness_metrics['equal_opportunity'] = 0.0
        
        # Equalized odds (difference in FPR between groups)
        try:
            equalized_odds = FairnessMetrics._compute_equalized_odds(predictions, labels, demographics)
            fairness_metrics['equalized_odds'] = equalized_odds
        except Exception as e:
            fairness_metrics['equalized_odds'] = 0.0
        
        # AUC disparity
        if len(group_aucs) >= 2:
            non_nan_aucs = [auc for auc in group_aucs if not np.isnan(auc)]
            if len(non_nan_aucs) >= 2:
                fairness_metrics['max_auc_disparity'] = max(non_nan_aucs) - min(non_nan_aucs)
            else:
                fairness_metrics['max_auc_disparity'] = 0.0
        else:
            fairness_metrics['max_auc_disparity'] = 0.0
        
        # Add fairness metrics to the main metrics dictionary
        group_metrics['fairness'] = fairness_metrics
        
        return group_metrics

    @staticmethod
    def _compute_demographic_parity(metrics):
        group_accuracies = [
            metrics[k]['accuracy'] 
            for k in metrics.keys() 
            if k.startswith('group_')
        ]
        return max(group_accuracies) - min(group_accuracies)

    @staticmethod
    def _compute_equal_opportunity(predictions, labels, demographics):
        unique_groups = np.unique(demographics)
        tpr_values = []
        
        for group in unique_groups:
            mask = (demographics == group) & (labels == 1)
            tpr = np.mean(predictions[mask] > 0.5)
            tpr_values.append(tpr)
            
        return max(tpr_values) - min(tpr_values)

    @staticmethod
    def _compute_equalized_odds(predictions, labels, demographics):
        unique_groups = np.unique(demographics)
        fpr_values = []
        
        for group in unique_groups:
            mask = (demographics == group) & (labels == 0)
            fpr = np.mean(predictions[mask] > 0.5)
            fpr_values.append(fpr)
            
        return max(fpr_values) - min(fpr_values)