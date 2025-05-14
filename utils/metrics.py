import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

class FairnessMetrics:
    @staticmethod
    def compute_group_metrics(predictions, labels, demographics):
        unique_groups = np.unique(demographics)
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = {
            'auc': roc_auc_score(labels, predictions),
            'accuracy': accuracy_score(labels, np.round(predictions)),
            'precision_recall': precision_recall_fscore_support(
                labels, np.round(predictions), average='binary'
            )
        }
        
        # Per-group metrics
        for group in unique_groups:
            group_mask = demographics == group
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            metrics[f'group_{group}'] = {
                'auc': roc_auc_score(group_labels, group_preds),
                'accuracy': accuracy_score(group_labels, np.round(group_preds)),
                'precision_recall': precision_recall_fscore_support(
                    group_labels, np.round(group_preds), average='binary'
                )
            }
        
        # Fairness metrics
        metrics['fairness'] = {
            'demographic_parity': FairnessMetrics._compute_demographic_parity(metrics),
            'equal_opportunity': FairnessMetrics._compute_equal_opportunity(
                predictions, labels, demographics
            ),
            'equalized_odds': FairnessMetrics._compute_equalized_odds(
                predictions, labels, demographics
            )
        }
        
        return metrics

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