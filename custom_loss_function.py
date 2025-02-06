import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# CUSTOM LOSS: CROSS-ENTROPY + SMOOTHING
# ==========================================

class AgeMortalityLoss(nn.Module):
    """
    Combines standard cross-entropy with a "second-derivative" penalty on the predicted
    probabilities to enforce smoothness across consecutive ages.
    """
    def __init__(self, num_ages=96, alpha=0.1, label_smoothing=0.0, 
                 heavy_smoothing_region=None, heavy_smoothing_factor=1.0):
        """
        Args:
            num_ages (int): Number of discrete age classes (e.g. 96 if 25..120).
            alpha (float): Weight for the second-derivative smoothing penalty.
            label_smoothing (float): Amount of label smoothing around the true age.
            heavy_smoothing_region (tuple or None): e.g. (90, 95) if you want extra 
                smoothing in that region. Indices must match zero-based classes.
            heavy_smoothing_factor (float): Factor by which to increase smoothing 
                in the specified region.
        """
        super(AgeMortalityLoss, self).__init__()
        self.num_ages = num_ages
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.heavy_smoothing_region = heavy_smoothing_region
        self.heavy_smoothing_factor = heavy_smoothing_factor
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_ages) - raw network outputs (before softmax)
            targets: (batch_size,) - integer class labels [0..num_ages-1]
                     or a tensor if you apply your own smoothing. 
        Returns:
            A scalar tensor (the combined loss).
        """
        batch_size = logits.size(0)

        # ============================
        # 1) BUILD/USE SMOOTHED TARGET
        # ============================
        if targets.dtype == torch.long:
            # We'll convert int targets -> smoothed one-hot
            with torch.no_grad():
                soft_targets = self._make_smooth_targets(targets)
        else:
            # If "targets" is already float or double, assume it's a distribution
            soft_targets = targets
        
        # ============================
        # 2) CROSS ENTROPY (with soft targets)
        # ============================
        log_probs = F.log_softmax(logits, dim=-1)  
        ce_loss = - (soft_targets * log_probs).sum(dim=-1).mean()  

        # ============================
        # 3) SECOND-DERIVATIVE SMOOTHING
        # ============================
        probs = F.softmax(logits, dim=-1)  # shape (batch_size, num_ages)
        
        # We'll slice out neighbors p_{i-1}, p_i, p_{i+1}
        # second derivative = p_{i-1} - 2*p_i + p_{i+1}
        if self.num_ages > 2:
            p_left   = probs[:, :-2]
            p_center = probs[:, 1:-1]
            p_right  = probs[:, 2:]
            laplacian = p_left - 2 * p_center + p_right  # (batch_size, num_ages-2)

            # weight certain age indices more (like 90..95).
            if self.heavy_smoothing_region is not None:
                age_indices = torch.arange(1, self.num_ages-1, device=logits.device)
                # boolean mask for region
                start_idx, end_idx = self.heavy_smoothing_region
                region_mask = (age_indices >= start_idx) & (age_indices <= end_idx)
                # shape (num_ages-2) -> broadcast to (batch_size, num_ages-2)
                region_mask = region_mask.unsqueeze(0).expand_as(laplacian)
                
                # apply heavier factor only in that region
                weighted_laplacian = torch.where(region_mask, 
                                                 laplacian.pow(2) * self.heavy_smoothing_factor,
                                                 laplacian.pow(2))
                smoothing_loss = weighted_laplacian.mean()
            else:
                smoothing_loss = laplacian.pow(2).mean()
        else:
            # If num_ages <= 2, no second derivative is possible in a normal sense
            smoothing_loss = torch.tensor(0.0, device=logits.device)

        # Combine them
        total_loss = ce_loss + self.alpha * smoothing_loss

        return total_loss
    
    def _make_smooth_targets(self, targets):
        """
        Convert integer targets [0..num_ages-1] -> smoothed one-hot distribution
        according to self.label_smoothing.
        """
        batch_size = targets.size(0)
        smooth_targets = torch.zeros(batch_size, self.num_ages, device=targets.device)
        for i in range(batch_size):
            age = targets[i].item()
            # main peak
            smooth_targets[i, age] = 1.0 - self.label_smoothing
            # spread to neighbors
            if self.label_smoothing > 0.0:
                # distribute smoothing among immediate neighbors
                neighbor_val = self.label_smoothing / 2
                if age > 0:
                    smooth_targets[i, age - 1] += neighbor_val
                if age < self.num_ages - 1:
                    smooth_targets[i, age + 1] += neighbor_val
        return smooth_targets