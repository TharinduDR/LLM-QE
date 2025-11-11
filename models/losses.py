import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss with margin.
    Learns relative ordering: if z_i > z_j, then pred_i should be > pred_j.

    Used in COMET and other SOTA QE systems.
    """

    def __init__(
            self,
            margin: float = 0.5,
            sample_ratio: float = 1.0,
            hard_negative_mining: bool = False
    ):
        """
        Args:
            margin: Minimum difference between better and worse predictions
            sample_ratio: Ratio of pairs to sample (1.0 = all pairs)
            hard_negative_mining: Focus on hard pairs (close quality scores)
        """
        super().__init__()
        self.margin = margin
        self.sample_ratio = sample_ratio
        self.hard_negative_mining = hard_negative_mining

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute pairwise ranking loss.

        Args:
            predictions: [batch_size] model predictions
            targets: [batch_size] ground truth z-scores

        Returns:
            loss: scalar loss
            stats: dict with statistics
        """
        batch_size = predictions.size(0)

        if predictions.dim() == 2:
            predictions = predictions.squeeze(-1)

        # Create all pairwise comparisons
        # pred_i: [N, 1], pred_j: [1, N] -> diff: [N, N]
        pred_i = predictions.unsqueeze(1)  # [N, 1]
        pred_j = predictions.unsqueeze(0)  # [1, N]
        pred_diff = pred_i - pred_j  # [N, N]

        target_i = targets.unsqueeze(1)  # [N, 1]
        target_j = targets.unsqueeze(0)  # [1, N]
        target_diff = target_i - target_j  # [N, N]

        # Create ranking indicators
        # ranking_labels = 1 if target_i > target_j, else 0
        ranking_labels = (target_diff > 0).float()

        # Mask for valid pairs (not comparing with self, and significant difference)
        not_self = (torch.eye(batch_size, device=predictions.device) == 0).float()
        significant_diff = (torch.abs(target_diff) > 0.1).float()  # Filter near-ties
        valid_pairs = not_self * significant_diff

        # Compute hinge loss
        # If target_i > target_j, we want pred_i > pred_j + margin
        # Loss = max(0, margin - (pred_i - pred_j))
        hinge_loss = torch.clamp(self.margin - pred_diff, min=0.0)

        # Apply to pairs where target_i > target_j
        pairwise_loss = hinge_loss * ranking_labels * valid_pairs

        # Hard negative mining: focus on pairs where model is wrong
        if self.hard_negative_mining:
            # Hard negatives: model predicts wrong order
            wrong_order = (pred_diff < 0) & (target_diff > 0)
            hard_weights = wrong_order.float() * 2.0 + 1.0  # 3x weight for hard pairs
            pairwise_loss = pairwise_loss * hard_weights

        # Sample pairs if needed (for very large batches)
        if self.sample_ratio < 1.0:
            num_pairs = valid_pairs.sum().item()
            num_sample = int(num_pairs * self.sample_ratio)
            if num_sample > 0:
                # Sample random pairs
                flat_indices = torch.where(valid_pairs.flatten() > 0)[0]
                sampled_indices = flat_indices[torch.randperm(len(flat_indices))[:num_sample]]
                mask = torch.zeros_like(valid_pairs.flatten())
                mask[sampled_indices] = 1.0
                valid_pairs = mask.reshape(valid_pairs.shape)

        # Average over valid pairs
        num_valid_pairs = valid_pairs.sum() + 1e-8
        loss = pairwise_loss.sum() / num_valid_pairs

        # Compute statistics
        with torch.no_grad():
            correct_order = ((pred_diff > 0) & (target_diff > 0)).float()
            accuracy = (correct_order * valid_pairs).sum() / num_valid_pairs

        stats = {
            'pairwise_loss': loss.item(),
            'pairwise_accuracy': accuracy.item(),
            'num_pairs': num_valid_pairs.item()
        }

        return loss, stats


class TripletRankingLoss(nn.Module):
    """
    Triplet loss for ranking: anchor, positive (better), negative (worse).
    More structured than pairwise loss.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Form triplets: (anchor, positive, negative)
        positive has higher quality than anchor
        negative has lower quality than anchor
        """
        if predictions.dim() == 2:
            predictions = predictions.squeeze(-1)

        batch_size = predictions.size(0)

        # For each sample, find positive and negative
        target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0)  # [N, N]

        losses = []
        num_triplets = 0

        for i in range(batch_size):
            # Find positives (higher quality)
            positives = torch.where(target_diffs[i] < -0.1)[0]  # target_j > target_i
            # Find negatives (lower quality)
            negatives = torch.where(target_diffs[i] > 0.1)[0]  # target_j < target_i

            if len(positives) > 0 and len(negatives) > 0:
                # Sample one positive and one negative
                pos_idx = positives[torch.randint(len(positives), (1,))]
                neg_idx = negatives[torch.randint(len(negatives), (1,))]

                # Triplet loss: pred[pos] should be > pred[anchor] > pred[neg]
                # We want: pred[pos] - pred[neg] > margin
                triplet_loss = torch.clamp(
                    self.margin - (predictions[pos_idx] - predictions[neg_idx]),
                    min=0.0
                )

                losses.append(triplet_loss)
                num_triplets += 1

        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, device=predictions.device)

        stats = {
            'triplet_loss': loss.item(),
            'num_triplets': num_triplets
        }

        return loss, stats


class ListwiseRankingLoss(nn.Module):
    """
    ListNet-style ranking loss.
    Optimizes the entire ranking distribution.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Convert predictions and targets to probability distributions
        and minimize KL divergence.
        """
        if predictions.dim() == 2:
            predictions = predictions.squeeze(-1)

        # Convert to probability distributions using softmax
        pred_probs = F.softmax(predictions / self.temperature, dim=0)
        target_probs = F.softmax(targets / self.temperature, dim=0)

        # KL divergence
        loss = F.kl_div(
            pred_probs.log(),
            target_probs,
            reduction='batchmean'
        )

        stats = {
            'listwise_loss': loss.item()
        }

        return loss, stats


class CombinedQELoss(nn.Module):
    """
    Combined loss: MSE + Pairwise Ranking
    Balances calibration and ranking.
    """

    def __init__(
            self,
            mse_weight: float = 0.3,
            ranking_weight: float = 0.7,
            margin: float = 0.5,
            use_triplet: bool = False,
            use_listwise: bool = False,
            hard_negative_mining: bool = False
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight

        self.mse_loss = nn.MSELoss()

        if use_triplet:
            self.ranking_loss = TripletRankingLoss(margin=margin)
        elif use_listwise:
            self.ranking_loss = ListwiseRankingLoss()
        else:
            self.ranking_loss = PairwiseRankingLoss(
                margin=margin,
                hard_negative_mining=hard_negative_mining
            )

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        """
        if predictions.dim() == 2:
            predictions_flat = predictions.squeeze(-1)
        else:
            predictions_flat = predictions

        # MSE for calibration
        mse = self.mse_loss(predictions_flat, targets)

        # Ranking loss for correlation
        ranking_loss, ranking_stats = self.ranking_loss(predictions_flat, targets)

        # Combine
        total_loss = self.mse_weight * mse + self.ranking_weight * ranking_loss

        stats = {
            'mse': mse.item(),
            'total_loss': total_loss.item(),
            **ranking_stats
        }

        return total_loss, stats