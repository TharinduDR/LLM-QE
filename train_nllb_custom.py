#!/usr/bin/env python3
"""
Train NLLB-QE with pairwise contrastive/ranking loss.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from functools import partial
from scipy.stats import pearsonr, spearmanr
import numpy as np

from models.nllb_qe_model import NLLBQEModel
from models.losses import (
    PairwiseRankingLoss,
    TripletRankingLoss,
    ListwiseRankingLoss,
    CombinedQELoss
)
from utils.custom_data_utils import CustomQEDataset, collate_custom_nllb


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            src_input = {k: v.to(device) for k, v in batch['source'].items()}
            tgt_input = {k: v.to(device) for k, v in batch['target'].items()}
            z_means = batch['z_means'].to(device)

            model_scores = batch.get('model_scores')
            if model_scores is not None:
                model_scores = model_scores.to(device)

            outputs = model(
                src_input_ids=src_input['input_ids'],
                src_attention_mask=src_input['attention_mask'],
                tgt_input_ids=tgt_input['input_ids'],
                tgt_attention_mask=tgt_input['attention_mask'],
                model_scores=model_scores
            )

            preds = outputs['z_score_pred'].squeeze(-1)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(z_means.cpu().numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Compute metrics
    results = {
        'pearson': pearsonr(predictions, targets)[0],
        'spearman': spearmanr(predictions, targets)[0],
        'mae': np.abs(predictions - targets).mean(),
        'rmse': np.sqrt(((predictions - targets) ** 2).mean())
    }

    return results


def train(
        model: NLLBQEModel,
        train_dataset: CustomQEDataset,
        val_dataset: CustomQEDataset,
        src_lang: str,
        tgt_lang: str,
        output_dir: str,
        batch_size: int = 16,
        num_epochs: int = 5,
        learning_rate: float = 1e-5,
        warmup_ratio: float = 0.1,
        device: str = "cuda",
        log_wandb: bool = False,
        gradient_accumulation_steps: int = 1,
        loss_type: str = "pairwise",
        margin: float = 0.5,
        mse_weight: float = 0.3,
        ranking_weight: float = 0.7,
        hard_negative_mining: bool = False
):
    """
    Train with pairwise contrastive loss.
    """
    model = model.to(device)

    # Data loaders
    collate_fn = partial(
        collate_custom_nllb,
        tokenizer=model.tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Loss function
    if loss_type == "pairwise":
        criterion = PairwiseRankingLoss(
            margin=margin,
            hard_negative_mining=hard_negative_mining
        )
        print(f"Using Pairwise Ranking Loss (margin={margin}, hard_mining={hard_negative_mining})")
    elif loss_type == "triplet":
        criterion = TripletRankingLoss(margin=margin)
        print(f"Using Triplet Ranking Loss (margin={margin})")
    elif loss_type == "listwise":
        criterion = ListwiseRankingLoss()
        print("Using Listwise Ranking Loss")
    elif loss_type == "combined":
        criterion = CombinedQELoss(
            mse_weight=mse_weight,
            ranking_weight=ranking_weight,
            margin=margin,
            hard_negative_mining=hard_negative_mining
        )
        print(f"Using Combined Loss (MSE {mse_weight:.1f} + Ranking {ranking_weight:.1f})")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_pearson = -1.0
    global_step = 0

    print(f"\n{'=' * 70}")
    print(f"Starting Training with Pairwise Contrastive Loss")
    print(f"{'=' * 70}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Pairs per batch: ~{batch_size * (batch_size - 1) // 2}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"{'=' * 70}\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_stats = {
            'pairwise_loss': 0.0,
            'pairwise_accuracy': 0.0,
            'mse': 0.0
        }
        num_batches = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src_input = {k: v.to(device) for k, v in batch['source'].items()}
            tgt_input = {k: v.to(device) for k, v in batch['target'].items()}
            z_means = batch['z_means'].to(device)

            model_scores = batch.get('model_scores')
            if model_scores is not None:
                model_scores = model_scores.to(device)

            # Forward pass
            outputs = model(
                src_input_ids=src_input['input_ids'],
                src_attention_mask=src_input['attention_mask'],
                tgt_input_ids=tgt_input['input_ids'],
                tgt_attention_mask=tgt_input['attention_mask'],
                model_scores=model_scores
            )

            z_score_pred = outputs['z_score_pred'].squeeze(-1)

            # Compute loss
            loss, stats = criterion(z_score_pred, z_means)
            loss = loss / gradient_accumulation_steps

            # Backward
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            train_loss += loss.item() * gradient_accumulation_steps

            # Accumulate stats
            for key in stats:
                if key in train_stats:
                    train_stats[key] += stats[key]
            num_batches += 1

            # Update progress bar
            postfix = {
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            }
            if 'pairwise_accuracy' in stats:
                postfix['pair_acc'] = f"{stats['pairwise_accuracy']:.3f}"
            if 'mse' in stats:
                postfix['mse'] = f"{stats['mse']:.4f}"

            pbar.set_postfix(postfix)

            if log_wandb and global_step % 10 == 0:
                log_dict = {
                    'train/loss': loss.item() * gradient_accumulation_steps,
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/step': global_step
                }
                for key, value in stats.items():
                    log_dict[f'train/{key}'] = value
                wandb.log(log_dict)

        avg_train_loss = train_loss / len(train_loader)

        # Average stats
        for key in train_stats:
            train_stats[key] /= num_batches

        # Evaluation
        metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        if 'pairwise_accuracy' in train_stats:
            print(f"Train Pair Accuracy: {train_stats['pairwise_accuracy']:.4f}")
        if 'mse' in train_stats:
            print(f"Train MSE: {train_stats['mse']:.4f}")
        print(f"Val Metrics:")
        print(f"  Pearson:  {metrics['pearson']:.4f}")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  MAE:      {metrics['mae']:.4f}")
        print(f"  RMSE:     {metrics['rmse']:.4f}")

        if log_wandb:
            wandb.log({
                'val/pearson': metrics['pearson'],
                'val/spearman': metrics['spearman'],
                'val/mae': metrics['mae'],
                'val/rmse': metrics['rmse'],
                'train/epoch_loss': avg_train_loss,
                'epoch': epoch + 1
            })

        # Save best model (based on Pearson)
        if metrics['pearson'] > best_pearson:
            best_pearson = metrics['pearson']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': {
                    'model_name': model.model_name,
                    'embedding_dim': model.embedding_dim,
                    'pooling': model.pooling,
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang,
                    'loss_type': loss_type,
                    'margin': margin
                }
            }, f"{output_dir}/best_model.pt")
            print(f"✓ Saved best model (Pearson={best_pearson:.4f})")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, f"{output_dir}/checkpoint_epoch_{epoch + 1}.pt")

    print(f"\nTraining complete!")
    print(f"Best Pearson: {best_pearson:.4f}")

    return best_pearson


def main():
    parser = argparse.ArgumentParser(description="Train NLLB-QE with contrastive ranking loss")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-1.3B",
        choices=[
            "facebook/nllb-200-distilled-600M",
            "facebook/nllb-200-1.3B",
            "facebook/nllb-200-3.3B"
        ]
    )
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_model_scores", action="store_true")

    # Data arguments
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--val_tsv", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="sin_Sinh")
    parser.add_argument("--tgt_lang", type=str, default="eng_Latn")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/nllb_contrastive")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Loss arguments
    parser.add_argument(
        "--loss_type",
        type=str,
        default="pairwise",
        choices=["pairwise", "triplet", "listwise", "combined"],
        help="Type of ranking loss"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin for pairwise/triplet loss"
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.3,
        help="Weight for MSE in combined loss"
    )
    parser.add_argument(
        "--ranking_weight",
        type=float,
        default=0.7,
        help="Weight for ranking loss in combined loss"
    )
    parser.add_argument(
        "--hard_negative_mining",
        action="store_true",
        help="Use hard negative mining in pairwise loss"
    )

    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="nllb-qe-contrastive")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="nllb-qe-contrastive",
            name=args.run_name,
            config=vars(args)
        )

    # Load datasets
    print("Loading datasets...")
    train_dataset = CustomQEDataset(
        tsv_path=args.train_tsv,
        use_model_scores=args.use_model_scores
    )

    val_dataset = CustomQEDataset(
        tsv_path=args.val_tsv,
        use_model_scores=args.use_model_scores
    )

    # Initialize model
    print(f"\nInitializing model...")
    model = NLLBQEModel(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pooling=args.pooling,
        freeze_encoder=args.freeze_encoder,
        use_model_scores=args.use_model_scores
    )

    # Train
    best_pearson = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
        log_wandb=args.use_wandb,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type=args.loss_type,
        margin=args.margin,
        mse_weight=args.mse_weight,
        ranking_weight=args.ranking_weight,
        hard_negative_mining=args.hard_negative_mining
    )

    print(f"\nFinal Best Pearson: {best_pearson:.4f}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()