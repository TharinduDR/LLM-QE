#!/usr/bin/env python3
"""
Train NLLB-QE with pairwise contrastive loss.
Includes step-based evaluation.
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
    """Evaluate model on validation set."""
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
        hard_negative_mining: bool = False,
        eval_steps: int = 200,  # NEW: Evaluate every N steps
        save_steps: int = None,  # NEW: Save checkpoint every N steps (None = only save best)
        max_eval_batches: int = None  # NEW: Limit eval batches for speed (None = all)
):
    """
    Train with step-based evaluation.
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

    # Training state
    best_pearson = -1.0
    best_step = 0
    global_step = 0
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps

    print(f"\n{'=' * 70}")
    print(f"Starting Training with Step-based Evaluation")
    print(f"{'=' * 70}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Eval every: {eval_steps} steps")
    if save_steps:
        print(f"Save every: {save_steps} steps")
    print(f"Pairs per batch: ~{batch_size * (batch_size - 1) // 2}")
    print(f"Device: {device}")
    print(f"{'=' * 70}\n")

    # Track running averages for logging
    running_loss = 0.0
    running_stats = {}
    running_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(train_loader)
        )

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

            # Accumulate for logging
            running_loss += loss.item() * gradient_accumulation_steps
            running_count += 1
            for key, value in stats.items():
                running_stats[key] = running_stats.get(key, 0.0) + value

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                postfix = {
                    'step': global_step,
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                }
                if 'pairwise_accuracy' in stats:
                    postfix['pair_acc'] = f"{stats['pairwise_accuracy']:.3f}"
                pbar.set_postfix(postfix)

                # ============================================
                # STEP-BASED EVALUATION
                # ============================================
                if global_step % eval_steps == 0:
                    print(f"\n{'=' * 70}")
                    print(f"Evaluation at Step {global_step} (Epoch {epoch + 1})")
                    print(f"{'=' * 70}")

                    # Compute average training metrics
                    avg_running_loss = running_loss / running_count
                    avg_running_stats = {k: v / running_count for k, v in running_stats.items()}

                    print(f"Train Loss (last {running_count} batches): {avg_running_loss:.4f}")
                    if 'pairwise_accuracy' in avg_running_stats:
                        print(f"Train Pair Accuracy: {avg_running_stats['pairwise_accuracy']:.4f}")
                    if 'mse' in avg_running_stats:
                        print(f"Train MSE: {avg_running_stats['mse']:.4f}")

                    # Reset running averages
                    running_loss = 0.0
                    running_stats = {}
                    running_count = 0

                    # Evaluate on validation set
                    if max_eval_batches:
                        # Fast evaluation: only use subset
                        print(f"Running fast evaluation ({max_eval_batches} batches)...")
                        eval_loader_subset = list(val_loader)[:max_eval_batches]

                        # Create temporary loader
                        class ListDataLoader:
                            def __init__(self, data):
                                self.data = data

                            def __iter__(self):
                                return iter(self.data)

                        metrics = evaluate(model, ListDataLoader(eval_loader_subset), device)
                    else:
                        # Full evaluation
                        print("Running full evaluation...")
                        metrics = evaluate(model, val_loader, device)

                    print(f"Val Metrics:")
                    print(f"  Pearson:  {metrics['pearson']:.4f}")
                    print(f"  Spearman: {metrics['spearman']:.4f}")
                    print(f"  MAE:      {metrics['mae']:.4f}")
                    print(f"  RMSE:     {metrics['rmse']:.4f}")

                    # Log to wandb
                    if log_wandb:
                        log_dict = {
                            'val/pearson': metrics['pearson'],
                            'val/spearman': metrics['spearman'],
                            'val/mae': metrics['mae'],
                            'val/rmse': metrics['rmse'],
                            'train/avg_loss': avg_running_loss,
                            'train/step': global_step,
                            'train/epoch': epoch + (batch_idx / len(train_loader))
                        }
                        for key, value in avg_running_stats.items():
                            log_dict[f'train/avg_{key}'] = value
                        wandb.log(log_dict)

                    # Save best model
                    if metrics['pearson'] > best_pearson:
                        best_pearson = metrics['pearson']
                        best_step = global_step

                        torch.save({
                            'step': global_step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'metrics': metrics,
                            'best_pearson': best_pearson,
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

                        print(f"✓ Saved best model (Pearson={best_pearson:.4f} at step {best_step})")

                    print(f"{'=' * 70}\n")

                    # Back to training mode
                    model.train()

                # ============================================
                # STEP-BASED CHECKPOINTING
                # ============================================
                if save_steps and global_step % save_steps == 0:
                    checkpoint_path = f"{output_dir}/checkpoint_step_{global_step}.pt"
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpoint_path)
                    print(f"💾 Saved checkpoint at step {global_step}")

                # Log training metrics every 10 steps
                if log_wandb and global_step % 10 == 0:
                    log_dict = {
                        'train/loss': loss.item() * gradient_accumulation_steps,
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/step': global_step
                    }
                    for key, value in stats.items():
                        log_dict[f'train/{key}'] = value
                    wandb.log(log_dict)

            epoch_loss += loss.item() * gradient_accumulation_steps

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n{'=' * 70}")
        print(f"End of Epoch {epoch + 1}/{num_epochs}")
        print(f"Average Epoch Loss: {avg_epoch_loss:.4f}")
        print(f"Steps completed: {global_step}/{total_steps}")
        print(f"Best Pearson so far: {best_pearson:.4f} (at step {best_step})")
        print(f"{'=' * 70}\n")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"Training Complete!")
    print(f"{'=' * 70}")
    print(f"Total steps: {global_step}")
    print(f"Best Pearson: {best_pearson:.4f}")
    print(f"Best step: {best_step}")
    print(f"Best model saved to: {output_dir}/best_model.pt")
    print(f"{'=' * 70}\n")

    return best_pearson


def main():
    parser = argparse.ArgumentParser(description="Train NLLB-QE with step-based evaluation")

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

    # Evaluation arguments (NEW)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: only save best)"
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Max batches for evaluation (None=all, use small number for fast eval)"
    )

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
        hard_negative_mining=args.hard_negative_mining,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_eval_batches=args.max_eval_batches
    )

    print(f"\nFinal Best Pearson: {best_pearson:.4f}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()