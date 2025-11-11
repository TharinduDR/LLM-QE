#!/usr/bin/env python3
"""
Train NLLB-QE with Full Encoder-Decoder (using decoder's cross-attention).
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

from models.nllb_qe_full_decoder import NLLBQEFullDecoder
from models.losses import PairwiseRankingLoss, CombinedQELoss
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

    results = {
        'pearson': pearsonr(predictions, targets)[0],
        'spearman': spearmanr(predictions, targets)[0],
        'mae': np.abs(predictions - targets).mean(),
        'rmse': np.sqrt(((predictions - targets) ** 2).mean())
    }

    return results


def train(
        model: NLLBQEFullDecoder,
        train_dataset: CustomQEDataset,
        val_dataset: CustomQEDataset,
        src_lang: str,
        tgt_lang: str,
        output_dir: str,
        batch_size: int = 8,  # Smaller batch for full model
        num_epochs: int = 5,
        learning_rate: float = 5e-6,  # Lower LR for full model
        warmup_ratio: float = 0.1,
        device: str = "cuda",
        log_wandb: bool = False,
        gradient_accumulation_steps: int = 2,  # More accumulation
        loss_type: str = "combined",
        margin: float = 0.5,
        mse_weight: float = 0.2,
        ranking_weight: float = 0.8,
        hard_negative_mining: bool = False,
        eval_steps: int = 200
):
    """Train with full encoder-decoder model."""
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

    # Loss
    if loss_type == "pairwise":
        criterion = PairwiseRankingLoss(
            margin=margin,
            hard_negative_mining=hard_negative_mining
        )
    elif loss_type == "combined":
        criterion = CombinedQELoss(
            mse_weight=mse_weight,
            ranking_weight=ranking_weight,
            margin=margin,
            hard_negative_mining=hard_negative_mining
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )

    # Scheduler
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

    print(f"\n{'=' * 70}")
    print(f"Starting Training with Full Encoder-Decoder")
    print(f"{'=' * 70}")
    print(f"Architecture: Encoder + Full Decoder (with pre-trained cross-attention)")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Eval every: {eval_steps} steps")
    print(f"{'=' * 70}\n")

    # Training loop
    running_loss = 0.0
    running_stats = {}
    running_count = 0

    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            src_input = {k: v.to(device) for k, v in batch['source'].items()}
            tgt_input = {k: v.to(device) for k, v in batch['target'].items()}
            z_means = batch['z_means'].to(device)

            model_scores = batch.get('model_scores')
            if model_scores is not None:
                model_scores = model_scores.to(device)

            # Forward
            outputs = model(
                src_input_ids=src_input['input_ids'],
                src_attention_mask=src_input['attention_mask'],
                tgt_input_ids=tgt_input['input_ids'],
                tgt_attention_mask=tgt_input['attention_mask'],
                model_scores=model_scores
            )

            z_score_pred = outputs['z_score_pred'].squeeze(-1)

            # Loss
            loss, stats = criterion(z_score_pred, z_means)
            loss = loss / gradient_accumulation_steps

            # Backward
            loss.backward()

            # Accumulate stats
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

                postfix = {
                    'step': global_step,
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                }
                if 'pairwise_accuracy' in stats:
                    postfix['pair_acc'] = f"{stats['pairwise_accuracy']:.3f}"
                pbar.set_postfix(postfix)

                # Evaluation
                if global_step % eval_steps == 0:
                    print(f"\n{'=' * 70}")
                    print(f"Evaluation at Step {global_step}")
                    print(f"{'=' * 70}")

                    avg_running_loss = running_loss / running_count
                    print(f"Train Loss: {avg_running_loss:.4f}")

                    running_loss = 0.0
                    running_stats = {}
                    running_count = 0

                    metrics = evaluate(model, val_loader, device)

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
                            'train/step': global_step
                        })

                    if metrics['pearson'] > best_pearson:
                        best_pearson = metrics['pearson']
                        best_step = global_step

                        torch.save({
                            'step': global_step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'metrics': metrics,
                            'config': {
                                'model_name': model.model_name,
                                'src_lang': src_lang,
                                'tgt_lang': tgt_lang
                            }
                        }, f"{output_dir}/best_model.pt")

                        print(f"✓ Saved best model (Pearson={best_pearson:.4f})")

                    print(f"{'=' * 70}\n")
                    model.train()

    print(f"\nBest Pearson: {best_pearson:.4f} at step {best_step}")
    return best_pearson


def main():
    parser = argparse.ArgumentParser(description="Train NLLB-QE with Full Decoder")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-1.3B")
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--use_model_scores", action="store_true")

    # Data arguments
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--val_tsv", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="sin_Sinh")
    parser.add_argument("--tgt_lang", type=str, default="eng_Latn")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/nllb_full_decoder")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=200)

    # Loss arguments
    parser.add_argument("--loss_type", type=str, default="combined")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--mse_weight", type=float, default=0.2)
    parser.add_argument("--ranking_weight", type=float, default=0.8)
    parser.add_argument("--hard_negative_mining", action="store_true")

    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="nllb-full-decoder")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project="nllb-qe-full-decoder",
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
    print(f"\nInitializing Full Encoder-Decoder model...")
    model = NLLBQEFullDecoder(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        pooling=args.pooling,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        num_decoder_layers=args.num_decoder_layers,
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
        eval_steps=args.eval_steps
    )

    print(f"\nFinal Best Pearson: {best_pearson:.4f}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()