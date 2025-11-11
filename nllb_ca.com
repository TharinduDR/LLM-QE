#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mem=80G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python train_cross_attention.py \
    --model_name "facebook/nllb-200-1.3B" \
    --train_tsv "data/si-en/train.sien.df.short.tsv" \
    --val_tsv "data/si-en/dev.sien.df.short.tsv" \
    --src_lang "sin_Sinh" \
    --tgt_lang "eng_Latn" \
    --loss_type "combined" \
    --mse_weight 0.2 \
    --ranking_weight 0.8 \
    --hard_negative_mining \
    --num_attention_heads 8 \
    --batch_size 16 \
    --num_epochs 5 \
    --eval_steps 200 \
    --learning_rate 1e-5
