#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mem=80G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python train_nllb_custom.py \
    --model_name "facebook/nllb-200-1.3B" \
    --train_csv "data/si-en/train.sien.df.short.tsv" \
    --val_csv "data/si-en/dev.sien.df.short.tsv" \
    --src_lang "sin_Sinh" \
    --tgt_lang "eng_Latn" \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 1e-5

python train_nllb_custom.py \
    --model_name "facebook/nllb-200-1.3B" \
    --train_tsv "data/si-en/train.sien.df.short.tsv" \
    --val_tsv "data/si-en/dev.sien.df.short.tsv" \
    --src_lang "sin_Sinh" \
    --tgt_lang "eng_Latn" \
    --loss_type "pairwise" \
    --margin 0.5 \
    --hard_negative_mining \
    --batch_size 16 \
    --num_epochs 5 \
    --eval_steps 200 \
    --learning_rate 1e-5
