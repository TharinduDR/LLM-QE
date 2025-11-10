import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, List, Dict


class CustomQEDataset(TorchDataset):
    """
    Dataset loader for your custom format with z-normalized scores.

    Expected CSV columns:
    - original: source text
    - translation: MT output
    - z_mean: z-normalized quality score (target)
    - scores: original scores (optional)
    - model_scores: MT model confidence (optional)
    """

    def __init__(
            self,
            csv_path: str,
            use_model_scores: bool = False
    ):
        """
        Args:
            csv_path: Path to CSV file
            use_model_scores: Whether to include model_scores as feature
        """
        print(f"Loading dataset from {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.use_model_scores = use_model_scores

        # Validate required columns
        required_cols = ['original', 'translation', 'z_mean']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Remove rows with NaN in critical columns
        self.data = self.data.dropna(subset=['original', 'translation', 'z_mean'])

        # Convert z_mean to float
        self.data['z_mean'] = self.data['z_mean'].astype(float)

        print(f"Loaded {len(self.data)} samples")
        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\nDataset Statistics:")
        print(f"  Samples: {len(self.data)}")
        print(f"  z_mean range: [{self.data['z_mean'].min():.3f}, {self.data['z_mean'].max():.3f}]")
        print(f"  z_mean mean: {self.data['z_mean'].mean():.3f}")
        print(f"  z_mean std: {self.data['z_mean'].std():.3f}")

        # Quality distribution
        poor = (self.data['z_mean'] < -0.5).sum()
        average = ((self.data['z_mean'] >= -0.5) & (self.data['z_mean'] <= 0.5)).sum()
        good = (self.data['z_mean'] > 0.5).sum()

        print(f"\nQuality Distribution:")
        print(f"  Poor (z < -0.5):     {poor:5d} ({100 * poor / len(self.data):5.1f}%)")
        print(f"  Average (-0.5 to 0.5): {average:5d} ({100 * average / len(self.data):5.1f}%)")
        print(f"  Good (z > 0.5):      {good:5d} ({100 * good / len(self.data):5.1f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        item = {
            'source': str(row['original']),
            'target': str(row['translation']),
            'z_mean': float(row['z_mean'])
        }

        # Optionally include model scores
        if self.use_model_scores and 'model_scores' in self.data.columns:
            item['model_score'] = float(row['model_scores'])

        return item


def collate_custom_nllb(batch: List[Dict], tokenizer, src_lang: str, tgt_lang: str) -> Dict:
    """
    Collate function for custom dataset with NLLB.

    Args:
        batch: List of items from CustomQEDataset
        tokenizer: NLLB tokenizer
        src_lang: Source language code (e.g., 'sin_Sinh' for Sinhala)
        tgt_lang: Target language code (e.g., 'eng_Latn' for English)
    """
    sources = [item['source'] for item in batch]
    targets = [item['target'] for item in batch]
    z_means = [item['z_mean'] for item in batch]

    # Tokenize source with source language
    tokenizer.src_lang = src_lang
    src_encoded = tokenizer(
        sources,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Tokenize target with target language
    tokenizer.src_lang = tgt_lang
    tgt_encoded = tokenizer(
        targets,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    result = {
        'source': src_encoded,
        'target': tgt_encoded,
        'z_means': torch.tensor(z_means, dtype=torch.float32)
    }

    # Include model scores if available
    if 'model_score' in batch[0]:
        result['model_scores'] = torch.tensor(
            [item['model_score'] for item in batch],
            dtype=torch.float32
        )

    return result