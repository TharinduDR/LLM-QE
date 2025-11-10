import pandas as pd
import csv
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, List, Dict


class CustomQEDataset(TorchDataset):
    """
    Dataset loader for TSV format with z-normalized scores.

    Expected TSV columns (tab-separated):
    - index: sample ID
    - original: source text
    - translation: MT output
    - z_mean: z-normalized quality score (target)
    - scores: original scores (optional)
    - model_scores: MT model confidence (optional)
    """

    def __init__(
            self,
            tsv_path: str,
            use_model_scores: bool = False,
            index_col: str = "index"
    ):
        """
        Args:
            tsv_path: Path to TSV file
            use_model_scores: Whether to include model_scores as feature
            index_col: Name of the index column
        """
        print(f"Loading dataset from {tsv_path}")

        # Read TSV with proper settings
        self.data = pd.read_csv(
            tsv_path,
            sep='\t',  # Tab separator
            encoding='utf-8-sig',  # Handle BOM
            quoting=csv.QUOTE_NONE,  # No quote handling
            engine='python'  # Use Python engine for flexibility
        )

        self.use_model_scores = use_model_scores

        # Validate required columns
        required_cols = ['original', 'translation', 'z_mean']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Available columns: {list(self.data.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove rows with NaN in critical columns
        initial_len = len(self.data)
        self.data = self.data.dropna(subset=['original', 'translation', 'z_mean'])
        dropped = initial_len - len(self.data)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN values")

        # Convert z_mean to float
        self.data['z_mean'] = self.data['z_mean'].astype(float)

        # Convert model_scores if needed
        if self.use_model_scores and 'model_scores' in self.data.columns:
            self.data['model_scores'] = pd.to_numeric(self.data['model_scores'], errors='coerce')

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
        print(f"  Poor (z < -0.5):       {poor:5d} ({100 * poor / len(self.data):5.1f}%)")
        print(f"  Average (-0.5 to 0.5): {average:5d} ({100 * average / len(self.data):5.1f}%)")
        print(f"  Good (z > 0.5):        {good:5d} ({100 * good / len(self.data):5.1f}%)")

        # Show sample
        print(f"\nSample data (first row):")
        sample = self.data.iloc[0]
        print(f"  Original: {sample['original'][:80]}...")
        print(f"  Translation: {sample['translation'][:80]}...")
        print(f"  z_mean: {sample['z_mean']:.3f}")

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
            if pd.notna(row['model_scores']):
                item['model_score'] = float(row['model_scores'])
            else:
                item['model_score'] = 0.0  # Default if missing

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


# Backward compatibility: add helper function using your original code
def read_annotated_file(path, index="index"):
    """
    Read annotated TSV file (compatible with your original code).
    Returns pandas DataFrame.
    """
    indices = []
    originals = []
    translations = []
    z_means = []

    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            z_means.append(float(row["z_mean"]))

    return pd.DataFrame({
        'index': indices,
        'original': originals,
        'translation': translations,
        'z_mean': z_means
    })


def read_test_file(path, index="index"):
    """
    Read test TSV file without z_mean.
    Returns pandas DataFrame.
    """
    indices = []
    originals = []
    translations = []

    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])

    return pd.DataFrame({
        'index': indices,
        'original': originals,
        'translation': translations
    })