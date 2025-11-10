# models/nllb_qe_model.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from typing import Dict, Optional


class NLLBQEModel(nn.Module):
    """
    NLLB encoder-based Quality Estimation for z-normalized scores.
    """

    def __init__(
            self,
            model_name: str = "facebook/nllb-200-1.3B",
            embedding_dim: int = 1024,
            dropout: float = 0.1,
            pooling: str = "mean",
            freeze_encoder: bool = False,
            use_model_scores: bool = False
    ):
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.use_model_scores = use_model_scores

        print(f"Loading NLLB model: {model_name}")

        # Load full model and extract encoder
        full_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = full_model.get_encoder()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Free memory
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Freeze encoder if requested
        if freeze_encoder:
            print("Freezing encoder parameters...")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get dimensions
        self.config = self.encoder.config
        self.hidden_size = self.config.d_model

        # Projection layer
        if self.embedding_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.embedding_dim)
        else:
            self.projection = nn.Identity()

        # Quality estimation head
        # Input: concatenated source + target embeddings
        qe_input_dim = self.embedding_dim * 2

        # Optionally include model scores
        if use_model_scores:
            qe_input_dim += 1  # Add 1 dimension for model score

        self.qe_head = nn.Sequential(
            nn.Linear(qe_input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim // 2, 1)
        )

        self._print_model_info(freeze_encoder)

    def _print_model_info(self, frozen: bool):
        """Print model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())

        print("\n" + "=" * 60)
        print("NLLB-QE Model Info")
        print("=" * 60)
        print(f"Base model: {self.model_name}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Pooling: {self.pooling}")
        print(f"Encoder params: {encoder_params:,}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(f"Encoder frozen: {frozen}")
        print(f"Use model scores: {self.use_model_scores}")
        print("=" * 60 + "\n")

    def pool_embeddings(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sentence embedding."""
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == "cls":
            return hidden_states[:, 0]

        elif self.pooling == "max":
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            hidden_states_masked = hidden_states.clone()
            hidden_states_masked[mask_expanded == 0] = -1e9
            return torch.max(hidden_states_masked, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
            self,
            src_input_ids: torch.Tensor,
            src_attention_mask: torch.Tensor,
            tgt_input_ids: torch.Tensor,
            tgt_attention_mask: torch.Tensor,
            model_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for quality estimation.

        Args:
            src_input_ids: Source token IDs [batch_size, src_len]
            src_attention_mask: Source attention mask [batch_size, src_len]
            tgt_input_ids: Target token IDs [batch_size, tgt_len]
            tgt_attention_mask: Target attention mask [batch_size, tgt_len]
            model_scores: Optional MT model confidence scores [batch_size]

        Returns:
            Dictionary with:
                - z_score_pred: Predicted z-normalized quality [batch_size, 1]
                - src_embedding: Source embeddings [batch_size, embedding_dim]
                - tgt_embedding: Target embeddings [batch_size, embedding_dim]
        """
        # Encode source
        src_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask
        )
        src_hidden = src_outputs.last_hidden_state
        src_embedding = self.pool_embeddings(src_hidden, src_attention_mask)
        src_embedding = self.projection(src_embedding)

        # Encode target
        tgt_outputs = self.encoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask
        )
        tgt_hidden = tgt_outputs.last_hidden_state
        tgt_embedding = self.pool_embeddings(tgt_hidden, tgt_attention_mask)
        tgt_embedding = self.projection(tgt_embedding)

        # Concatenate embeddings
        combined = torch.cat([src_embedding, tgt_embedding], dim=1)

        # Optionally add model scores
        if self.use_model_scores and model_scores is not None:
            model_scores = model_scores.unsqueeze(1)  # [batch_size, 1]
            combined = torch.cat([combined, model_scores], dim=1)

        # Predict z-score
        z_score_pred = self.qe_head(combined)

        return {
            'z_score_pred': z_score_pred,
            'src_embedding': src_embedding,
            'tgt_embedding': tgt_embedding
        }

    def predict(
            self,
            source_texts: list[str],
            target_texts: list[str],
            src_lang: str = "sin_Sinh",
            tgt_lang: str = "eng_Latn",
            model_scores: Optional[list[float]] = None,
            batch_size: int = 32
    ) -> list[float]:
        """
        Predict z-scores for translation pairs.
        """
        self.eval()
        device = next(self.parameters()).device

        predictions = []

        for i in range(0, len(source_texts), batch_size):
            batch_src = source_texts[i:i + batch_size]
            batch_tgt = target_texts[i:i + batch_size]
            batch_model_scores = model_scores[i:i + batch_size] if model_scores else None

            # Tokenize source
            self.tokenizer.src_lang = src_lang
            src_encoded = self.tokenizer(
                batch_src,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            src_encoded = {k: v.to(device) for k, v in src_encoded.items()}

            # Tokenize target
            self.tokenizer.src_lang = tgt_lang
            tgt_encoded = self.tokenizer(
                batch_tgt,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            tgt_encoded = {k: v.to(device) for k, v in tgt_encoded.items()}

            # Prepare model scores
            model_scores_tensor = None
            if batch_model_scores:
                model_scores_tensor = torch.tensor(batch_model_scores, device=device)

            with torch.no_grad():
                outputs = self.forward(
                    src_input_ids=src_encoded['input_ids'],
                    src_attention_mask=src_encoded['attention_mask'],
                    tgt_input_ids=tgt_encoded['input_ids'],
                    tgt_attention_mask=tgt_encoded['attention_mask'],
                    model_scores=model_scores_tensor
                )

                batch_preds = outputs['z_score_pred'].squeeze(-1).cpu().tolist()
                predictions.extend(batch_preds)

        return predictions