import torch
import torch.nn as nn
from transformers import M2M100ForConditionalGeneration, AutoTokenizer
from typing import Dict, Optional


class NLLBQECrossAttention(nn.Module):
    """
    NLLB-based Quality Estimation with Cross-Attention.

    Architecture:
    1. Encode source and target separately with NLLB encoder
    2. Apply cross-attention: target attends to source
    3. Combine representations and predict quality

    This models translation alignment explicitly!
    """

    def __init__(
            self,
            model_name: str = "facebook/nllb-200-1.3B",
            embedding_dim: int = 1024,
            num_attention_heads: int = 8,
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
        self.hidden_size = self.config.d_model  # 1024 for 1.3B

        # Cross-attention layer (target queries, source keys/values)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization after cross-attention
        self.cross_attn_layer_norm = nn.LayerNorm(self.hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Projection layer (if needed)
        if self.embedding_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.embedding_dim)
        else:
            self.projection = nn.Identity()

        # Quality estimation head
        # Input: [src_embedding, tgt_embedding, cross_attn_embedding]
        qe_input_dim = self.embedding_dim * 3

        if use_model_scores:
            qe_input_dim += 1  # Add model score

        self.qe_head = nn.Sequential(
            nn.Linear(qe_input_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 1)
        )

        self._print_model_info(freeze_encoder, num_attention_heads)

    def _print_model_info(self, frozen: bool, num_heads: int):
        """Print model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        cross_attn_params = sum(p.numel() for p in self.cross_attention.parameters())

        print("\n" + "=" * 60)
        print("NLLB-QE with Cross-Attention")
        print("=" * 60)
        print(f"Base model: {self.model_name}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Pooling: {self.pooling}")
        print(f"Cross-attention heads: {num_heads}")
        print(f"Encoder params: {encoder_params:,}")
        print(f"Cross-attention params: {cross_attn_params:,}")
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
            # Mean pooling with attention mask
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
            model_scores: Optional[torch.Tensor] = None,
            return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-attention.

        Args:
            src_input_ids: Source token IDs [batch_size, src_len]
            src_attention_mask: Source attention mask [batch_size, src_len]
            tgt_input_ids: Target token IDs [batch_size, tgt_len]
            tgt_attention_mask: Target attention mask [batch_size, tgt_len]
            model_scores: Optional MT model confidence [batch_size]
            return_attention_weights: Whether to return attention weights

        Returns:
            Dictionary with predictions and embeddings
        """
        batch_size = src_input_ids.size(0)

        # ============================================
        # 1. Encode source and target separately
        # ============================================

        # Encode source
        src_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask
        )
        src_hidden = src_outputs.last_hidden_state  # [batch, src_len, hidden]

        # Encode target
        tgt_outputs = self.encoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask
        )
        tgt_hidden = tgt_outputs.last_hidden_state  # [batch, tgt_len, hidden]

        # ============================================
        # 2. Cross-attention: target attends to source
        # ============================================

        # Create key padding mask (True for positions to IGNORE)
        src_key_padding_mask = ~src_attention_mask.bool()  # [batch, src_len]

        # Cross-attention
        # query: target representations
        # key, value: source representations
        cross_attn_output, attention_weights = self.cross_attention(
            query=tgt_hidden,  # [batch, tgt_len, hidden]
            key=src_hidden,  # [batch, src_len, hidden]
            value=src_hidden,  # [batch, src_len, hidden]
            key_padding_mask=src_key_padding_mask,  # [batch, src_len]
            need_weights=return_attention_weights,
            average_attn_weights=True
        )
        # cross_attn_output: [batch, tgt_len, hidden]
        # attention_weights: [batch, tgt_len, src_len] (if return_attention_weights)

        # Residual connection + layer norm
        tgt_hidden_with_cross = self.cross_attn_layer_norm(
            tgt_hidden + self.dropout(cross_attn_output)
        )

        # ============================================
        # 3. Pool representations
        # ============================================

        # Pool source (semantic meaning of source)
        src_embedding = self.pool_embeddings(src_hidden, src_attention_mask)

        # Pool target (semantic meaning of target)
        tgt_embedding = self.pool_embeddings(tgt_hidden, tgt_attention_mask)

        # Pool cross-attended target (alignment-aware target)
        tgt_cross_embedding = self.pool_embeddings(
            tgt_hidden_with_cross,
            tgt_attention_mask
        )

        # Project to embedding dimension
        src_embedding = self.projection(src_embedding)
        tgt_embedding = self.projection(tgt_embedding)
        tgt_cross_embedding = self.projection(tgt_cross_embedding)

        # ============================================
        # 4. Combine and predict quality
        # ============================================

        # Concatenate all representations
        # - src_embedding: what the source says
        # - tgt_embedding: what the target says
        # - tgt_cross_embedding: how well target aligns with source
        combined = torch.cat([
            src_embedding,
            tgt_embedding,
            tgt_cross_embedding
        ], dim=1)  # [batch, embedding_dim * 3]

        # Optionally add model scores
        if self.use_model_scores and model_scores is not None:
            model_scores = model_scores.unsqueeze(1)  # [batch, 1]
            combined = torch.cat([combined, model_scores], dim=1)

        # Predict quality score
        z_score_pred = self.qe_head(combined)  # [batch, 1]

        # ============================================
        # 5. Return results
        # ============================================

        result = {
            'z_score_pred': z_score_pred,
            'src_embedding': src_embedding,
            'tgt_embedding': tgt_embedding,
            'tgt_cross_embedding': tgt_cross_embedding
        }

        if return_attention_weights and attention_weights is not None:
            result['attention_weights'] = attention_weights

        return result

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