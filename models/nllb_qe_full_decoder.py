import torch
import torch.nn as nn
from transformers import M2M100ForConditionalGeneration, AutoTokenizer
from typing import Dict, Optional


class NLLBQEFullDecoder(nn.Module):
    """
    NLLB-based Quality Estimation using Full Encoder-Decoder.

    Architecture:
    1. Source → Encoder → Hidden states
    2. Target → Decoder (with cross-attention to source) → Decoder output
    3. Pool decoder output → Quality Score

    This uses NLLB's pre-trained decoder cross-attention,
    which already knows how to align source and target!
    """

    def __init__(
            self,
            model_name: str = "facebook/nllb-200-1.3B",
            embedding_dim: int = 1024,
            dropout: float = 0.1,
            pooling: str = "mean",
            freeze_encoder: bool = False,
            freeze_decoder: bool = False,
            use_model_scores: bool = False,
            num_decoder_layers: int = None  # None = use all layers
    ):
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.use_model_scores = use_model_scores

        print(f"Loading NLLB model: {model_name}")

        # Load full encoder-decoder model
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get config
        self.config = self.model.config
        self.hidden_size = self.config.d_model

        # Optionally use subset of decoder layers
        if num_decoder_layers is not None and num_decoder_layers < len(self.decoder.layers):
            print(f"Using only first {num_decoder_layers} decoder layers (out of {len(self.decoder.layers)})")
            self.decoder.layers = self.decoder.layers[:num_decoder_layers]

        # Freeze components if requested
        if freeze_encoder:
            print("Freezing encoder parameters...")
            for param in self.encoder.parameters():
                param.requires_grad = False

        if freeze_decoder:
            print("Freezing decoder parameters...")
            for param in self.decoder.parameters():
                param.requires_grad = False

        # Projection layer
        if self.embedding_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, self.embedding_dim)
        else:
            self.projection = nn.Identity()

        # Quality estimation head
        # Input: [src_pool, tgt_pool, decoder_pool]
        qe_input_dim = self.embedding_dim * 3

        if use_model_scores:
            qe_input_dim += 1

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

        self._print_model_info(freeze_encoder, freeze_decoder)

    def _print_model_info(self, freeze_enc: bool, freeze_dec: bool):
        """Print model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        print("\n" + "=" * 60)
        print("NLLB-QE with Full Encoder-Decoder")
        print("=" * 60)
        print(f"Base model: {self.model_name}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Embedding dim: {self.embedding_dim}")
        print(f"Pooling: {self.pooling}")
        print(f"Decoder layers: {len(self.decoder.layers)}")
        print(f"Encoder params: {encoder_params:,}")
        print(f"Decoder params: {decoder_params:,}")
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(f"Encoder frozen: {freeze_enc}")
        print(f"Decoder frozen: {freeze_dec}")
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

        elif self.pooling == "last":
            # Last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            return hidden_states[torch.arange(batch_size), sequence_lengths]

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
            return_cross_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with full encoder-decoder.

        Args:
            src_input_ids: Source token IDs [batch_size, src_len]
            src_attention_mask: Source attention mask [batch_size, src_len]
            tgt_input_ids: Target token IDs [batch_size, tgt_len]
            tgt_attention_mask: Target attention mask [batch_size, tgt_len]
            model_scores: Optional MT model confidence [batch_size]
            return_cross_attentions: Whether to return cross-attention weights

        Returns:
            Dictionary with predictions and embeddings
        """
        batch_size = src_input_ids.size(0)

        # ============================================
        # 1. Encode source
        # ============================================
        encoder_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch, src_len, hidden]

        # ============================================
        # 2. Decode target with cross-attention to source
        # ============================================

        # The decoder will:
        # - Take target tokens as input
        # - Self-attend within target (with causal mask)
        # - Cross-attend to encoder_hidden_states (source)
        # This is exactly what happens during translation!

        decoder_outputs = self.decoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=src_attention_mask,
            output_attentions=return_cross_attentions,
            return_dict=True
        )

        decoder_hidden_states = decoder_outputs.last_hidden_state  # [batch, tgt_len, hidden]

        # ============================================
        # 3. Pool representations
        # ============================================

        # Pool encoder output (source semantics)
        src_embedding = self.pool_embeddings(encoder_hidden_states, src_attention_mask)

        # Pool decoder input side (target semantics before cross-attention)
        # We can get this from the decoder's self-attention
        # For simplicity, we'll encode target separately
        tgt_encoder_outputs = self.encoder(
            input_ids=tgt_input_ids,
            attention_mask=tgt_attention_mask,
            return_dict=True
        )
        tgt_embedding = self.pool_embeddings(
            tgt_encoder_outputs.last_hidden_state,
            tgt_attention_mask
        )

        # Pool decoder output (target with cross-attention to source)
        # This contains alignment information!
        decoder_embedding = self.pool_embeddings(decoder_hidden_states, tgt_attention_mask)

        # Project all embeddings
        src_embedding = self.projection(src_embedding)
        tgt_embedding = self.projection(tgt_embedding)
        decoder_embedding = self.projection(decoder_embedding)

        # ============================================
        # 4. Combine and predict quality
        # ============================================

        # Concatenate representations:
        # - src_embedding: source semantics
        # - tgt_embedding: target semantics
        # - decoder_embedding: alignment-aware target (with decoder's cross-attention)
        combined = torch.cat([
            src_embedding,
            tgt_embedding,
            decoder_embedding
        ], dim=1)

        # Optionally add model scores
        if self.use_model_scores and model_scores is not None:
            model_scores = model_scores.unsqueeze(1)
            combined = torch.cat([combined, model_scores], dim=1)

        # Predict quality
        z_score_pred = self.qe_head(combined)

        # ============================================
        # 5. Return results
        # ============================================

        result = {
            'z_score_pred': z_score_pred,
            'src_embedding': src_embedding,
            'tgt_embedding': tgt_embedding,
            'decoder_embedding': decoder_embedding
        }

        # Optionally return cross-attention weights
        if return_cross_attentions and decoder_outputs.cross_attentions is not None:
            # Average across layers and heads for visualization
            cross_attns = torch.stack(decoder_outputs.cross_attentions)  # [layers, batch, heads, tgt_len, src_len]
            avg_cross_attn = cross_attns.mean(dim=(0, 2))  # [batch, tgt_len, src_len]
            result['cross_attentions'] = avg_cross_attn

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