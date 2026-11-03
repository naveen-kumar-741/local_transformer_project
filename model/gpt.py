import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv = nn.Linear(
            cfg.n_embd,
            3 * cfg.n_embd,
            bias=cfg.bias,
        )

        self.proj = nn.Linear(
            cfg.n_embd,
            cfg.n_embd,
            bias=cfg.bias,
        )

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape

        q, k, v = self.qkv(x).split(channels, dim=2)

        q = q.view(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)

        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                cfg.n_embd,
                4 * cfg.n_embd,
                bias=cfg.bias,
            ),
            nn.GELU(),
            nn.Linear(
                4 * cfg.n_embd,
                cfg.n_embd,
                bias=cfg.bias,
            ),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)

        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg

        self.token_embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.n_embd,
        )

        self.position_embedding = nn.Embedding(
            cfg.block_size,
            cfg.n_embd,
        )

        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        self.ln_f = nn.LayerNorm(cfg.n_embd)

        self.lm_head = nn.Linear(
            cfg.n_embd,
            cfg.vocab_size,
            bias=False,
        )

        # Weight tying.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        for name, parameter in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.n_layer),
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        if sequence_length > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block_size {self.cfg.block_size}"
            )

        positions = torch.arange(
            sequence_length,
            device=idx.device,
        )

        x = self.drop(
            self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        )

        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x)

        logits = self.lm_head(self.ln_f(x))

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @staticmethod
    def _apply_top_k(logits, top_k):
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits

        threshold = torch.topk(
            logits,
            top_k,
            dim=-1,
        ).values[:, -1, None]

        return logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

    @staticmethod
    def _apply_top_p(logits, top_p):
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep the first token that crosses the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(
            logits,
            dtype=torch.bool,
        )

        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        return logits.masked_fill(
            indices_to_remove,
            float("-inf"),
        )

    @staticmethod
    def _apply_repetition_penalty(logits, idx, penalty):
        if penalty == 1.0:
            return logits

        for batch_index in range(idx.size(0)):
            previous_tokens = torch.unique(idx[batch_index])

            token_logits = logits[batch_index, previous_tokens]

            adjusted_logits = torch.where(
                token_logits < 0,
                token_logits * penalty,
                token_logits / penalty,
            )

            logits[batch_index, previous_tokens] = adjusted_logits

        return logits

    @staticmethod
    def _apply_no_repeat_ngram(logits, idx, ngram_size):
        if ngram_size <= 0:
            return logits

        batch_size, sequence_length = idx.shape

        if sequence_length < ngram_size:
            return logits

        for batch_index in range(batch_size):
            sequence = idx[batch_index].tolist()

            if ngram_size == 1:
                banned_tokens = set(sequence)

            else:
                current_prefix = tuple(sequence[-(ngram_size - 1) :])

                banned_tokens = set()

                for position in range(sequence_length - ngram_size + 1):
                    previous_prefix = tuple(
                        sequence[position : position + ngram_size - 1]
                    )

                    if previous_prefix == current_prefix:
                        banned_tokens.add(sequence[position + ngram_size - 1])

            if banned_tokens:
                banned_ids = list(banned_tokens)

                original_logits = logits[
                    batch_index,
                    banned_ids,
                ].clone()

                logits[
                    batch_index,
                    banned_ids,
                ] = float("-inf")

                # Avoid an invalid probability distribution if every
                # vocabulary token becomes blocked.
                if torch.isneginf(logits[batch_index]).all():
                    logits[
                        batch_index,
                        banned_ids,
                    ] = original_logits

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    ):
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError(
                "Input must have shape [batch_size, sequence_length] "
                "with a non-empty sequence."
            )

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        if top_k < 0:
            raise ValueError("top_k cannot be negative.")

        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1].")

        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be greater than or equal to 1.0.")

        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative.")

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the most recent context tokens.
            idx_cond = idx[:, -self.cfg.block_size :]

            logits, _ = self(idx_cond)

            # Use the last token's predictions.
            logits = logits[:, -1, :]

            # Temperature sampling.
            logits = logits / temperature

            # Repetition penalty.
            logits = self._apply_repetition_penalty(
                logits,
                idx,
                repetition_penalty,
            )

            # No-repeat n-gram.
            logits = self._apply_no_repeat_ngram(
                logits,
                idx,
                no_repeat_ngram_size,
            )

            # Top-k sampling.
            logits = self._apply_top_k(
                logits,
                top_k,
            )

            # Top-p sampling.
            logits = self._apply_top_p(
                logits,
                top_p,
            )

            # Convert logits into probabilities.
            probs = F.softmax(logits, dim=-1)

            # Sample the next token.
            next_id = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, next_id),
                dim=1,
            )

        return idx


def count_parameters(model):
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
