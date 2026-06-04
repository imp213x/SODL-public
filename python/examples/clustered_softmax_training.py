"""Minimal clustered softmax training example."""

from __future__ import annotations

import torch
import torch.nn as nn

from sodl import create_clustered_loss


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 32) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed_tokens(idx))


def main() -> None:
    model = TinyLM()
    loss_fn, token_index, stats = create_clustered_loss(model, n_clusters=16, adaptive=True)
    hidden = model(torch.randint(0, 256, (2, 8)))
    labels = torch.randint(0, 256, (2, 8))
    loss = loss_fn(hidden, labels)
    print(f"clusters={stats['n_clusters']} vocab={token_index.vocab_size} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
