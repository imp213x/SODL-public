from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from sodl_weights import create_clustered_loss


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        hidden = self.proj(self.embed_tokens(input_ids))
        return hidden


def _time(label: str, fn, *, iterations: int) -> dict[str, float | str]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    avg_ms = statistics.mean(samples)
    return {"label": label, "avg_ms": avg_ms, "steps_per_sec": 1000.0 / avg_ms if avg_ms > 0 else 0.0}


def run(vocab_size: int, d_model: int, batch_size: int, seq_len: int, iterations: int) -> list[dict[str, float | str]]:
    model = _TinyLM(vocab_size=vocab_size, d_model=d_model)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    def _cross_entropy_step() -> None:
        hidden = model(input_ids)
        logits = model.lm_head(hidden)
        F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    clustered_loss, _, _ = create_clustered_loss(model, n_clusters=max(16, int(vocab_size ** 0.5)))

    def _clustered_step() -> None:
        hidden = model(input_ids)
        clustered_loss(hidden, labels)

    return [
        _time("cross_entropy", _cross_entropy_step, iterations=iterations),
        _time("clustered_softmax", _clustered_step, iterations=iterations),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ClusteredSoftmax against standard cross-entropy")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for result in run(args.vocab_size, args.d_model, args.batch_size, args.seq_len, args.iterations):
        print(f"{result['label']:>20}  avg_ms={result['avg_ms']:.3f}  steps_per_sec={result['steps_per_sec']:.2f}")


if __name__ == "__main__":
    main()
