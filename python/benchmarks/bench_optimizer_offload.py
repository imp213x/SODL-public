from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from sodl_weights import BlobStore
from sodl_weights.checkpoint import CheckpointManager
from sodl_weights.offload_optimizer import SODLAdamW
from sodl_weights.optimizer_state import OptimizerStateStore


class BenchNet(torch.nn.Module):
    def __init__(self, width: int = 512, depth: int = 4) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        depth = max(2, int(depth))
        for layer_index in range(depth):
            layers.append(torch.nn.Linear(width, width))
            if layer_index < depth - 1:
                layers.append(torch.nn.GELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.WorkingSetSize)
        return 0

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(usage.ru_maxrss)
        if rss < 1024 * 1024:
            rss *= 1024
        return rss
    except Exception:
        return 0


def _run_in_memory_adamw(
    steps: int,
    width: int,
    depth: int,
    checkpoint_mgr: CheckpointManager,
) -> dict[str, Any]:
    model = BenchNet(width, depth=depth)
    x = torch.randn(8, width)
    y = torch.randn(8, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rss_before = _rss_bytes()
    start = time.perf_counter()
    losses: list[float] = []
    step_latencies: list[float] = []
    for _ in range(steps):
        step_start = time.perf_counter()
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        step_latencies.append(time.perf_counter() - step_start)
        losses.append(float(loss.detach().item()))
    train_elapsed = time.perf_counter() - start
    rss_after = _rss_bytes()

    checkpoint_start = time.perf_counter()
    checkpoint_id = checkpoint_mgr.save_checkpoint(
        model,
        optimizer,
        step=steps,
        origin_id="bench-adamw",
        notes="benchmark in-memory AdamW",
    )
    checkpoint_elapsed = time.perf_counter() - checkpoint_start
    record = checkpoint_mgr.get_latest("bench-adamw")

    return {
        "optimizer": "adamw",
        "checkpoint_id": checkpoint_id,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "train_elapsed_sec": round(train_elapsed, 4),
        "avg_step_latency_sec": round(sum(step_latencies) / max(1, len(step_latencies)), 4),
        "checkpoint_write_sec": round(checkpoint_elapsed, 4),
        "final_loss": round(losses[-1], 6) if losses else None,
        "checkpoint_size_stored": int(record.size_stored) if record is not None else 0,
    }


def _run_sodl_offload(
    steps: int,
    width: int,
    depth: int,
    checkpoint_mgr: CheckpointManager,
    tmpdir: str,
    cache_capacity: int,
    writeback_threshold: int,
    *,
    origin_id: str,
    label: str,
    use_batch_state_ops: bool,
    block_size: int,
) -> dict[str, Any]:
    model = BenchNet(width, depth=depth)
    x = torch.randn(8, width)
    y = torch.randn(8, width)
    store = OptimizerStateStore(
        f"{tmpdir}/optimizer_blobs",
        f"{tmpdir}/optimizer_registry",
        cache_capacity=cache_capacity,
        writeback_threshold=writeback_threshold,
    )
    optimizer = SODLAdamW(
        list(model.named_parameters()),
        state_store=store,
        origin_id=origin_id,
        block_size=block_size,
        flush_every=1,
        prefetch_lookahead=1,
        async_writeback=True,
        use_batch_state_ops=use_batch_state_ops,
        lr=1e-3,
    )

    rss_before = _rss_bytes()
    start = time.perf_counter()
    losses: list[float] = []
    step_latencies: list[float] = []
    for _ in range(steps):
        step_start = time.perf_counter()
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.wait_for_pending_flush()
        step_latencies.append(time.perf_counter() - step_start)
        losses.append(float(loss.detach().item()))
    train_elapsed = time.perf_counter() - start
    rss_after = _rss_bytes()

    checkpoint_start = time.perf_counter()
    checkpoint_id = checkpoint_mgr.save_checkpoint(
        model,
        optimizer,
        step=steps,
        origin_id=origin_id,
        dataset_manifests=["dataset://bench-synthetic"],
        knowledge_logs=["knowledge://bench-optimizer-offload.jsonl"],
        notes=f"benchmark {label}",
    )
    checkpoint_elapsed = time.perf_counter() - checkpoint_start
    record = checkpoint_mgr.get_latest(origin_id)
    manifest = store.manifest(origin_id)
    total_raw = sum(item.size_raw for item in manifest.blocks.values())
    total_stored = sum(item.size_stored for item in manifest.blocks.values())
    compression_ratio = round(total_raw / total_stored, 4) if total_stored else None

    return {
        "optimizer": label,
        "checkpoint_id": checkpoint_id,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "train_elapsed_sec": round(train_elapsed, 4),
        "avg_step_latency_sec": round(sum(step_latencies) / max(1, len(step_latencies)), 4),
        "checkpoint_write_sec": round(checkpoint_elapsed, 4),
        "final_loss": round(losses[-1], 6) if losses else None,
        "checkpoint_size_stored": int(record.size_stored) if record is not None else 0,
        "offloaded_blocks": len(manifest.blocks),
        "optimizer_state_raw_bytes": total_raw,
        "optimizer_state_stored_bytes": total_stored,
        "optimizer_state_compression_ratio": compression_ratio,
        "parameter_tensors": len(list(model.parameters())),
        "use_batch_state_ops": use_batch_state_ops,
        "layout_fingerprint": optimizer.external_state_dict().get("layout_fingerprint"),
    }


def _write_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SODL optimizer offload against in-memory AdamW")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--cache-capacity", type=int, default=16)
    parser.add_argument("--writeback-threshold", type=int, default=1)
    parser.add_argument("--jsonl-out", type=str, default="")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_mgr = CheckpointManager(
            BlobStore(f"{tmpdir}/checkpoint_blobs"),
            f"{tmpdir}/checkpoint_registry",
        )
        adam_metrics = _run_in_memory_adamw(args.steps, args.width, args.depth, checkpoint_mgr)
        batched_offload_metrics = _run_sodl_offload(
            args.steps,
            args.width,
            args.depth,
            checkpoint_mgr,
            tmpdir,
            args.cache_capacity,
            args.writeback_threshold,
            origin_id="bench-offload-batched",
            label="sodl_offload_batched",
            use_batch_state_ops=True,
            block_size=args.block_size,
        )
        legacy_offload_metrics = _run_sodl_offload(
            args.steps,
            args.width,
            args.depth,
            checkpoint_mgr,
            tmpdir,
            args.cache_capacity,
            args.writeback_threshold,
            origin_id="bench-offload-legacy",
            label="sodl_offload_legacy",
            use_batch_state_ops=False,
            block_size=args.block_size,
        )

        payload = {
            "ts": _utc_now(),
            "kind": "optimizer_offload_benchmark",
            "framework": "sodl",
            "steps": args.steps,
            "width": args.width,
            "depth": args.depth,
            "block_size": args.block_size,
            "cache_capacity": args.cache_capacity,
            "writeback_threshold": args.writeback_threshold,
            "adamw": adam_metrics,
            "sodl_offload_batched": batched_offload_metrics,
            "sodl_offload_legacy": legacy_offload_metrics,
            "delta": {
                "batched_vs_adamw": {
                    "train_elapsed_sec": round(
                        batched_offload_metrics["train_elapsed_sec"] - adam_metrics["train_elapsed_sec"],
                        4,
                    ),
                    "avg_step_latency_sec": round(
                        batched_offload_metrics["avg_step_latency_sec"] - adam_metrics["avg_step_latency_sec"],
                        4,
                    ),
                    "checkpoint_write_sec": round(
                        batched_offload_metrics["checkpoint_write_sec"] - adam_metrics["checkpoint_write_sec"],
                        4,
                    ),
                    "rss_after_bytes": int(
                        batched_offload_metrics["rss_after_bytes"] - adam_metrics["rss_after_bytes"]
                    ),
                },
                "legacy_vs_adamw": {
                    "train_elapsed_sec": round(
                        legacy_offload_metrics["train_elapsed_sec"] - adam_metrics["train_elapsed_sec"],
                        4,
                    ),
                    "avg_step_latency_sec": round(
                        legacy_offload_metrics["avg_step_latency_sec"] - adam_metrics["avg_step_latency_sec"],
                        4,
                    ),
                    "checkpoint_write_sec": round(
                        legacy_offload_metrics["checkpoint_write_sec"] - adam_metrics["checkpoint_write_sec"],
                        4,
                    ),
                    "rss_after_bytes": int(
                        legacy_offload_metrics["rss_after_bytes"] - adam_metrics["rss_after_bytes"]
                    ),
                },
                "batched_vs_legacy": {
                    "train_elapsed_sec": round(
                        batched_offload_metrics["train_elapsed_sec"] - legacy_offload_metrics["train_elapsed_sec"],
                        4,
                    ),
                    "avg_step_latency_sec": round(
                        batched_offload_metrics["avg_step_latency_sec"] - legacy_offload_metrics["avg_step_latency_sec"],
                        4,
                    ),
                    "checkpoint_write_sec": round(
                        batched_offload_metrics["checkpoint_write_sec"] - legacy_offload_metrics["checkpoint_write_sec"],
                        4,
                    ),
                    "rss_after_bytes": int(
                        batched_offload_metrics["rss_after_bytes"] - legacy_offload_metrics["rss_after_bytes"]
                    ),
                },
            },
        }

        if args.jsonl_out:
            _write_jsonl(args.jsonl_out, payload)

        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
