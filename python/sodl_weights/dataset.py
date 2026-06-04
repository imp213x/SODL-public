"""SODL-backed Dataset — PyTorch Dataset with content-addressed storage.

Enables training pipelines to load data from SODL blob stores with:
  - Manifest-based shard management
  - Lazy loading with LRU caching
  - Content deduplication across dataset versions
  - Optional prefetching

Example
-------
>>> ds = SODLDataset.from_manifest("manifest.json", "./blobs")
>>> sample = ds[0]  # lazy-loads and caches the shard
>>> loader = DataLoader(ds, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zstandard as zstd

from sodl_weights.store import BlobStore, compute_blob_id

try:
    from torch.utils.data import Dataset as TorchDataset
except Exception:  # pragma: no cover - torch is optional
    class TorchDataset:  # type: ignore[no-redef]
        pass


class SODLDataset(TorchDataset):
    """A dataset backed by SODL content-addressed blob storage.

    Each shard is a numpy array stored as a SODL blob. Shards are loaded
    lazily and cached in memory for fast repeated access.

    Parameters
    ----------
    shards : list[dict]
        List of shard descriptors, each with:
          - blob_id: str — SODL blob ID
          - n_samples: int — number of samples in this shard
          - shape: list[int] — shape of each sample (optional)
    blob_store : BlobStore
        SODL blob store containing the shard data.
    transform : callable, optional
        Transform applied to each sample after loading.
    """

    def __init__(
        self,
        shards: list[dict[str, Any]],
        blob_store: BlobStore,
        transform: Any = None,
        cache_capacity: int = 2,
    ) -> None:
        self._shards = shards
        self._blob_store = blob_store
        self._transform = transform
        self._decompressor = zstd.ZstdDecompressor()
        self._cache_capacity = max(0, int(cache_capacity))

        # Build cumulative index for O(1) shard lookup
        self._cumulative = []
        total = 0
        for shard in shards:
            total += shard["n_samples"]
            self._cumulative.append(total)
        self._total_samples = total

        # Shard cache: shard_idx -> numpy array
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx = self._total_samples + idx
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_samples})")

        # Find which shard this sample belongs to
        shard_idx = bisect.bisect_right(self._cumulative, idx)

        # Offset within the shard
        offset = idx if shard_idx == 0 else idx - self._cumulative[shard_idx - 1]

        # Load shard if not cached
        if shard_idx not in self._cache:
            self._cache_shard(shard_idx, self._load_shard(shard_idx))
        else:
            self._cache.move_to_end(shard_idx)

        sample = self._cache[shard_idx][offset]

        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        """Load and decompress a shard from the blob store."""
        shard = self._shards[shard_idx]
        blob_id = shard["blob_id"]
        compressed = self._blob_store.get(blob_id)

        try:
            raw = self._decompressor.decompress(compressed)
        except zstd.ZstdError:
            raw = compressed

        arr = np.load(
            __import__("io").BytesIO(raw),
            allow_pickle=False,
        )
        return arr

    def _cache_shard(self, shard_idx: int, shard: np.ndarray) -> None:
        self._cache[shard_idx] = shard
        self._cache.move_to_end(shard_idx)
        if self._cache_capacity <= 0:
            self._cache.clear()
            return
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Free cached shards from memory."""
        self._cache.clear()

    def prefetch_shards(self, shard_indices: Sequence[int]) -> None:
        """Warm the local shard cache ahead of iteration."""
        for shard_idx in shard_indices:
            if shard_idx < 0 or shard_idx >= len(self._shards):
                continue
            if shard_idx in self._cache:
                self._cache.move_to_end(shard_idx)
                continue
            self._cache_shard(shard_idx, self._load_shard(shard_idx))

    def shard_ids_for_worker(
        self,
        *,
        worker_id: int,
        num_workers: int,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> list[int]:
        """Assign shards deterministically across data-loader workers."""
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        shard_ids = list(range(len(self._shards)))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(shard_ids)
        return shard_ids[worker_id::num_workers]

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def shard_info(self) -> list[dict]:
        return list(self._shards)

    # ── Constructors ───────────────────────────────────────────────────

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        blob_dir: str | Path,
        transform: Any = None,
        cache_capacity: int = 2,
    ) -> SODLDataset:
        """Create a dataset from a SODL manifest file.

        The manifest is a JSON file with a "shards" key containing a list
        of shard descriptors.
        """
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        shards = data["shards"]
        blob_store = BlobStore(str(blob_dir))
        return cls(shards, blob_store, transform, cache_capacity=cache_capacity)

    @classmethod
    def from_numpy(
        cls,
        arrays: Sequence[np.ndarray],
        blob_store: BlobStore,
        manifest_path: str | Path | None = None,
        transform: Any = None,
        cache_capacity: int = 2,
    ) -> SODLDataset:
        """Create a dataset by storing numpy arrays as SODL shards.

        Each array becomes one shard. Arrays are compressed and stored
        in the blob store with deduplication.

        Parameters
        ----------
        arrays : list[np.ndarray]
            Each array is one shard. First dimension is samples.
        blob_store : BlobStore
            Where to store the shard blobs.
        manifest_path : str, optional
            If provided, saves the manifest for later reloading.
        """
        compressor = zstd.ZstdCompressor(level=3)
        shards = []

        for arr in arrays:
            # Serialize
            buf = __import__("io").BytesIO()
            np.save(buf, arr, allow_pickle=False)
            raw = buf.getvalue()

            # Compress and store
            compressed = compressor.compress(raw)
            blob_id = compute_blob_id(compressed)
            blob_store.put(blob_id, compressed)

            shards.append({
                "blob_id": blob_id,
                "n_samples": arr.shape[0],
                "shape": list(arr.shape[1:]),
                "dtype": str(arr.dtype),
            })

        # Save manifest
        if manifest_path:
            manifest = {"shards": shards, "total_samples": sum(s["n_samples"] for s in shards)}
            Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
            Path(manifest_path).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        return cls(shards, blob_store, transform, cache_capacity=cache_capacity)
