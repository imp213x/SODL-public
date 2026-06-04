"""ClusteredSoftmax — hierarchical softmax for SODL-accelerated training.

Replaces the standard O(V) cross-entropy loss with a two-level hierarchical
loss: O(K + V/K), where K = number of clusters.

Works with any HuggingFace CausalLM model. Non-invasive: wraps the existing
model without modifying its weights or architecture.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sodl_weights.token_hash import TokenHashIndex


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for ClusteredSoftmax. Install with: pip install torch")


def _module_weight(module: Any) -> "torch.Tensor | None":
    weight = getattr(module, "weight", None)
    return weight if HAS_TORCH and isinstance(weight, torch.Tensor) else None


def _resolve_weight_tensor(
    source: Any,
    *,
    role: str,
    explicit_weight: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Resolve an input/output weight tensor from a model, module, or tensor."""
    _require_torch()
    if explicit_weight is not None:
        return explicit_weight

    if isinstance(source, torch.Tensor):
        return source

    direct_weight = _module_weight(source)
    if direct_weight is not None:
        return direct_weight

    if role == "input":
        getter_candidates = ("get_input_embeddings",)
        attr_candidates = ("input_embeddings", "token_embeddings", "embeddings")
        exact_named_candidates = ("embed_tokens.weight", "wte.weight", "embedding.weight")
        fuzzy_terms = ("embed", "token", "input")
    elif role == "output":
        getter_candidates = ("get_output_embeddings",)
        attr_candidates = ("lm_head", "output_embeddings", "output_projection")
        exact_named_candidates = ("lm_head.weight", "output_projection.weight", "embed_tokens.weight", "wte.weight")
        fuzzy_terms = ("lm_head", "output", "projection", "embed")
    else:  # pragma: no cover
        raise ValueError(f"unknown weight role: {role}")

    for getter_name in getter_candidates:
        getter = getattr(source, getter_name, None)
        if callable(getter):
            module = getter()
            weight = _module_weight(module)
            if weight is not None:
                return weight

    for attr_name in attr_candidates:
        module = getattr(source, attr_name, None)
        weight = _module_weight(module)
        if weight is not None:
            return weight

    if hasattr(source, "named_parameters"):
        params = list(source.named_parameters())
        for candidate in exact_named_candidates:
            for name, param in params:
                if name == candidate and isinstance(param, torch.Tensor):
                    return param
        for name, param in params:
            lowered = name.lower()
            if "weight" in lowered and any(term in lowered for term in fuzzy_terms):
                return param

    if role == "output":
        return _resolve_weight_tensor(source, role="input")

    raise ValueError(
        f"Could not resolve {role} weight tensor. "
        "Provide an explicit tensor/module or use a model exposing embedding accessors."
    )


class ClusteredSoftmaxLoss(nn.Module):
    """Hierarchical cross-entropy loss using SODL token clusters.

    Two-level computation:
      1. Cluster-level:  P(cluster | hidden)  via centroid logits
      2. Token-level:    P(token | cluster, hidden)  via full embeddings within cluster
      3. Combined:       loss = -log P(cluster) - log P(token|cluster)

    This decomposes the standard cross-entropy into two smaller softmaxes,
    reducing computation from O(V) to O(K + max_cluster_size).

    Parameters
    ----------
    token_index : TokenHashIndex
        Pre-built token hash index with cluster assignments.
    embedding_weight : torch.Tensor
        The model's output embedding weight matrix, shape (V, D).
    top_k_clusters : int
        Number of clusters to evaluate per token (default 3).
        Higher = more accurate but slower.
    """

    def __init__(
        self,
        token_index: TokenHashIndex,
        embedding_weight: "torch.Tensor",
        top_k_clusters: int = 3,
    ) -> None:
        _require_torch()
        super().__init__()

        self._index = token_index
        self._top_k = top_k_clusters
        self._vocab_size = token_index.vocab_size
        self._n_clusters = token_index.n_clusters
        self._dim = token_index.dim

        # Pre-compute cluster centroids as a parameter (no grad — updated periodically)
        centroids_np = token_index._centroids.astype(np.float32)
        self.register_buffer(
            "cluster_centroids",
            torch.from_numpy(centroids_np),  # (K, D)
        )

        # Store reference to the model's embedding weight (shared, not copied)
        self._embedding_weight = embedding_weight  # (V, D)

        # Pre-compute cluster membership as padded tensor for batched lookup
        max_cluster_size = max(
            len(members) for members in token_index._cluster_members.values()
        )
        self._max_cluster_size = max_cluster_size

        # Build membership index: (K, max_size) with -1 padding
        membership = torch.full(
            (self._n_clusters, max_cluster_size), -1, dtype=torch.long
        )
        for cid, members in token_index._cluster_members.items():
            membership[cid, :len(members)] = torch.tensor(members, dtype=torch.long)
        self.register_buffer("cluster_membership", membership)

        # Build reverse index: token_id → (cluster_id, position_in_cluster)
        token_to_cluster = torch.zeros(self._vocab_size, dtype=torch.long)
        token_to_position = torch.zeros(self._vocab_size, dtype=torch.long)
        for cid, members in token_index._cluster_members.items():
            for pos, tid in enumerate(members):
                token_to_cluster[tid] = cid
                token_to_position[tid] = pos
        self.register_buffer("token_to_cluster", token_to_cluster)
        self.register_buffer("token_to_position", token_to_position)

    def forward(
        self,
        hidden_states: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute hierarchical cross-entropy loss.

        Parameters
        ----------
        hidden_states : (batch, seq_len, dim)
            Model's last hidden states.
        labels : (batch, seq_len)
            Target token IDs. -100 = ignore.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        batch_size, seq_len, dim = hidden_states.shape

        # Flatten
        h_flat = hidden_states.reshape(-1, dim)         # (B*S, D)
        labels_flat = labels.reshape(-1)                  # (B*S,)

        # Mask valid positions (labels != -100)
        valid_mask = labels_flat != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        h_valid = h_flat[valid_mask]                   # (N, D)
        labels_valid = labels_flat[valid_mask]          # (N,)
        n_valid = h_valid.shape[0]

        # --- Level 1: Cluster-level cross-entropy ---
        # Compute logits over cluster centroids
        centroids = self.cluster_centroids.to(device=h_valid.device, dtype=h_valid.dtype)
        cluster_logits = h_valid @ centroids.t()  # (N, K)

        # Get target cluster for each label
        target_clusters = self.token_to_cluster[labels_valid]   # (N,)

        cluster_loss = F.cross_entropy(cluster_logits, target_clusters)

        # --- Level 2: Token-level cross-entropy within target cluster ---
        # Group by target cluster so memory scales with actual cluster size,
        # not with N * max_cluster_size * D for the whole batch.
        total_token_loss_sum = torch.zeros((), device=h_valid.device, dtype=h_valid.dtype)
        total_token_count = 0
        unique_clusters = torch.unique(target_clusters)

        for cluster_id in unique_clusters:
            cluster_mask = target_clusters == cluster_id
            if not cluster_mask.any():
                continue

            cluster_hidden = h_valid[cluster_mask]                      # (Nc, D)
            cluster_labels = labels_valid[cluster_mask]                 # (Nc,)
            member_ids = self.cluster_membership[cluster_id]
            valid_members = member_ids[member_ids >= 0]
            if valid_members.numel() == 0:
                continue

            member_embeddings = self._embedding_weight[valid_members].to(
                device=h_valid.device,
                dtype=h_valid.dtype,
            )  # (Mc, D)
            within_logits = cluster_hidden @ member_embeddings.t()      # (Nc, Mc)
            target_positions = self.token_to_position[cluster_labels]   # (Nc,)
            total_token_loss_sum = total_token_loss_sum + F.cross_entropy(
                within_logits,
                target_positions,
                reduction="sum",
            )
            total_token_count += int(cluster_labels.shape[0])

        if total_token_count <= 0:
            total_token_loss = torch.zeros((), device=h_valid.device, dtype=h_valid.dtype)
        else:
            total_token_loss = total_token_loss_sum / float(total_token_count)

        # Combined loss: both levels must be correct
        return cluster_loss + total_token_loss


def build_index_from_model(
    model: Any,
    n_clusters: int = 512,
    top_k: int = 3,
    build_max_iter: int = 50,
    fit_sample_size: Optional[int] = None,
    build_batch_size: Optional[int] = None,
    build_n_init: int = 3,
    adaptive: bool = False,
    max_cluster_ratio: float = 5.0,
    uniform_ratio_threshold: float = 1.5,
    max_rebalance_steps: int = 8,
) -> tuple[TokenHashIndex, dict]:
    """Build a TokenHashIndex from a model, module, or explicit embedding weight.

    Parameters
    ----------
    model : nn.Module | nn.Embedding | nn.Linear | torch.Tensor
        Source of the input embeddings. Supports explicit tensors, embedding
        modules, or models exposing standard embedding accessors.
    n_clusters : int
        Number of clusters.
    top_k : int
        Top-k clusters for hierarchical softmax.

    Returns
    -------
    (TokenHashIndex, build_stats)
    """
    _require_torch()

    embed_weight = _resolve_weight_tensor(model, role="input")

    # NumPy cannot consume bf16 tensors directly on this path, so normalize first.
    embeddings_np = embed_weight.detach().to(dtype=torch.float32).cpu().numpy()

    index = TokenHashIndex(n_clusters=n_clusters, top_k_clusters=top_k)
    stats = index.build(
        embeddings_np,
        max_iter=build_max_iter,
        fit_sample_size=fit_sample_size,
        batch_size=build_batch_size,
        n_init=build_n_init,
        adaptive=adaptive,
        max_cluster_ratio=max_cluster_ratio,
        uniform_ratio_threshold=uniform_ratio_threshold,
        max_rebalance_steps=max_rebalance_steps,
    )

    return index, stats


def create_clustered_loss(
    model: Any,
    n_clusters: int = 512,
    top_k: int = 3,
    build_max_iter: int = 50,
    fit_sample_size: Optional[int] = None,
    build_batch_size: Optional[int] = None,
    build_n_init: int = 3,
    adaptive: bool = False,
    max_cluster_ratio: float = 5.0,
    uniform_ratio_threshold: float = 1.5,
    max_rebalance_steps: int = 8,
    *,
    embedding_weight: "torch.Tensor | None" = None,
    output_weight: "torch.Tensor | None" = None,
) -> tuple["ClusteredSoftmaxLoss", TokenHashIndex, dict]:
    """End-to-end: build index from model → create ClusteredSoftmaxLoss.

    Returns
    -------
    (loss_module, token_index, build_stats)
    """
    _require_torch()

    index, stats = build_index_from_model(
        embedding_weight if embedding_weight is not None else model,
        n_clusters,
        top_k,
        build_max_iter=build_max_iter,
        fit_sample_size=fit_sample_size,
        build_batch_size=build_batch_size,
        build_n_init=build_n_init,
        adaptive=adaptive,
        max_cluster_ratio=max_cluster_ratio,
        uniform_ratio_threshold=uniform_ratio_threshold,
        max_rebalance_steps=max_rebalance_steps,
    )

    lm_head_weight = _resolve_weight_tensor(
        model,
        role="output",
        explicit_weight=output_weight if output_weight is not None else embedding_weight,
    )

    loss_fn = ClusteredSoftmaxLoss(index, lm_head_weight, top_k)
    loss_fn = loss_fn.to(lm_head_weight.device)

    return loss_fn, index, stats
