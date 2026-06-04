"""SODL Fast BPE — Python FFI wrapper for sodl-tokenizer Rust crate.

Calls the Rust-accelerated BPE training engine via ctypes for
50-200x speedup over pure-Python CognitiveTokenizer.train().

Usage::

    from sodl_weights.fast_bpe import rust_train_bpe

    merges, vocab = rust_train_bpe(
        text=b"hello world hello world",
        vocab_size=32768,
        special_token_count=21,
    )
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path


def _find_dll() -> Path | None:
    """Locate sodl_tokenizer shared library."""
    system = platform.system()
    if system == "Windows":
        lib_name = "sodl_tokenizer.dll"
    elif system == "Darwin":
        lib_name = "libsodl_tokenizer.dylib"
    else:
        lib_name = "libsodl_tokenizer.so"

    # Search order:
    # 1. SODL_LIB_DIR env var
    # 2. Relative to this file: ../../target/release/
    # 3. SODL workspace target/release/
    search_paths = []

    env_dir = os.environ.get("SODL_LIB_DIR")
    if env_dir:
        search_paths.append(Path(env_dir))

    # Relative to sodl_weights package → SODL/python/sodl_weights/fast_bpe.py
    pkg_dir = Path(__file__).resolve().parent
    search_paths.extend([
        pkg_dir.parent.parent / "target" / "release",          # SODL/target/release
        pkg_dir.parent.parent.parent / "target" / "release",   # SODL/target/release (if nested)
    ])

    # Common developer paths
    for dev_root in [Path.cwd(), Path.home() / "SODL"]:
        search_paths.append(dev_root / "target" / "release")

    for search_dir in search_paths:
        candidate = search_dir / lib_name
        if candidate.exists():
            return candidate

    return None


def _load_lib() -> ctypes.CDLL | None:
    """Load the sodl_tokenizer shared library."""
    dll_path = _find_dll()
    if dll_path is None:
        return None
    try:
        lib = ctypes.CDLL(str(dll_path))
        # sodl_bpe_train(text_ptr, text_len, vocab_size, special_count, verbose) -> *c_char
        lib.sodl_bpe_train.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        lib.sodl_bpe_train.restype = ctypes.c_char_p
        # sodl_free_string(ptr)
        lib.sodl_free_string.argtypes = [ctypes.c_char_p]
        lib.sodl_free_string.restype = None
        return lib
    except OSError:
        return None


# Lazy-load the library
_LIB: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL | None:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


def is_available() -> bool:
    """Check if Rust BPE backend is available."""
    return _get_lib() is not None


def rust_train_bpe(
    text: bytes,
    vocab_size: int = 32768,
    special_token_count: int = 21,
    verbose: bool = True,
) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Train BPE using the Rust engine.

    Parameters
    ----------
    text : bytes
        Raw UTF-8 text to train on.
    vocab_size : int
        Target vocabulary size.
    special_token_count : int
        Number of reserved special token IDs (0..special_token_count).
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    tuple[dict, dict]
        (merges, vocab) where:
        - merges: {(token_a, token_b): merged_id}
        - vocab: {token_id: byte_sequence}

    Raises
    ------
    RuntimeError
        If the Rust library is not available or returns an error.
    """
    lib = _get_lib()
    if lib is None:
        raise RuntimeError(
            "sodl-tokenizer Rust library not found. "
            "Build with: cargo build -p sodl-tokenizer --release"
        )

    result_ptr = lib.sodl_bpe_train(
        text,
        len(text),
        vocab_size,
        special_token_count,
        1 if verbose else 0,
    )

    if result_ptr is None:
        raise RuntimeError("sodl_bpe_train returned null — BPE training failed")

    result_json = result_ptr.decode("utf-8")
    result = json.loads(result_json)

    # Convert merges: [(a, b, merged_id), ...] -> {(a, b): merged_id}
    merges = {}
    for a, b, merged_id in result["merges"]:
        merges[(a, b)] = merged_id

    # Convert vocab: {"id": [byte, ...]} -> {id: bytes}
    vocab = {}
    for str_id, byte_list in result["vocab"].items():
        vocab[int(str_id)] = bytes(byte_list)

    return merges, vocab
