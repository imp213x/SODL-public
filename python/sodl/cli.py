"""Minimal command-line utilities for the SODL Python SDK."""

from __future__ import annotations

import argparse

from sodl import __version__


def _doctor() -> int:
    from sodl import rust_bridge_summary

    print(f"sodl {__version__}")
    print(rust_bridge_summary())
    return 0


def _blob_id(text: str) -> int:
    from sodl import compute_blob_id

    print(compute_blob_id(text.encode("utf-8")))
    return 0


def _blob_count(root: str) -> int:
    from sodl import BlobStore

    store = BlobStore(root)
    print(store.blob_count())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SODL Python SDK utilities")
    parser.add_argument("--version", action="store_true", help="Print the installed package version")

    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Print package diagnostics")
    doctor.set_defaults(func=lambda args: _doctor())

    blob_id = subparsers.add_parser("blob-id", help="Compute the content-addressed id for text")
    blob_id.add_argument("text", help="Text payload")
    blob_id.set_defaults(func=lambda args: _blob_id(args.text))

    blob_count = subparsers.add_parser("blob-count", help="Count locally stored blobs")
    blob_count.add_argument("root", help="Blob store root directory")
    blob_count.set_defaults(func=lambda args: _blob_count(args.root))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args))
