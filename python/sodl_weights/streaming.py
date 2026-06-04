"""Streaming Compression — chunk-wise zstd for large data.

Provides streaming compressor/decompressor for processing data larger
than available memory, or for incremental compression during training.

Example
-------
>>> sc = StreamingCompressor(chunk_size=1024*1024)
>>> for chunk in data_stream:
...     compressed_chunks = sc.feed(chunk)
...     for cc in compressed_chunks:
...         output.write(cc)
>>> final = sc.finish()
"""

from __future__ import annotations

import io
from typing import Iterator

import zstandard as zstd


class StreamingCompressor:
    """Chunk-wise zstd compression for streaming data.

    Accepts data in chunks and emits compressed output incrementally.
    Useful for compressing large model weights or datasets that don't
    fit in memory.

    Parameters
    ----------
    level : int
        Zstandard compression level (default 3).
    chunk_size : int
        Output chunk size in bytes (default 256KB).
    """

    def __init__(self, level: int = 3, chunk_size: int = 256 * 1024) -> None:
        self._compressor = zstd.ZstdCompressor(level=level)
        self._chunk_size = chunk_size
        self._accumulated = io.BytesIO()
        self._total_input = 0
        self._total_output = 0
        self._emitted_chunks: list[bytes] = []

    def feed(self, data: bytes) -> list[bytes]:
        """Feed input data and get compressed chunks.

        Data is accumulated internally. When the accumulated buffer reaches
        chunk_size, it is compressed and emitted as an output chunk.

        Parameters
        ----------
        data : bytes
            Input data chunk.

        Returns
        -------
        list of bytes
            Zero or more compressed output chunks.
        """
        self._total_input += len(data)
        self._accumulated.write(data)

        chunks = []
        if self._accumulated.tell() >= self._chunk_size:
            compressed = self._compressor.compress(self._accumulated.getvalue())
            self._total_output += len(compressed)
            chunks.append(compressed)
            self._emitted_chunks.append(compressed)
            self._accumulated = io.BytesIO()

        return chunks

    def finish(self) -> bytes:
        """Flush remaining data and finalize the stream.

        Returns
        -------
        bytes
            Final compressed output (may be empty if all data was already emitted).
        """
        remaining = self._accumulated.getvalue()
        self._accumulated = io.BytesIO()

        if not remaining and not self._emitted_chunks:
            return b""

        if remaining:
            compressed = self._compressor.compress(remaining)
            self._total_output += len(compressed)
            return compressed

        return b""

    @property
    def total_input(self) -> int:
        return self._total_input

    @property
    def total_output(self) -> int:
        return self._total_output

    @property
    def ratio(self) -> float:
        if self._total_input == 0:
            return 0.0
        return 1.0 - (self._total_output / self._total_input)


class StreamingDecompressor:
    """Chunk-wise zstd decompression for streaming data.

    Parameters
    ----------
    chunk_size : int
        Output chunk size in bytes (default 256KB).
    """

    def __init__(self, chunk_size: int = 256 * 1024) -> None:
        self._decompressor = zstd.ZstdDecompressor()
        self._chunk_size = chunk_size
        self._total_input = 0
        self._total_output = 0

    def decompress_stream(self, data: bytes) -> Iterator[bytes]:
        """Decompress data and yield output chunks.

        Parameters
        ----------
        data : bytes
            Compressed data (full zstd frame).

        Yields
        ------
        bytes
            Decompressed output chunks.
        """
        self._total_input += len(data)
        reader = self._decompressor.stream_reader(io.BytesIO(data))

        while True:
            chunk = reader.read(self._chunk_size)
            if not chunk:
                break
            self._total_output += len(chunk)
            yield chunk

        reader.close()

    def decompress_all(self, data: bytes) -> bytes:
        """Decompress all data at once.

        Parameters
        ----------
        data : bytes
            Compressed data.

        Returns
        -------
        bytes
            Full decompressed output.
        """
        self._total_input += len(data)
        result = self._decompressor.decompress(data)
        self._total_output += len(result)
        return result

    @property
    def total_input(self) -> int:
        return self._total_input

    @property
    def total_output(self) -> int:
        return self._total_output


def compress_file(input_path: str, output_path: str, level: int = 3, chunk_size: int = 1024 * 1024) -> dict:
    """Compress a file using streaming zstd.

    Parameters
    ----------
    input_path : str
        Path to input file.
    output_path : str
        Path for compressed output.
    level : int
        Compression level.
    chunk_size : int
        Read chunk size (default 1MB).

    Returns
    -------
    dict
        Stats: input_bytes, output_bytes, ratio.
    """
    import os
    compressor = zstd.ZstdCompressor(level=level)
    input_bytes = os.path.getsize(input_path)

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        compressor.copy_stream(fin, fout, read_size=chunk_size)

    output_bytes = os.path.getsize(output_path)
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "ratio": 1.0 - (output_bytes / max(input_bytes, 1)),
    }


def decompress_file(input_path: str, output_path: str, chunk_size: int = 1024 * 1024) -> dict:
    """Decompress a zstd-compressed file.

    Parameters
    ----------
    input_path : str
        Path to compressed file.
    output_path : str
        Path for decompressed output.

    Returns
    -------
    dict
        Stats: input_bytes, output_bytes.
    """
    import os
    decompressor = zstd.ZstdDecompressor()

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        decompressor.copy_stream(fin, fout, read_size=chunk_size)

    return {
        "input_bytes": os.path.getsize(input_path),
        "output_bytes": os.path.getsize(output_path),
    }
