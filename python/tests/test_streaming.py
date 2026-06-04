import pytest
import tempfile
from pathlib import Path

from sodl_weights.streaming import (
    StreamingCompressor,
    StreamingDecompressor,
    compress_file,
    decompress_file,
)


class TestStreamingCompressor:
    def test_basic_compress(self):
        sc = StreamingCompressor(chunk_size=100)
        data = b"Hello world! " * 100
        
        all_output = b""
        chunks = sc.feed(data)
        for c in chunks:
            all_output += c
        all_output += sc.finish()
        
        assert len(all_output) > 0
        assert len(all_output) < len(data)

    def test_multi_feed(self):
        sc = StreamingCompressor(chunk_size=256)
        
        feeds = [b"chunk-" + bytes([i]) * 200 for i in range(10)]
        all_output = b""
        for feed_data in feeds:
            for chunk in sc.feed(feed_data):
                all_output += chunk
        all_output += sc.finish()
        
        assert sc.total_input == sum(len(f) for f in feeds)
        assert sc.total_output > 0

    def test_ratio_property(self):
        sc = StreamingCompressor()
        data = b"A" * 10000
        sc.feed(data)
        sc.finish()
        assert sc.ratio > 0  # should compress well


class TestStreamingDecompressor:
    def test_decompress_stream(self):
        import zstandard as zstd
        
        original = b"Streaming decompression test! " * 500
        compressed = zstd.ZstdCompressor().compress(original)
        
        sd = StreamingDecompressor(chunk_size=100)
        chunks = list(sd.decompress_stream(compressed))
        result = b"".join(chunks)
        assert result == original

    def test_decompress_all(self):
        import zstandard as zstd
        
        original = b"All at once decompression"
        compressed = zstd.ZstdCompressor().compress(original)
        
        sd = StreamingDecompressor()
        result = sd.decompress_all(compressed)
        assert result == original

    def test_stats_tracking(self):
        import zstandard as zstd
        
        original = b"Stats tracking test data"
        compressed = zstd.ZstdCompressor().compress(original)
        
        sd = StreamingDecompressor()
        sd.decompress_all(compressed)
        assert sd.total_input == len(compressed)
        assert sd.total_output == len(original)


class TestFileCompression:
    def test_compress_decompress_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "input.bin")
            compressed_path = str(Path(tmpdir) / "compressed.zst")
            output_path = str(Path(tmpdir) / "output.bin")
            
            # Write test data
            data = b"File compression roundtrip! " * 1000
            with open(input_path, "wb") as f:
                f.write(data)
            
            # Compress
            stats_c = compress_file(input_path, compressed_path)
            assert stats_c["ratio"] > 0
            assert stats_c["output_bytes"] < stats_c["input_bytes"]
            
            # Decompress
            stats_d = decompress_file(compressed_path, output_path)
            assert stats_d["output_bytes"] == len(data)
            
            # Verify roundtrip
            with open(output_path, "rb") as f:
                result = f.read()
            assert result == data
