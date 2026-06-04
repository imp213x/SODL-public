//! FastCDC — a fast content-defined chunking algorithm.
//!
//! This is a **gear-hash** implementation inspired by the FastCDC paper
//! (Wen Xia et al., 2016).  It produces variable-size chunks whose boundaries
//! are determined by the content itself, so inserting or removing bytes near
//! the beginning of a file only affects the chunk around the edit point —
//! all other chunks remain identical, enabling sub-file deduplication.
//!
//! # Parameters
//!
//! | Field | Default | Meaning |
//! |-------|---------|---------|
//! | `min_size` | 16 KiB | Minimum chunk size (below this, no boundary check) |
//! | `avg_size` | 64 KiB | Target average size (controls mask bit count) |
//! | `max_size` | 256 KiB | Hard maximum — force a cut regardless of hash |

use bytes::Bytes;

use crate::{ChunkDescriptor, Chunker};

/// Gear-hash lookup table (256 random u64 values).
///
/// Generated from a deterministic PRNG seeded with `42` so that all SODL
/// instances produce identical chunking for the same data.
static GEAR_TABLE: [u64; 256] = {
    // We use a simple xorshift64 seeded with 42 to generate the table
    // at compile time.
    let mut table = [0u64; 256];
    let mut state: u64 = 42;
    let mut i = 0;
    while i < 256 {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        table[i] = state;
        i += 1;
    }
    table
};

/// FastCDC chunker with configurable size bounds.
#[derive(Debug, Clone)]
pub struct FastCdcChunker {
    /// Minimum chunk size in bytes. No boundary is checked below this.
    pub min_size: u32,
    /// Target average chunk size. Used to derive the gear-hash mask.
    pub avg_size: u32,
    /// Maximum chunk size. A boundary is forced at this length.
    pub max_size: u32,
}

impl Default for FastCdcChunker {
    fn default() -> Self {
        Self {
            min_size: 16 * 1024,  // 16 KiB
            avg_size: 64 * 1024,  // 64 KiB
            max_size: 256 * 1024, // 256 KiB
        }
    }
}

impl FastCdcChunker {
    /// Create a chunker with custom size parameters.
    ///
    /// # Panics
    ///
    /// Panics if `min_size >= avg_size || avg_size >= max_size`.
    pub fn new(min_size: u32, avg_size: u32, max_size: u32) -> Self {
        assert!(
            min_size < avg_size && avg_size < max_size,
            "must satisfy min < avg < max; got {min_size} / {avg_size} / {max_size}"
        );
        Self {
            min_size,
            avg_size,
            max_size,
        }
    }

    /// Compute the mask from the target average size.
    ///
    /// The mask has `ceil(log2(avg_size)) - 1` low bits set.  This ensures
    /// that on average, the gear hash lands on a boundary once per
    /// `avg_size` bytes.
    fn mask(&self) -> u64 {
        let bits = 64 - (self.avg_size as u64).leading_zeros();
        (1u64 << (bits - 1)) - 1
    }

    /// Find the cut point within `data[min..max]` using the gear hash.
    ///
    /// Returns the offset *relative to the start of `data`* where the cut
    /// should happen.  If no boundary is found within `max`, returns `max`.
    fn cut_point(&self, data: &[u8]) -> usize {
        let len = data.len();
        let min = (self.min_size as usize).min(len);
        let max = (self.max_size as usize).min(len);
        let mask = self.mask();

        let mut hash: u64 = 0;
        let mut i = min;

        while i < max {
            hash = (hash << 1).wrapping_add(GEAR_TABLE[data[i] as usize]);
            if hash & mask == 0 {
                return i + 1; // cut after this byte
            }
            i += 1;
        }

        max // forced cut at max_size
    }
}

impl Chunker for FastCdcChunker {
    fn chunk(&self, data: &[u8]) -> Vec<ChunkDescriptor> {
        if data.is_empty() {
            return vec![ChunkDescriptor {
                offset: 0,
                length: 0,
                data: Bytes::new(),
            }];
        }

        let mut chunks = Vec::new();
        let mut offset = 0usize;

        while offset < data.len() {
            let remaining = &data[offset..];
            let cut = self.cut_point(remaining);
            chunks.push(ChunkDescriptor {
                offset: offset as u64,
                length: cut as u32,
                data: Bytes::copy_from_slice(&remaining[..cut]),
            });
            offset += cut;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_min_size() {
        let chunker = FastCdcChunker::new(100, 200, 400);
        let data = vec![0u8; 500];
        let chunks = chunker.chunk(&data);

        for c in &chunks[..chunks.len() - 1] {
            assert!(c.length >= 100, "non-final chunk too small: {}", c.length);
        }
    }

    #[test]
    fn respects_max_size() {
        let chunker = FastCdcChunker::new(100, 200, 400);
        let data = vec![42u8; 5000];
        let chunks = chunker.chunk(&data);

        for c in &chunks {
            assert!(c.length <= 400, "chunk too large: {}", c.length);
        }
    }

    #[test]
    fn full_coverage() {
        let chunker = FastCdcChunker::default();
        let data: Vec<u8> = (0u8..=255).cycle().take(1_000_000).collect();
        let chunks = chunker.chunk(&data);

        let total: u64 = chunks.iter().map(|c| c.length as u64).sum();
        assert_eq!(total, data.len() as u64, "chunks must cover full payload");

        // Verify contiguous offsets.
        let mut expected = 0u64;
        for c in &chunks {
            assert_eq!(c.offset, expected);
            expected += c.length as u64;
        }
    }

    #[test]
    fn deterministic() {
        let chunker = FastCdcChunker::default();
        let data: Vec<u8> = (0u8..=255).cycle().take(500_000).collect();

        let a = chunker.chunk(&data);
        let b = chunker.chunk(&data);

        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.offset, y.offset);
            assert_eq!(x.length, y.length);
        }
    }

    #[test]
    fn single_byte_change_localises_boundary_shift() {
        let chunker = FastCdcChunker::new(1024, 4096, 16384);
        let mut data: Vec<u8> = (0u8..=255).cycle().take(200_000).collect();
        let original = chunker.chunk(&data);

        // Flip one byte near the middle.
        data[100_000] ^= 0xFF;
        let modified = chunker.chunk(&data);

        // Most chunks should be identical — only the one containing the edit
        // and possibly its immediate neighbour should differ.
        let same = original
            .iter()
            .zip(modified.iter())
            .filter(|(a, b)| a.offset == b.offset && a.length == b.length)
            .count();

        let pct = (same as f64) / (original.len() as f64);
        assert!(
            pct > 0.5,
            "expected >50% of chunks unchanged, got {:.1}%",
            pct * 100.0
        );
    }

    #[test]
    fn empty_data() {
        let chunker = FastCdcChunker::default();
        let chunks = chunker.chunk(b"");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].length, 0);
    }
}
