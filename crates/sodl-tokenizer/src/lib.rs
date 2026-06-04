//! SODL Tokenizer — Word-Frequency BPE Training Engine
//!
//! Phase A: Pre-tokenize corpus into words, count word frequencies,
//! then run BPE merges on unique words weighted by frequency.
//!
//! This reduces 191M chars → ~500K unique words, giving 500-2000x speedup
//! over naive byte-level BPE.
//!
//! Exposed to Python via C FFI (cdylib).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of BPE training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTrainResult {
    /// Merge rules: (token_a, token_b) -> merged_token_id
    pub merges: Vec<(u32, u32, u32)>,
    /// Vocabulary: token_id -> byte sequence
    pub vocab: HashMap<u32, Vec<u8>>,
    /// Number of merges performed
    pub num_merges: usize,
}

/// A word with its byte-token representation and corpus frequency.
#[derive(Debug, Clone)]
struct WordEntry {
    /// Current token IDs for this word
    tokens: Vec<u32>,
    /// How many times this word appears in the corpus
    frequency: u64,
}

/// Pre-tokenize text into words by splitting on whitespace and punctuation boundaries.
/// Returns (word_string, frequency) pairs.
fn pre_tokenize(text: &[u8]) -> HashMap<Vec<u8>, u64> {
    let mut word_freqs: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut current_word: Vec<u8> = Vec::new();

    for &byte in text {
        let is_boundary = byte == b' '
            || byte == b'\n'
            || byte == b'\r'
            || byte == b'\t'
            || byte == b'('
            || byte == b')'
            || byte == b'['
            || byte == b']'
            || byte == b'{'
            || byte == b'}'
            || byte == b','
            || byte == b';'
            || byte == b':'
            || byte == b'!'
            || byte == b'?'
            || byte == b'"'
            || byte == b'\''
            || byte == b'`';

        if is_boundary {
            if !current_word.is_empty() {
                *word_freqs.entry(current_word.clone()).or_insert(0) += 1;
                current_word.clear();
            }
            // The boundary character itself is a "word" too
            *word_freqs.entry(vec![byte]).or_insert(0) += 1;
        } else {
            current_word.push(byte);
        }
    }
    // Don't forget the last word
    if !current_word.is_empty() {
        *word_freqs.entry(current_word).or_insert(0) += 1;
    }

    word_freqs
}

/// Count pair frequencies across all words, weighted by word frequency.
fn count_weighted_pairs(words: &[WordEntry]) -> HashMap<(u32, u32), u64> {
    let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
    for entry in words {
        if entry.tokens.len() < 2 {
            continue;
        }
        for window in entry.tokens.windows(2) {
            *counts.entry((window[0], window[1])).or_insert(0) += entry.frequency;
        }
    }
    counts
}

/// Merge a specific pair in a single word's token list.
fn merge_word_tokens(tokens: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(new_id);
            i += 2;
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }
    result
}

/// Train BPE using word-frequency pre-tokenization.
///
/// # Arguments
/// * `text` — raw UTF-8 bytes
/// * `vocab_size` — target vocabulary size
/// * `special_token_count` — IDs 0..special_token_count are reserved
/// * `verbose` — print progress
pub fn train_bpe(
    text: &[u8],
    vocab_size: usize,
    special_token_count: usize,
    verbose: bool,
) -> BpeTrainResult {
    let num_merges = vocab_size.saturating_sub(256 + special_token_count);
    let offset = special_token_count as u32;

    // ── Step 1: Pre-tokenize into words with frequencies ──
    let word_freqs = pre_tokenize(text);
    let unique_words = word_freqs.len();

    if verbose {
        eprintln!(
            "sodl-tokenizer: pre-tokenized {} bytes → {} unique words",
            text.len(),
            unique_words,
        );
    }

    // ── Step 2: Convert each word to initial byte tokens ──
    let mut words: Vec<WordEntry> = word_freqs
        .into_iter()
        .map(|(word_bytes, freq)| WordEntry {
            tokens: word_bytes.iter().map(|&b| b as u32 + offset).collect(),
            frequency: freq,
        })
        .collect();

    // Build initial vocabulary (256 byte tokens shifted past special tokens)
    let mut vocab: HashMap<u32, Vec<u8>> = HashMap::with_capacity(vocab_size);
    for byte_val in 0u16..256 {
        vocab.insert(byte_val as u32 + offset, vec![byte_val as u8]);
    }

    // ── Step 3: Iterative BPE merges on UNIQUE WORDS (weighted by freq) ──
    let mut merges: Vec<(u32, u32, u32)> = Vec::with_capacity(num_merges);
    let mut next_id = 256u32 + offset;

    for i in 0..num_merges {
        // Count pairs across all unique words, weighted by word frequency
        let counts = count_weighted_pairs(&words);
        if counts.is_empty() {
            break;
        }

        // Find most frequent pair
        let (&best_pair, &best_count) = counts.iter().max_by_key(|(_, &c)| c).unwrap();

        if best_count < 2 {
            if verbose {
                eprintln!(
                    "sodl-tokenizer: stopping at merge {}/{} (no pair freq >= 2)",
                    i, num_merges
                );
            }
            break;
        }

        let new_id = next_id;
        next_id += 1;

        // Merge this pair in ALL words that contain it
        for entry in words.iter_mut() {
            if entry.tokens.len() >= 2 {
                entry.tokens = merge_word_tokens(&entry.tokens, best_pair, new_id);
            }
        }

        // Build merged byte sequence for vocab
        let mut merged_bytes = vocab.get(&best_pair.0).cloned().unwrap_or_default();
        merged_bytes.extend(vocab.get(&best_pair.1).cloned().unwrap_or_default());
        vocab.insert(new_id, merged_bytes);

        merges.push((best_pair.0, best_pair.1, new_id));

        if verbose && (i % 1000 == 0 || i == num_merges - 1) {
            let total_tokens: usize = words.iter().map(|w| w.tokens.len()).sum();
            eprintln!(
                "sodl-tokenizer: merge {}/{} ({:.1}%) freq={} total_word_tokens={}",
                i,
                num_merges,
                (i as f64 / num_merges as f64) * 100.0,
                best_count,
                total_tokens,
            );
        }
    }

    let num_merges_done = merges.len();

    if verbose {
        eprintln!(
            "sodl-tokenizer: done — {} merges, vocab_size={}, unique_words={}",
            num_merges_done,
            vocab.len(),
            unique_words,
        );
    }

    BpeTrainResult {
        merges,
        num_merges: num_merges_done,
        vocab,
    }
}

// ── C FFI ──────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sodl_bpe_train(
    text_ptr: *const u8,
    text_len: usize,
    vocab_size: u32,
    special_token_count: u32,
    verbose: i32,
) -> *mut std::os::raw::c_char {
    let text = std::slice::from_raw_parts(text_ptr, text_len);
    let result = train_bpe(
        text,
        vocab_size as usize,
        special_token_count as usize,
        verbose != 0,
    );

    match serde_json::to_string(&result) {
        Ok(json) => {
            let c_str = std::ffi::CString::new(json).unwrap_or_default();
            c_str.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn sodl_free_string(ptr: *mut std::os::raw::c_char) {
    if !ptr.is_null() {
        drop(std::ffi::CString::from_raw(ptr));
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_tokenize() {
        let text = b"hello world hello";
        let freqs = pre_tokenize(text);
        assert_eq!(freqs.get(b"hello" as &[u8]), Some(&2));
        assert_eq!(freqs.get(b"world" as &[u8]), Some(&1));
        assert_eq!(freqs.get(b" " as &[u8]), Some(&2));
    }

    #[test]
    fn test_pre_tokenize_code() {
        let text = b"def foo(x): return x + 1";
        let freqs = pre_tokenize(text);
        assert!(freqs.contains_key(b"def" as &[u8]));
        assert!(freqs.contains_key(b"foo" as &[u8]));
        assert!(freqs.contains_key(b"(" as &[u8]));
        assert!(freqs.contains_key(b"x" as &[u8]));
    }

    #[test]
    fn test_basic_bpe() {
        let text = b"aaabdaaabac";
        let result = train_bpe(text, 259, 0, false);
        assert!(result.num_merges >= 1);
        assert!(result.vocab.len() > 256);
    }

    #[test]
    fn test_with_special_tokens() {
        let text = b"hello world hello world";
        let result = train_bpe(text, 280, 21, false);
        assert!(result.num_merges >= 1);
    }

    #[test]
    fn test_empty_text() {
        let text = b"";
        let result = train_bpe(text, 280, 21, false);
        assert_eq!(result.num_merges, 0);
    }

    #[test]
    fn test_large_repetitive() {
        let text = "the quick brown fox ".repeat(10000);
        let result = train_bpe(text.as_bytes(), 300, 0, false);
        assert!(result.num_merges >= 5);
    }

    #[test]
    fn test_word_frequency_efficiency() {
        // 1M chars of repeated text — should be fast because only ~10 unique words
        let text = "hello world foo bar baz qux ".repeat(40000);
        let result = train_bpe(text.as_bytes(), 300, 21, true);
        assert!(result.num_merges >= 1);
    }

    #[test]
    fn test_code_corpus() {
        let text = "def main():\n    print('hello')\n\ndef test():\n    assert True\n".repeat(1000);
        let result = train_bpe(text.as_bytes(), 320, 21, false);
        assert!(result.num_merges >= 1);
    }
}
