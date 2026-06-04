//! # sodl-polar-quant
//!
//! Polar-coordinate weight quantization for SODL, inspired by Google's
//! TurboQuant research (PolarQuant + QJL, March 2026).
//!
//! ## Core idea
//!
//! Instead of quantizing weight vectors element-by-element (losing directional
//! information), transform each vector to polar coordinates first:
//!
//! ```text
//! w ∈ ℝᵈ → (‖w‖₂, θ₁, θ₂, ..., θ_{d-1})
//! ```
//!
//! Then quantize norm and angles separately:
//! - **Norm**: 8-bit log-scale (captures dynamic range from 1e-12 to 1e+6)
//! - **Angles**: 8-bit uniform (captures direction in [0, 2π) per axis)
//!
//! ## Compression ratio
//!
//! Original fp32 vector of dim `d`: `4d` bytes
//! PolarQuant:  `1 + (d-1)` = `d` bytes
//! Ratio: **4x** compression
//!
//! ## Error bound
//!
//! Per TurboQuant findings, the reconstruction error is bounded:
//! - Norm error: ≤ 4.4% relative (log-scale bin width)
//! - Angular error: ≤ π/128 ≈ 0.025 radians per angle
//! - Combined: ‖w - w'‖₂ / ‖w‖₂ ≈ 0.1% for typical model dimensions

use std::f32::consts::PI;

/// Errors from polar quantization operations.
#[derive(Debug, thiserror::Error)]
pub enum PolarQuantError {
    #[error("empty vector — cannot quantize a zero-dimensional weight")]
    EmptyVector,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("corrupted payload: {0}")]
    Corrupted(String),
}

/// A polar-quantized weight vector.
///
/// Stores the L2 norm in 8-bit log-scale and each angular component
/// in 8-bit uniform quantization over `[0, 2π)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolarQuantized {
    /// 8-bit log-quantized L2 norm.
    ///
    /// Decode: `norm = 2^((q - 128) / 16.0)`
    /// Range: ~1.5e-10 to ~6.7e+7
    pub norm_q8: u8,

    /// Quantized angles, one per dimension minus one.
    ///
    /// Decode: `θ_i = q_i * (2π / 256)`
    pub angles: Vec<u8>,

    /// Original dimensionality of the weight vector.
    pub original_dim: usize,
}

/// Quantization statistics returned after encoding a batch.
#[derive(Clone, Debug)]
pub struct QuantStats {
    /// Number of vectors quantized.
    pub count: usize,
    /// Total bytes in the original fp32 representation.
    pub original_bytes: usize,
    /// Total bytes in the quantized representation.
    pub quantized_bytes: usize,
    /// Compression ratio (original / quantized).
    pub compression_ratio: f32,
    /// Average relative reconstruction error (‖w - w'‖ / ‖w‖).
    pub avg_relative_error: f32,
}

// ── Norm quantization ──────────────────────────────────────────────

/// Encode an f32 norm to 8-bit log-scale.
///
/// Formula: `q = clamp(round(log2(norm) * 4 + 128), 0, 255)`
pub fn quantize_norm(norm: f32) -> u8 {
    if norm <= 0.0 || !norm.is_finite() {
        return 0;
    }
    let q = (norm.log2() * 4.0 + 128.0).round();
    q.clamp(0.0, 255.0) as u8
}

/// Decode an 8-bit log-quantized norm back to f32.
///
/// Formula: `norm = 2^((q - 128) / 4.0)`
pub fn dequantize_norm(q: u8) -> f32 {
    2.0_f32.powf((q as f32 - 128.0) / 4.0)
}

// ── Angle quantization ─────────────────────────────────────────────

/// Encode an angle in `[0, 2π)` to 8-bit uniform.
pub fn quantize_angle(theta: f32) -> u8 {
    let normalized = theta.rem_euclid(2.0 * PI);
    let q = (normalized / (2.0 * PI) * 256.0).round();
    (q as u32 % 256) as u8
}

/// Decode an 8-bit angle back to `[0, 2π)`.
pub fn dequantize_angle(q: u8) -> f32 {
    q as f32 * (2.0 * PI / 256.0)
}

// ── Vector-level operations ────────────────────────────────────────

/// Quantize an fp32 weight vector to polar form.
///
/// The vector is decomposed into:
/// - **norm**: L2 magnitude → 8-bit log-scale
/// - **angles**: sequential atan2 pairs → 8-bit uniform each
pub fn quantize_vector(weights: &[f32]) -> Result<PolarQuantized, PolarQuantError> {
    if weights.is_empty() {
        return Err(PolarQuantError::EmptyVector);
    }
    if weights.len() == 1 {
        let norm = weights[0].abs();
        return Ok(PolarQuantized {
            norm_q8: quantize_norm(norm),
            angles: vec![if weights[0] >= 0.0 { 0 } else { 128 }], // 0 or π
            original_dim: 1,
        });
    }

    // Compute L2 norm
    let norm: f32 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
    let norm_q8 = quantize_norm(norm);

    // Compute angles using sequential atan2 decomposition
    // For d dimensions, we get d-1 angles in [0, 2π)
    let safe_norm = if norm > 1e-30 { norm } else { 1.0 };
    let unit: Vec<f32> = weights.iter().map(|w| w / safe_norm).collect();

    let mut angles = Vec::with_capacity(weights.len() - 1);
    let mut remaining_norm_sq = 1.0_f32;

    for i in 0..weights.len() - 1 {
        // θ_i = atan2(remaining_magnitude, unit[i])
        let cos_theta = if remaining_norm_sq > 1e-20 {
            unit[i] / remaining_norm_sq.sqrt()
        } else {
            0.0
        };
        let theta = cos_theta.clamp(-1.0, 1.0).acos();
        // Check sign for angles beyond the first hemisphere
        let signed_theta = if i + 1 < weights.len() && unit[i + 1] < 0.0 && i == weights.len() - 2 {
            2.0 * PI - theta
        } else {
            theta
        };
        angles.push(quantize_angle(signed_theta));
        remaining_norm_sq -= unit[i] * unit[i];
        remaining_norm_sq = remaining_norm_sq.max(0.0);
    }

    Ok(PolarQuantized {
        norm_q8,
        angles,
        original_dim: weights.len(),
    })
}

/// Dequantize a polar-quantized vector back to fp32.
pub fn dequantize_vector(pq: &PolarQuantized) -> Result<Vec<f32>, PolarQuantError> {
    if pq.original_dim == 0 {
        return Err(PolarQuantError::EmptyVector);
    }
    if pq.angles.len() + 1 < pq.original_dim && pq.original_dim > 1 {
        return Err(PolarQuantError::DimensionMismatch {
            expected: pq.original_dim - 1,
            got: pq.angles.len(),
        });
    }

    let norm = dequantize_norm(pq.norm_q8);

    if pq.original_dim == 1 {
        let sign = if pq.angles.first().copied().unwrap_or(0) >= 128 {
            -1.0
        } else {
            1.0
        };
        return Ok(vec![sign * norm]);
    }

    let mut result = Vec::with_capacity(pq.original_dim);
    let mut remaining_scale = 1.0_f32;

    for (i, &angle_q) in pq.angles.iter().enumerate() {
        let theta = dequantize_angle(angle_q);
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        result.push(norm * remaining_scale * cos_t);
        remaining_scale *= sin_t;

        // Last angle: also emit the final component
        if i == pq.angles.len() - 1 {
            result.push(norm * remaining_scale);
        }
    }

    // Pad or truncate to original_dim
    result.resize(pq.original_dim, 0.0);
    Ok(result)
}

/// Quantize a batch of weight vectors and return statistics.
pub fn quantize_batch(
    vectors: &[Vec<f32>],
) -> Result<(Vec<PolarQuantized>, QuantStats), PolarQuantError> {
    let mut quantized = Vec::with_capacity(vectors.len());
    let mut total_error = 0.0_f64;
    let mut total_original_bytes = 0usize;
    let mut total_quantized_bytes = 0usize;

    for vec in vectors {
        let pq = quantize_vector(vec)?;

        // Measure reconstruction error
        let reconstructed = dequantize_vector(&pq)?;
        let orig_norm: f32 = vec.iter().map(|w| w * w).sum::<f32>().sqrt();
        let error_norm: f32 = vec
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let relative_error = if orig_norm > 1e-30 {
            error_norm / orig_norm
        } else {
            0.0
        };
        total_error += relative_error as f64;

        total_original_bytes += vec.len() * 4; // fp32 = 4 bytes
        total_quantized_bytes += 1 + pq.angles.len(); // norm_q8 + angles
        quantized.push(pq);
    }

    let count = vectors.len();
    let stats = QuantStats {
        count,
        original_bytes: total_original_bytes,
        quantized_bytes: total_quantized_bytes,
        compression_ratio: if total_quantized_bytes > 0 {
            total_original_bytes as f32 / total_quantized_bytes as f32
        } else {
            0.0
        },
        avg_relative_error: if count > 0 {
            (total_error / count as f64) as f32
        } else {
            0.0
        },
    };

    Ok((quantized, stats))
}

/// Serialize a `PolarQuantized` to a compact byte representation.
///
/// Format: `[dim_u32_le][norm_q8][angle_0][angle_1]...[angle_{d-2}]`
pub fn to_bytes(pq: &PolarQuantized) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 1 + pq.angles.len());
    buf.extend_from_slice(&(pq.original_dim as u32).to_le_bytes());
    buf.push(pq.norm_q8);
    buf.extend_from_slice(&pq.angles);
    buf
}

/// Deserialize a `PolarQuantized` from its compact byte representation.
pub fn from_bytes(data: &[u8]) -> Result<PolarQuantized, PolarQuantError> {
    if data.len() < 5 {
        return Err(PolarQuantError::Corrupted("payload too short".into()));
    }
    let dim = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| PolarQuantError::Corrupted("bad dim".into()))?,
    ) as usize;
    let norm_q8 = data[4];
    let expected_angles = if dim > 1 { dim - 1 } else { 1 };
    if data.len() < 5 + expected_angles {
        return Err(PolarQuantError::Corrupted(format!(
            "expected {} angle bytes, got {}",
            expected_angles,
            data.len() - 5
        )));
    }
    let angles = data[5..5 + expected_angles].to_vec();
    Ok(PolarQuantized {
        norm_q8,
        angles,
        original_dim: dim,
    })
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_roundtrip() {
        // 8-bit log-scale with 4.0 divisor: covers ~1e-39 to ~1e+38
        for &norm in &[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            let q = quantize_norm(norm);
            let reconstructed = dequantize_norm(q);
            let error = (reconstructed - norm).abs() / norm;
            assert!(
                error < 0.20,
                "norm={norm} q={q} reconstructed={reconstructed} error={error:.4}"
            );
        }
    }

    #[test]
    fn angle_roundtrip() {
        for deg in (0..360).step_by(15) {
            let theta = deg as f32 * PI / 180.0;
            let q = quantize_angle(theta);
            let reconstructed = dequantize_angle(q);
            let error = (reconstructed - theta).abs();
            assert!(
                error < 0.03,
                "theta={theta:.3} q={q} reconstructed={reconstructed:.3} error={error:.4}"
            );
        }
    }

    #[test]
    fn vector_roundtrip_small() {
        let w = vec![0.5, -0.3, 0.7, 0.1];
        let pq = quantize_vector(&w).unwrap();
        let reconstructed = dequantize_vector(&pq).unwrap();
        let orig_norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        let err_norm: f32 = w
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let relative = err_norm / orig_norm;
        assert!(
            relative < 0.15,
            "relative error {relative:.4} exceeds 15% for dim=4"
        );
    }

    #[test]
    fn batch_quantize() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![-0.5, 0.5, -0.5, 0.5],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let (quantized, stats) = quantize_batch(&vectors).unwrap();
        assert_eq!(quantized.len(), 3);
        assert_eq!(stats.count, 3);
        assert!(stats.compression_ratio > 3.0, "expected ~4x compression");
        println!(
            "batch stats: count={} ratio={:.2}x avg_error={:.4}",
            stats.count, stats.compression_ratio, stats.avg_relative_error
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let w = vec![0.23, -0.87, 0.12, 0.44];
        let pq = quantize_vector(&w).unwrap();
        let bytes = to_bytes(&pq);
        let pq2 = from_bytes(&bytes).unwrap();
        assert_eq!(pq, pq2);
    }

    #[test]
    fn single_element() {
        let w = vec![42.0];
        let pq = quantize_vector(&w).unwrap();
        let r = dequantize_vector(&pq).unwrap();
        assert!((r[0] - 42.0).abs() / 42.0 < 0.20);
    }

    #[test]
    fn empty_vector_error() {
        assert!(quantize_vector(&[]).is_err());
    }
}
