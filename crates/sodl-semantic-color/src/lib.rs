use serde::{Deserialize, Serialize};

/// Color-spectrum-inspired semantic encoding for compact AI artifact tagging.
///
/// Design choice:
/// - Use a 32-bit code: 8-bit hierarchy + 24-bit RGB.
/// - Hierarchy carries the coarse semantic family.
/// - RGB carries the local semantic position inside that family.
/// - Hue is derived from RGB when needed, rather than stored redundantly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SemanticColor32 {
    /// Coarse semantic family / hierarchy bucket.
    pub hierarchy: u8,
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

impl SemanticColor32 {
    pub const fn new(hierarchy: u8, r: u8, g: u8, b: u8) -> Self {
        Self { hierarchy, r, g, b }
    }

    /// Pack into a 32-bit integer in H-R-G-B order.
    pub const fn pack(self) -> u32 {
        ((self.hierarchy as u32) << 24)
            | ((self.r as u32) << 16)
            | ((self.g as u32) << 8)
            | (self.b as u32)
    }

    pub const fn unpack(bits: u32) -> Self {
        Self {
            hierarchy: ((bits >> 24) & 0xFF) as u8,
            r: ((bits >> 16) & 0xFF) as u8,
            g: ((bits >> 8) & 0xFF) as u8,
            b: (bits & 0xFF) as u8,
        }
    }

    /// Derived hue in degrees [0, 360). This is intentionally computed on demand
    /// so hierarchy remains a separate semantic layer rather than being collapsed into hue.
    pub fn hue_degrees(&self) -> f32 {
        let r = self.r as f32 / 255.0;
        let g = self.g as f32 / 255.0;
        let b = self.b as f32 / 255.0;
        let max = r.max(g.max(b));
        let min = r.min(g.min(b));
        let delta = max - min;

        if delta == 0.0 {
            return 0.0;
        }

        let h = if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };

        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }

    /// Fast distance for clustering / ANN prefiltering.
    /// Penalizes hierarchy mismatch heavily while keeping RGB proximity meaningful.
    pub fn semantic_distance(&self, other: &Self) -> u32 {
        let hr = if self.hierarchy == other.hierarchy {
            0
        } else {
            1024
        };
        let dr = (self.r as i32 - other.r as i32).unsigned_abs();
        let dg = (self.g as i32 - other.g as i32).unsigned_abs();
        let db = (self.b as i32 - other.b as i32).unsigned_abs();
        hr + dr * dr + dg * dg + db * db
    }
}

/// Compact manifest for a color-coded semantic artifact derived from an origin dataset.
/// This does not replace the source bytes; it is a lineage-tracked representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticColorArtifact {
    pub schema: String,
    pub origin_id: sodl_core::OriginId,
    pub artifact_id: String,
    pub created_at: time::OffsetDateTime,

    /// Human-readable meaning for the hierarchy byte, e.g. "medical", "financial", "safety".
    pub hierarchy_label: String,

    /// Packed semantic color codes. In a real pipeline this may correspond to rows,
    /// chunks, concepts, or compressed semantic tokens.
    pub codes: Vec<u32>,
}

impl SemanticColorArtifact {
    pub fn validate(&self) -> sodl_core::Result<()> {
        if self.artifact_id.trim().is_empty() {
            return Err(sodl_core::SodlError::Invalid("artifact_id is empty".into()));
        }
        if self.hierarchy_label.trim().is_empty() {
            return Err(sodl_core::SodlError::Invalid(
                "hierarchy_label is empty".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let c = SemanticColor32::new(7, 10, 20, 30);
        assert_eq!(SemanticColor32::unpack(c.pack()), c);
    }

    #[test]
    fn hierarchy_penalty_matters() {
        let a = SemanticColor32::new(1, 20, 20, 20);
        let b = SemanticColor32::new(1, 25, 25, 25);
        let c = SemanticColor32::new(2, 20, 20, 20);
        assert!(a.semantic_distance(&b) < a.semantic_distance(&c));
    }
}
