use serde::{Deserialize, Serialize};

/// A semantic lattice point in a 3D Ruby-cube-like space.
/// Coordinates are signed so the lattice can represent centered neighborhoods.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LatticePoint3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
    /// Optional hierarchy layer for coarse routing between semantic regions.
    pub layer: u8,
}

impl LatticePoint3 {
    pub const fn new(x: i16, y: i16, z: i16, layer: u8) -> Self {
        Self { x, y, z, layer }
    }

    pub fn manhattan_distance(&self, other: &Self) -> u32 {
        let base = (self.x as i32 - other.x as i32).unsigned_abs()
            + (self.y as i32 - other.y as i32).unsigned_abs()
            + (self.z as i32 - other.z as i32).unsigned_abs();
        if self.layer == other.layer {
            base
        } else {
            base + 256
        }
    }
}

/// Interpretable axis labels for a semantic cube.
/// Example:
/// - x_axis = "topic"
/// - y_axis = "intent"
/// - z_axis = "emotion"
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CubeAxes {
    pub x_axis: String,
    pub y_axis: String,
    pub z_axis: String,
}

/// One semantic item mapped into the cube.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticCubeEntry {
    pub label: String,
    pub point: LatticePoint3,
}

/// Artifact representing a semantic lattice view derived from an origin dataset.
/// This is intended for experimentation and reuse-first AI pipelines.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticCubeArtifact {
    pub schema: String,
    pub origin_id: sodl_core::OriginId,
    pub artifact_id: String,
    pub created_at: time::OffsetDateTime,
    pub axes: CubeAxes,
    pub entries: Vec<SemanticCubeEntry>,
}

impl SemanticCubeArtifact {
    pub fn validate(&self) -> sodl_core::Result<()> {
        if self.artifact_id.trim().is_empty() {
            return Err(sodl_core::SodlError::Invalid("artifact_id is empty".into()));
        }
        if self.axes.x_axis.trim().is_empty()
            || self.axes.y_axis.trim().is_empty()
            || self.axes.z_axis.trim().is_empty()
        {
            return Err(sodl_core::SodlError::Invalid(
                "cube axes must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Return nearest semantic labels from the lattice using Manhattan distance.
    pub fn nearest_labels(&self, query: LatticePoint3, k: usize) -> Vec<String> {
        let mut pairs: Vec<(u32, String)> = self
            .entries
            .iter()
            .map(|e| (e.point.manhattan_distance(&query), e.label.clone()))
            .collect();
        pairs.sort_by_key(|(d, _)| *d);
        pairs.into_iter().take(k).map(|(_, s)| s).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_labels_orders_by_distance() {
        let art = SemanticCubeArtifact {
            schema: "sodl:cube:v1".into(),
            origin_id: sodl_core::new_origin_id(),
            artifact_id: "cube:test".into(),
            created_at: time::OffsetDateTime::now_utc(),
            axes: CubeAxes {
                x_axis: "topic".into(),
                y_axis: "intent".into(),
                z_axis: "emotion".into(),
            },
            entries: vec![
                SemanticCubeEntry {
                    label: "therapy".into(),
                    point: LatticePoint3::new(1, 1, 1, 0),
                },
                SemanticCubeEntry {
                    label: "anxiety".into(),
                    point: LatticePoint3::new(2, 2, 2, 0),
                },
                SemanticCubeEntry {
                    label: "finance".into(),
                    point: LatticePoint3::new(20, 20, 20, 0),
                },
            ],
        };

        let labels = art.nearest_labels(LatticePoint3::new(0, 0, 0, 0), 2);
        assert_eq!(labels, vec!["therapy".to_string(), "anxiety".to_string()]);
    }
}
