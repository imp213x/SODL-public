//! Policy engine: durability classes, retention, access, pinning, and garbage collection rules.
//!
//! SODL durability is **explicit**.
//! - Ephemeral: TTL + best-effort retrieval
//! - BestEffort: soft retention targets, reclaimable under pressure
//! - Durable: pinned with minimum replica guarantees (implementation dependent)

use serde::{Deserialize, Serialize};
use sodl_core::{Capability, Durability, OriginId, PrincipalId, Result};

/// Identifies a durable failure domain (region/provider/node/etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageZone(pub String);

/// Retention policy for an origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub durability: Durability,
    /// Optional TTL for ephemeral/best-effort content.
    pub ttl_seconds: Option<u64>,
    /// Minimum replica count target for durable content.
    pub min_replicas: Option<u8>,
}

/// Access policy for an origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPolicy {
    /// Default capabilities granted to newly shared principals (if any).
    pub default_caps: Vec<Capability>,
    /// Whether principals can re-share.
    pub allow_reshare: bool,
    /// Whether principals can create derivations.
    pub allow_derivation: bool,
}

/// Policy attached to an origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginPolicy {
    pub origin_id: OriginId,
    pub retention: RetentionPolicy,
    pub access: AccessPolicy,
}

pub trait PolicyStore: Send + Sync {
    fn get_origin_policy(&self, origin_id: OriginId) -> Result<OriginPolicy>;
    fn put_origin_policy(&self, policy: OriginPolicy) -> Result<()>;
}

/// Policy evaluation boundary.
pub trait Authorizer: Send + Sync {
    fn check(&self, origin_id: OriginId, principal: &PrincipalId, cap: Capability) -> Result<()>;
}

/// Pin target granularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PinTarget {
    Origin { origin_id: OriginId },
    Representation { origin_id: OriginId, name: String },
}

/// Pin lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PinState {
    Pending,
    Active,
    Released,
}

/// Pin record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinRecord {
    pub pin_id: String,
    pub target: PinTarget,
    pub requested_by: PrincipalId,
    pub created_at: time::OffsetDateTime,
    pub state: PinState,

    /// Required replicas for this pin (overrides policy if set).
    pub min_replicas: Option<u8>,

    /// Optional zones where replicas must be placed (future).
    pub required_zones: Vec<StorageZone>,
}

/// Pin store boundary.
pub trait PinStore: Send + Sync {
    fn create_pin(&self, pin: PinRecord) -> Result<()>;
    fn get_pin(&self, pin_id: &str) -> Result<PinRecord>;
    fn list_pins_for_origin(&self, origin_id: OriginId) -> Result<Vec<PinRecord>>;
    fn update_pin(&self, pin: PinRecord) -> Result<()>;
    fn release_pin(&self, pin_id: &str) -> Result<()>;
}

/// Pin satisfaction planner: ensures bytes are durably stored according to pin requirements.
pub trait PinPlanner: Send + Sync {
    /// Returns pin ids that require action (e.g., pending or under-replicated).
    fn plan(&self) -> Result<Vec<String>>;
    /// Executes actions for the given pins (replication/upload to durable stores).
    fn execute(&self, pin_ids: &[String]) -> Result<()>;
}

/// GC planner interface: computes candidates eligible for deletion.
pub trait GarbageCollector: Send + Sync {
    fn plan(&self) -> Result<Vec<OriginId>>;
    fn execute(&self, origins: &[OriginId]) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Adaptive tiering: classifies blobs as Hot / Warm / Cold based on access
// frequency and recency. Used by the Python TieringManager background thread.
// ---------------------------------------------------------------------------

/// Storage tier for a blob based on access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessTier {
    /// Accessed frequently (>= threshold accesses in hot_window): keep pinned in RAM.
    Hot,
    /// Normal access or newly inserted: keep on local SSD/disk.
    Warm,
    /// Not accessed for cold_after_seconds: candidate for demotion to object storage.
    Cold,
}

impl Default for AccessTier {
    fn default() -> Self {
        Self::Warm
    }
}

/// Policy parameters that drive tier classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringPolicy {
    /// Minimum accesses within `hot_window_seconds` to qualify as Hot.
    pub hot_threshold_accesses: u64,
    /// Recency window (seconds) for Hot classification.
    pub hot_window_seconds: u64,
    /// Seconds of inactivity before a blob is classified as Cold.
    pub cold_after_seconds: u64,
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self {
            hot_threshold_accesses: 5,
            hot_window_seconds: 86_400,     // 24 h
            cold_after_seconds: 7 * 86_400, // 7 days
        }
    }
}

/// Evaluates a single blob's access record against a tiering policy.
pub trait TieringEvaluator: Send + Sync {
    fn evaluate(
        &self,
        access_count: u64,
        seconds_since_last_access: u64,
        policy: &TieringPolicy,
    ) -> AccessTier;
}

/// Default evaluator: pure function with no external state.
pub struct DefaultTieringEvaluator;

impl TieringEvaluator for DefaultTieringEvaluator {
    fn evaluate(
        &self,
        access_count: u64,
        seconds_since_last_access: u64,
        policy: &TieringPolicy,
    ) -> AccessTier {
        if access_count >= policy.hot_threshold_accesses
            && seconds_since_last_access <= policy.hot_window_seconds
        {
            AccessTier::Hot
        } else if seconds_since_last_access > policy.cold_after_seconds {
            AccessTier::Cold
        } else {
            AccessTier::Warm
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sodl_core::{new_origin_id, Capability, Durability, PrincipalId};

    #[test]
    fn can_serialize_pin_record() {
        let origin_id = new_origin_id();
        let pin = PinRecord {
            pin_id: "pin_1".to_string(),
            target: PinTarget::Origin { origin_id },
            requested_by: PrincipalId("user:a".into()),
            created_at: time::OffsetDateTime::now_utc(),
            state: PinState::Pending,
            min_replicas: Some(1),
            required_zones: vec![StorageZone("zone:uk-lon-1".into())],
        };

        let s = serde_json::to_string(&pin).unwrap();
        let back: PinRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(back.pin_id, "pin_1");
        assert!(matches!(back.state, PinState::Pending));
    }

    #[test]
    fn origin_policy_shapes() {
        let origin_id = new_origin_id();
        let policy = OriginPolicy {
            origin_id,
            retention: RetentionPolicy {
                durability: Durability::Ephemeral,
                ttl_seconds: Some(60),
                min_replicas: None,
            },
            access: AccessPolicy {
                default_caps: vec![Capability::Read],
                allow_reshare: true,
                allow_derivation: true,
            },
        };
        let _ = serde_json::to_string(&policy).unwrap();
    }
}
