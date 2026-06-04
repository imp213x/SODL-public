use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityQuery {
    pub principal: String,
    pub query_text: String,
    pub requested_caps: Vec<String>,
    pub max_results: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteCandidate {
    pub label: String,
    pub score: u32,
    pub basis: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteDecision {
    pub allow: bool,
    pub candidates: Vec<RouteCandidate>,
}

pub struct SemanticRouter;

impl SemanticRouter {
    pub fn simple_route(labels: Vec<String>, query: &CapabilityQuery) -> RouteDecision {
        let mut candidates: Vec<RouteCandidate> = labels
            .into_iter()
            .map(|label| RouteCandidate {
                label,
                score: 1,
                basis: "semantic".to_string(),
            })
            .collect();
        candidates.truncate(query.max_results);
        RouteDecision {
            allow: !query.requested_caps.is_empty(),
            candidates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_route_limits_candidates() {
        let query = CapabilityQuery {
            principal: "carla".into(),
            query_text: "route this".into(),
            requested_caps: vec!["semantic_search".into()],
            max_results: 2,
        };
        let decision =
            SemanticRouter::simple_route(vec!["a".into(), "b".into(), "c".into()], &query);
        assert!(decision.allow);
        assert_eq!(decision.candidates.len(), 2);
    }
}
