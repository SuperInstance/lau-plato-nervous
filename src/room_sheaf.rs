//! PLATO rooms as sheaves: local data at each room, restriction maps for aggregation.
//!
//! Each room in PLATO is modeled as an open set in a topological space. The sheaf
//! assigns to each room a data structure (local section), and restriction maps
//! project data when moving from a room to a sub-room or aggregate view.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A section of data over a room — the local data assigned to that room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSection {
    /// Room identifier.
    pub room_id: String,
    /// Metric values stored in this room.
    pub metrics: DVector<f64>,
    /// Metadata labels.
    pub labels: HashMap<String, String>,
}

/// A restriction map: projects a section from one room to another (coarser view).
pub type RestrictionMap = fn(&RoomSection) -> RoomSection;

/// The room sheaf: maps room IDs to their local sections, with restriction maps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomSheaf {
    /// Local sections indexed by room ID.
    pub sections: HashMap<String, RoomSection>,
    /// Restriction map adjacency: (source_room, target_room).
    #[serde(skip)]
    pub restrictions: Vec<(String, String, RestrictionMap)>,
}

impl RoomSheaf {
    /// Create an empty sheaf.
    pub fn new() -> Self {
        Self {
            sections: HashMap::new(),
            restrictions: Vec::new(),
        }
    }

    /// Add a room section.
    pub fn add_section(&mut self, section: RoomSection) {
        self.sections.insert(section.room_id.clone(), section);
    }

    /// Identity restriction: returns the section unchanged.
    pub fn identity_restriction(section: &RoomSection) -> RoomSection {
        section.clone()
    }

    /// Aggregate restriction: sums metric values into a single scalar section.
    pub fn aggregate_restriction(section: &RoomSection) -> RoomSection {
        let sum = section.metrics.iter().sum();
        RoomSection {
            room_id: format!("{}_agg", section.room_id),
            metrics: DVector::from_element(1, sum),
            labels: section.labels.clone(),
        }
    }

    /// Restrict from one room to another using a named restriction map.
    pub fn restrict(&self, from: &str, map: RestrictionMap) -> Option<RoomSection> {
        self.sections.get(from).map(|s| map(s))
    }

    /// Compute the global section: aggregate all rooms into one section.
    pub fn global_section(&self) -> Option<RoomSection> {
        if self.sections.is_empty() {
            return None;
        }
        let mut combined_metrics = Vec::new();
        let mut labels = HashMap::new();
        let mut room_ids = Vec::new();
        for (_, section) in &self.sections {
            combined_metrics.extend(section.metrics.iter().cloned());
            for (k, v) in &section.labels {
                labels.insert(k.clone(), v.clone());
            }
            room_ids.push(section.room_id.clone());
        }
        Some(RoomSection {
            room_id: room_ids.join("+"),
            metrics: DVector::from_vec(combined_metrics),
            labels,
        })
    }

    /// Check the sheaf axiom: if two sections agree on overlaps, they glue.
    /// Returns true if all pairwise restrictions are compatible.
    pub fn check_gluing_axiom(&self) -> bool {
        // Simplified: check that all sections have consistent dimension
        // where restriction maps exist
        let dims: Vec<_> = self.sections.values().map(|s| s.metrics.nrows()).collect();
        if dims.is_empty() {
            return true;
        }
        // For the basic check, all sections should have same dimension
        dims.windows(2).all(|w| w[0] == w[1])
    }

    /// Stalk at a room: the direct limit of sections over neighborhoods.
    /// Simplified as just the section at that room.
    pub fn stalk(&self, room_id: &str) -> Option<&RoomSection> {
        self.sections.get(room_id)
    }

    /// Count sections.
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }
}

impl Default for RoomSheaf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_sheaf() {
        let sheaf = RoomSheaf::new();
        assert!(sheaf.is_empty());
        assert_eq!(sheaf.len(), 0);
    }

    #[test]
    fn test_add_section() {
        let mut sheaf = RoomSheaf::new();
        let section = RoomSection {
            room_id: "room-1".into(),
            metrics: DVector::from_vec(vec![1.0, 2.0, 3.0]),
            labels: HashMap::new(),
        };
        sheaf.add_section(section);
        assert_eq!(sheaf.len(), 1);
        assert!(sheaf.stalk("room-1").is_some());
    }

    #[test]
    fn test_identity_restriction() {
        let section = RoomSection {
            room_id: "room-1".into(),
            metrics: DVector::from_vec(vec![1.0, 2.0]),
            labels: HashMap::new(),
        };
        let result = RoomSheaf::identity_restriction(&section);
        assert_eq!(result.metrics, section.metrics);
    }

    #[test]
    fn test_aggregate_restriction() {
        let section = RoomSection {
            room_id: "room-1".into(),
            metrics: DVector::from_vec(vec![1.0, 2.0, 3.0]),
            labels: HashMap::new(),
        };
        let result = RoomSheaf::aggregate_restriction(&section);
        assert_eq!(result.metrics.nrows(), 1);
        assert!((result.metrics[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_restrict() {
        let mut sheaf = RoomSheaf::new();
        sheaf.add_section(RoomSection {
            room_id: "room-1".into(),
            metrics: DVector::from_vec(vec![3.0, 4.0]),
            labels: HashMap::new(),
        });
        let result = sheaf.restrict("room-1", RoomSheaf::aggregate_restriction);
        assert!(result.is_some());
        assert!((result.unwrap().metrics[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_section() {
        let mut sheaf = RoomSheaf::new();
        sheaf.add_section(RoomSection {
            room_id: "a".into(),
            metrics: DVector::from_vec(vec![1.0, 2.0]),
            labels: HashMap::new(),
        });
        sheaf.add_section(RoomSection {
            room_id: "b".into(),
            metrics: DVector::from_vec(vec![3.0, 4.0]),
            labels: HashMap::new(),
        });
        let global = sheaf.global_section().unwrap();
        assert_eq!(global.metrics.nrows(), 4);
    }

    #[test]
    fn test_global_section_empty() {
        let sheaf = RoomSheaf::new();
        assert!(sheaf.global_section().is_none());
    }

    #[test]
    fn test_gluing_axiom_uniform_dims() {
        let mut sheaf = RoomSheaf::new();
        sheaf.add_section(RoomSection {
            room_id: "a".into(),
            metrics: DVector::from_vec(vec![1.0, 2.0]),
            labels: HashMap::new(),
        });
        sheaf.add_section(RoomSection {
            room_id: "b".into(),
            metrics: DVector::from_vec(vec![3.0, 4.0]),
            labels: HashMap::new(),
        });
        assert!(sheaf.check_gluing_axiom());
    }

    #[test]
    fn test_gluing_axiom_mismatched_dims() {
        let mut sheaf = RoomSheaf::new();
        sheaf.add_section(RoomSection {
            room_id: "a".into(),
            metrics: DVector::from_vec(vec![1.0, 2.0]),
            labels: HashMap::new(),
        });
        sheaf.add_section(RoomSection {
            room_id: "b".into(),
            metrics: DVector::from_vec(vec![3.0]),
            labels: HashMap::new(),
        });
        assert!(!sheaf.check_gluing_axiom());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut sheaf = RoomSheaf::new();
        sheaf.add_section(RoomSection {
            room_id: "room-x".into(),
            metrics: DVector::from_vec(vec![1.0]),
            labels: HashMap::new(),
        });
        let json = serde_json::to_string(&sheaf).unwrap();
        let back: RoomSheaf = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
    }
}
