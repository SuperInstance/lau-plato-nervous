//! Configuration as a category: objects are configs, morphisms are valid reconfigurations.
//!
//! PLATO configurations form a category where:
//! - Objects are configuration states
//! - Morphisms are valid reconfiguration paths
//! - Composition chains reconfigurations
//! - Identity is a no-op reconfiguration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A PLATO configuration state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Unique config identifier.
    pub id: String,
    /// Key-value parameters.
    pub params: HashMap<String, ConfigValue>,
    /// Version.
    pub version: u64,
}

/// A configuration value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value")]
pub enum ConfigValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// A morphism (reconfiguration) from one config to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reconfiguration {
    pub source_id: String,
    pub target_id: String,
    /// Changes applied.
    pub changes: HashMap<String, ConfigValue>,
    /// Cost of this reconfiguration.
    pub cost: f64,
}

/// The category of configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigCategory {
    /// Objects (configs).
    pub configs: HashMap<String, Config>,
    /// Morphisms (reconfigurations).
    pub morphisms: Vec<Reconfiguration>,
}

impl ConfigCategory {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            morphisms: Vec::new(),
        }
    }

    /// Add a configuration object.
    pub fn add_config(&mut self, config: Config) {
        self.configs.insert(config.id.clone(), config);
    }

    /// Add a morphism (reconfiguration).
    pub fn add_morphism(&mut self, morphism: Reconfiguration) {
        self.morphisms.push(morphism);
    }

    /// Identity morphism: a no-op reconfiguration with zero cost.
    pub fn identity(config_id: &str) -> Reconfiguration {
        Reconfiguration {
            source_id: config_id.to_string(),
            target_id: config_id.to_string(),
            changes: HashMap::new(),
            cost: 0.0,
        }
    }

    /// Compose two morphisms: g ∘ f.
    /// Returns None if the target of f != source of g.
    pub fn compose(f: &Reconfiguration, g: &Reconfiguration) -> Option<Reconfiguration> {
        if f.target_id != g.source_id {
            return None;
        }
        let mut changes = f.changes.clone();
        for (k, v) in &g.changes {
            changes.insert(k.clone(), v.clone());
        }
        Some(Reconfiguration {
            source_id: f.source_id.clone(),
            target_id: g.target_id.clone(),
            changes,
            cost: f.cost + g.cost,
        })
    }

    /// Check associativity: (h ∘ g) ∘ f == h ∘ (g ∘ f).
    pub fn check_associativity(
        f: &Reconfiguration,
        g: &Reconfiguration,
        h: &Reconfiguration,
    ) -> bool {
        let left = Self::compose(&Self::compose(f, g).unwrap(), h);
        let right = Self::compose(f, &Self::compose(g, h).unwrap());
        match (left, right) {
            (Some(l), Some(r)) => {
                l.source_id == r.source_id
                    && l.target_id == r.target_id
                    && (l.cost - r.cost).abs() < 1e-10
            }
            _ => false,
        }
    }

    /// Find the cheapest path from source to target using Dijkstra-like search.
    pub fn find_cheapest_path(&self, source: &str, target: &str) -> Option<Reconfiguration> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(Debug)]
        struct State {
            current: String,
            cost: f64,
            changes: HashMap<String, ConfigValue>,
        }

        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.cost.partial_cmp(&self.cost)
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap = BinaryHeap::new();
        heap.push(State {
            current: source.to_string(),
            cost: 0.0,
            changes: HashMap::new(),
        });

        let mut visited: HashMap<String, f64> = HashMap::new();

        while let Some(state) = heap.pop() {
            if state.current == target {
                return Some(Reconfiguration {
                    source_id: source.to_string(),
                    target_id: target.to_string(),
                    changes: state.changes,
                    cost: state.cost,
                });
            }

            if let Some(&prev_cost) = visited.get(&state.current) {
                if state.cost > prev_cost {
                    continue;
                }
            }
            visited.insert(state.current.clone(), state.cost);

            for morphism in &self.morphisms {
                if morphism.source_id == state.current {
                    let next_cost = state.cost + morphism.cost;
                    if let Some(&best) = visited.get(&morphism.target_id) {
                        if next_cost >= best {
                            continue;
                        }
                    }
                    let mut next_changes = state.changes.clone();
                    for (k, v) in &morphism.changes {
                        next_changes.insert(k.clone(), v.clone());
                    }
                    heap.push(State {
                        current: morphism.target_id.clone(),
                        cost: next_cost,
                        changes: next_changes,
                    });
                }
            }
        }
        None
    }

    /// Count configs.
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }
}

impl Default for ConfigCategory {
    fn default() -> Self {
        Self::new()
    }
}

/// A functor from one config category to another (config transformation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFunctor {
    /// Mapping of object IDs.
    pub object_map: HashMap<String, String>,
    /// Mapping of morphism indices.
    pub morphism_map: HashMap<usize, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(id: &str, version: u64) -> Config {
        Config {
            id: id.to_string(),
            params: HashMap::new(),
            version,
        }
    }

    #[test]
    fn test_empty_category() {
        let cat = ConfigCategory::new();
        assert!(cat.is_empty());
    }

    #[test]
    fn test_add_config() {
        let mut cat = ConfigCategory::new();
        cat.add_config(make_config("cfg-1", 1));
        assert_eq!(cat.len(), 1);
    }

    #[test]
    fn test_identity_morphism() {
        let id = ConfigCategory::identity("cfg-1");
        assert_eq!(id.source_id, "cfg-1");
        assert_eq!(id.target_id, "cfg-1");
        assert!(id.changes.is_empty());
        assert!((id.cost).abs() < 1e-10);
    }

    #[test]
    fn test_compose_morphisms() {
        let f = Reconfiguration {
            source_id: "A".into(),
            target_id: "B".into(),
            changes: [("x".into(), ConfigValue::Int(1))].into(),
            cost: 1.0,
        };
        let g = Reconfiguration {
            source_id: "B".into(),
            target_id: "C".into(),
            changes: [("y".into(), ConfigValue::Int(2))].into(),
            cost: 2.0,
        };
        let composed = ConfigCategory::compose(&f, &g).unwrap();
        assert_eq!(composed.source_id, "A");
        assert_eq!(composed.target_id, "C");
        assert!((composed.cost - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose_incompatible() {
        let f = Reconfiguration {
            source_id: "A".into(),
            target_id: "B".into(),
            changes: HashMap::new(),
            cost: 1.0,
        };
        let g = Reconfiguration {
            source_id: "C".into(),
            target_id: "D".into(),
            changes: HashMap::new(),
            cost: 1.0,
        };
        assert!(ConfigCategory::compose(&f, &g).is_none());
    }

    #[test]
    fn test_identity_laws() {
        let f = Reconfiguration {
            source_id: "A".into(),
            target_id: "B".into(),
            changes: [("k".into(), ConfigValue::Bool(true))].into(),
            cost: 5.0,
        };
        let id_a = ConfigCategory::identity("A");
        let id_b = ConfigCategory::identity("B");
        let left = ConfigCategory::compose(&id_a, &f).unwrap();
        let right = ConfigCategory::compose(&f, &id_b).unwrap();
        assert_eq!(left.target_id, f.target_id);
        assert_eq!(right.source_id, f.source_id);
    }

    #[test]
    fn test_associativity() {
        let f = Reconfiguration {
            source_id: "A".into(),
            target_id: "B".into(),
            changes: HashMap::new(),
            cost: 1.0,
        };
        let g = Reconfiguration {
            source_id: "B".into(),
            target_id: "C".into(),
            changes: HashMap::new(),
            cost: 2.0,
        };
        let h = Reconfiguration {
            source_id: "C".into(),
            target_id: "D".into(),
            changes: HashMap::new(),
            cost: 3.0,
        };
        assert!(ConfigCategory::check_associativity(&f, &g, &h));
    }

    #[test]
    fn test_find_cheapest_path() {
        let mut cat = ConfigCategory::new();
        cat.add_config(make_config("A", 1));
        cat.add_config(make_config("B", 1));
        cat.add_config(make_config("C", 1));
        cat.add_morphism(Reconfiguration {
            source_id: "A".into(),
            target_id: "B".into(),
            changes: HashMap::new(),
            cost: 5.0,
        });
        cat.add_morphism(Reconfiguration {
            source_id: "A".into(),
            target_id: "C".into(),
            changes: HashMap::new(),
            cost: 10.0,
        });
        cat.add_morphism(Reconfiguration {
            source_id: "B".into(),
            target_id: "C".into(),
            changes: HashMap::new(),
            cost: 2.0,
        });
        let path = cat.find_cheapest_path("A", "C").unwrap();
        // A->B->C = 7.0 < A->C = 10.0
        assert!((path.cost - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_path_not_found() {
        let cat = ConfigCategory::new();
        assert!(cat.find_cheapest_path("X", "Y").is_none());
    }

    #[test]
    fn test_config_serialization() {
        let cfg = make_config("test", 3);
        let json = serde_json::to_string(&cfg).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "test");
        assert_eq!(back.version, 3);
    }

    #[test]
    fn test_config_value_types() {
        let v1 = ConfigValue::Bool(true);
        let v2 = ConfigValue::Int(42);
        let v3 = ConfigValue::Float(3.14);
        let v4 = ConfigValue::Str("hello".into());
        let json = serde_json::to_string(&vec![&v1, &v2, &v3, &v4]).unwrap();
        assert!(!json.is_empty());
    }
}
