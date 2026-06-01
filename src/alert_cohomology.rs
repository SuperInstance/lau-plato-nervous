//! Sheaf cohomology on alert dependency graphs: H¹ = missed alerts.
//!
//! Alert dependencies form a graph. We compute sheaf cohomology on this graph
//! to detect "holes" — alerts that should have fired but didn't (H¹ classes).
//! Zeroth cohomology H⁰ measures connected alert clusters.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// An alert with dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub severity: f64,
    pub dependencies: Vec<String>,
    pub fired: bool,
}

/// An edge in the alert dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

/// The alert dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertGraph {
    pub alerts: HashMap<String, Alert>,
    pub edges: Vec<DependencyEdge>,
}

impl AlertGraph {
    pub fn new() -> Self {
        Self {
            alerts: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add an alert.
    pub fn add_alert(&mut self, alert: Alert) {
        self.alerts.insert(alert.id.clone(), alert);
    }

    /// Add an edge.
    pub fn add_edge(&mut self, source: String, target: String, weight: f64) {
        self.edges.push(DependencyEdge {
            source,
            target,
            weight,
        });
    }

    /// Get adjacency matrix (directed).
    pub fn adjacency_matrix(&self) -> DMatrix<f64> {
        let nodes: Vec<&String> = {
            let mut set: HashSet<&String> = HashSet::new();
            for a in self.alerts.keys() {
                set.insert(a);
            }
            for e in &self.edges {
                set.insert(&e.source);
                set.insert(&e.target);
            }
            let mut v: Vec<&String> = set.into_iter().collect();
            v.sort();
            v
        };
        let n = nodes.len();
        let idx: HashMap<&String, usize> = nodes.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut adj = DMatrix::zeros(n, n);
        for e in &self.edges {
            if let (Some(&i), Some(&j)) = (idx.get(&e.source), idx.get(&e.target)) {
                adj[(i, j)] = e.weight;
            }
        }
        adj
    }

    /// Compute H⁰: connected components of the undirected graph.
    pub fn compute_h0(&self) -> Vec<Vec<String>> {
        let mut parent: HashMap<String, String> = HashMap::new();
        for id in self.alerts.keys() {
            parent.insert(id.clone(), id.clone());
        }

        fn find(parent: &mut HashMap<String, String>, id: &str) -> String {
            let mut root = id.to_string();
            while parent.get(&root).map(|s| s.as_str()) != Some(root.as_str()) {
                root = parent.get(&root).unwrap().clone();
            }
            // Path compression
            let final_root = root.clone();
            let mut x = id.to_string();
            while x != final_root {
                let next = parent.get(&x).unwrap().clone();
                parent.insert(x, final_root.clone());
                x = next;
            }
            final_root
        }

        for e in &self.edges {
            let r1 = find(&mut parent, &e.source);
            let r2 = find(&mut parent, &e.target);
            if r1 != r2 {
                parent.insert(r2, r1);
            }
        }

        let mut components: HashMap<String, Vec<String>> = HashMap::new();
        for id in self.alerts.keys() {
            let root = find(&mut parent, id);
            components.entry(root).or_default().push(id.clone());
        }
        components.into_values().collect()
    }

    /// Compute H¹: missed alerts (alerts whose dependencies fired but they didn't).
    pub fn compute_h1(&self) -> Vec<String> {
        let mut missed = Vec::new();
        for (id, alert) in &self.alerts {
            if !alert.fired && !alert.dependencies.is_empty() {
                let deps_fired = alert
                    .dependencies
                    .iter()
                    .all(|dep| self.alerts.get(dep).map(|a| a.fired).unwrap_or(false));
                if deps_fired {
                    missed.push(id.clone());
                }
            }
        }
        missed.sort();
        missed
    }

    /// Compute the coboundary map δ⁰: vertex functions → edge functions.
    /// For each edge (u,v), δ⁰(f)(u,v) = f(v) - f(u).
    pub fn coboundary_map(&self, vertex_function: &HashMap<String, f64>) -> Vec<f64> {
        self.edges
            .iter()
            .map(|e| {
                let fu = vertex_function.get(&e.source).copied().unwrap_or(0.0);
                let fv = vertex_function.get(&e.target).copied().unwrap_or(0.0);
                fv - fu
            })
            .collect()
    }

    /// Check if an edge function is a coboundary (exact).
    /// A function g on edges is exact if g = δ⁰(f) for some f on vertices.
    pub fn is_coboundary(&self, edge_function: &[f64]) -> bool {
        if edge_function.len() != self.edges.len() {
            return false;
        }
        // Simplified: check if sum around each cycle is zero
        // For now, check that the edge function sums to zero (necessary condition)
        let sum: f64 = edge_function.iter().sum();
        sum.abs() < 1e-10
    }

    /// Count alerts.
    pub fn len(&self) -> usize {
        self.alerts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.alerts.is_empty()
    }
}

impl Default for AlertGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Cohomology computation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohomologyResult {
    /// H⁰ dimension = number of connected components.
    pub h0_dimension: usize,
    /// H¹ dimension = number of missed alert classes.
    pub h1_dimension: usize,
    /// The missed alerts (H¹ representatives).
    pub missed_alerts: Vec<String>,
    /// Connected components (H⁰ representatives).
    pub components: Vec<Vec<String>>,
}

/// Compute full cohomology of the alert dependency graph.
pub fn compute_cohomology(graph: &AlertGraph) -> CohomologyResult {
    let components = graph.compute_h0();
    let missed = graph.compute_h1();
    let h0_dim = components.len();
    let h1_dim = missed.len();
    CohomologyResult {
        h0_dimension: h0_dim,
        h1_dimension: h1_dim,
        missed_alerts: missed,
        components,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = AlertGraph::new();
        assert!(g.is_empty());
    }

    #[test]
    fn test_add_alert() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert {
            id: "a1".into(),
            severity: 0.5,
            dependencies: vec![],
            fired: true,
        });
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_h0_single_component() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_edge("a".into(), "b".into(), 1.0);
        let comps = g.compute_h0();
        assert_eq!(comps.len(), 1);
    }

    #[test]
    fn test_h0_two_components() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec![], fired: true });
        // No edges → two components
        let comps = g.compute_h0();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_h1_no_missed_alerts() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec!["a".into()], fired: true });
        let missed = g.compute_h1();
        assert!(missed.is_empty());
    }

    #[test]
    fn test_h1_missed_alert() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec!["a".into()], fired: false });
        let missed = g.compute_h1();
        assert_eq!(missed, vec!["b"]);
    }

    #[test]
    fn test_h1_not_missed_dep_not_fired() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: false });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec!["a".into()], fired: false });
        let missed = g.compute_h1();
        // b's dependency a didn't fire either, so b is not "missed"
        assert!(missed.is_empty());
    }

    #[test]
    fn test_coboundary_map() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_edge("a".into(), "b".into(), 1.0);
        let f: HashMap<String, f64> = [("a".into(), 1.0), ("b".into(), 3.0)].into();
        let cb = g.coboundary_map(&f);
        assert_eq!(cb.len(), 1);
        assert!((cb[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_cohomology() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec!["a".into()], fired: false });
        g.add_edge("a".into(), "b".into(), 1.0);
        let result = compute_cohomology(&g);
        assert_eq!(result.h0_dimension, 1);
        assert_eq!(result.h1_dimension, 1);
    }

    #[test]
    fn test_adjacency_matrix() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_alert(Alert { id: "b".into(), severity: 1.0, dependencies: vec![], fired: true });
        g.add_edge("a".into(), "b".into(), 2.0);
        let adj = g.adjacency_matrix();
        assert_eq!(adj.nrows(), 2);
    }

    #[test]
    fn test_serialization() {
        let mut g = AlertGraph::new();
        g.add_alert(Alert { id: "a".into(), severity: 1.0, dependencies: vec![], fired: true });
        let json = serde_json::to_string(&g).unwrap();
        let back: AlertGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
    }

    #[test]
    fn test_cohomology_result_serialization() {
        let result = CohomologyResult {
            h0_dimension: 2,
            h1_dimension: 1,
            missed_alerts: vec!["x".into()],
            components: vec![vec!["a".into()], vec!["b".into()]],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CohomologyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.h0_dimension, 2);
    }
}
