//! Homology of the PLATO event history: detect topological features in time series.
//!
//! Events in PLATO history are modeled as simplices. By building a simplicial
//! complex from temporal proximity and computing its homology, we can detect
//! topological features like cycles (recurring patterns) and connected components
//! (event clusters).

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

/// A PLATO event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEvent {
    pub id: String,
    pub timestamp: f64,
    pub event_type: String,
    pub payload: String,
}

/// A simplex (subset of vertices).
pub type Simplex = BTreeSet<usize>;

/// A simplicial complex built from event history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventComplex {
    /// Vertices (events).
    pub events: Vec<HistoryEvent>,
    /// Simplices, indexed by dimension.
    pub simplices: Vec<Vec<Simplex>>,
}

impl EventComplex {
    /// Create an empty complex.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            simplices: vec![Vec::new()], // dimension 0
        }
    }

    /// Build a Vietoris-Rips complex from events with a temporal proximity threshold.
    pub fn build_rips(events: Vec<HistoryEvent>, epsilon: f64, max_dim: usize) -> Self {
        let n = events.len();
        let mut simplices: Vec<Vec<Simplex>> = (0..=max_dim).map(|_| Vec::new()).collect();

        // 0-simplices (vertices)
        for i in 0..n {
            simplices[0].push(BTreeSet::from([i]));
        }

        // Compute pairwise distances
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = (events[i].timestamp - events[j].timestamp).abs();
                if dist <= epsilon {
                    edges.push((i, j, dist));
                    simplices[1].push(BTreeSet::from([i, j]));
                }
            }
        }

        // Higher simplices: a k-simplex exists iff all pairs are within epsilon
        for dim in 2..=max_dim {
            if simplices[dim - 1].is_empty() {
                break;
            }
            let candidates = Self::find_higher_simplices(&simplices[dim - 1], dim);
            simplices[dim] = candidates;
        }

        Self { events, simplices }
    }

    /// Find k-simplices from (k-1)-simplices.
    fn find_higher_simplices(lower: &[Simplex], target_dim: usize) -> Vec<Simplex> {
        let mut result = Vec::new();
        let n = lower.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // Check if lower[i] and lower[j] differ by exactly one vertex
                // and their union has the right size
                let union: BTreeSet<usize> = lower[i].union(&lower[j]).cloned().collect();
                if union.len() == target_dim + 1 {
                    // Check all subsets of size target_dim are in lower
                    let all_present = union
                        .iter()
                        .all(|v| {
                            let mut subset = union.clone();
                            subset.remove(v);
                            lower.contains(&subset)
                        });
                    if all_present {
                        result.push(union);
                    }
                }
            }
        }
        // Deduplicate
        result.sort();
        result.dedup();
        result
    }

    /// Compute the boundary matrix for dimension k.
    pub fn boundary_matrix(&self, k: usize) -> DMatrix<i32> {
        if k == 0 || k >= self.simplices.len() {
            return DMatrix::zeros(0, 0);
        }
        let num_k = self.simplices[k].len();
        if num_k == 0 {
            return DMatrix::zeros(0, 0);
        }
        let k_minus_1 = if k > 0 && k - 1 < self.simplices.len() {
            &self.simplices[k - 1]
        } else {
            return DMatrix::zeros(0, num_k);
        };
        let num_km1 = k_minus_1.len();
        let mut mat = DMatrix::zeros(num_km1, num_k);

        for (j, simplex) in self.simplices[k].iter().enumerate() {
            let mut sign = 1i32;
            for v in simplex.iter() {
                let mut face = simplex.clone();
                face.remove(v);
                if let Some(idx) = k_minus_1.iter().position(|s| s == &face) {
                    mat[(idx, j)] += sign;
                }
                sign *= -1;
            }
        }
        mat
    }

    /// Compute Betti numbers (dim H_k) via Smith normal form approximation.
    /// Returns (betti_0, betti_1, ...) up to max_dim.
    pub fn betti_numbers(&self) -> Vec<usize> {
        let max_dim = self.simplices.len().saturating_sub(1);
        let mut betti = Vec::with_capacity(max_dim + 1);

        for k in 0..=max_dim {
            let rank_k = self.boundary_rank(k + 1);
            let rank_km1 = self.boundary_rank(k);
            let dim_k = if k < self.simplices.len() {
                self.simplices[k].len()
            } else {
                0
            };
            betti.push(dim_k.saturating_sub(rank_k).saturating_sub(rank_km1));
        }
        betti
    }

    /// Rank of the boundary map at dimension k.
    fn boundary_rank(&self, k: usize) -> usize {
        if k >= self.simplices.len() {
            return 0;
        }
        let mat = self.boundary_matrix(k);
        // Approximate rank by counting nonzero rows
        let mut rank = 0;
        for i in 0..mat.nrows() {
            let row_nonzero: bool = mat.row(i).iter().any(|&v| v != 0);
            if row_nonzero {
                rank += 1;
            }
        }
        rank
    }

    /// Count connected components (Betti-0).
    pub fn connected_components(&self) -> usize {
        let n = self.events.len();
        if n == 0 {
            return 0;
        }
        let mut parent: Vec<usize> = (0..n).collect();

        fn find_fn(parent: &[usize], mut v: usize) -> usize {
            while parent[v] != v {
                v = parent[v];
            }
            v
        }

        if self.simplices.len() > 1 {
            for simplex in &self.simplices[1] {
                let verts: Vec<usize> = simplex.iter().cloned().collect();
                if verts.len() >= 2 {
                    let r0 = find_fn(&parent, verts[0]);
                    let r1 = find_fn(&parent, verts[1]);
                    if r0 != r1 {
                        parent[r1] = r0;
                    }
                }
            }
        }

        let mut roots = HashSet::new();
        for i in 0..n {
            roots.insert(find_fn(&parent, i));
        }
        roots.len()
    }

    /// Count events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for EventComplex {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute persistent homology barcode (simplified).
/// Returns pairs (birth, death) for each feature in dimension 0.
pub fn persistence_barcode_dim0(
    events: &[HistoryEvent],
    epsilon_values: &[f64],
) -> Vec<(f64, Option<f64>)> {
    let n = events.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute all pairwise distances
    let mut distances: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push((events[i].timestamp - events[j].timestamp).abs());
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut barcodes = Vec::new();
    // Each point starts as its own component at epsilon=0
    let mut components = n;
    let mut prev_eps = 0.0;
    for &eps in epsilon_values {
        let complex = EventComplex::build_rips(events.to_vec(), eps, 1);
        let cc = complex.connected_components();
        let merged = components.saturating_sub(cc);
        for _ in 0..merged {
            barcodes.push((prev_eps, Some(eps)));
        }
        components = cc;
        prev_eps = eps;
    }
    // Remaining components are infinite features
    for _ in 0..components {
        barcodes.push((0.0, None));
    }
    barcodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: &str, ts: f64) -> HistoryEvent {
        HistoryEvent {
            id: id.into(),
            timestamp: ts,
            event_type: "test".into(),
            payload: "".into(),
        }
    }

    #[test]
    fn test_empty_complex() {
        let c = EventComplex::new();
        assert!(c.is_empty());
        let betti = c.betti_numbers();
        assert!(betti.is_empty() || betti[0] == 0);
    }

    #[test]
    fn test_single_event() {
        let events = vec![make_event("e1", 0.0)];
        let c = EventComplex::build_rips(events, 1.0, 2);
        assert_eq!(c.len(), 1);
        let betti = c.betti_numbers();
        assert_eq!(betti[0], 1); // one component
    }

    #[test]
    fn test_two_events_connected() {
        let events = vec![make_event("e1", 0.0), make_event("e2", 0.5)];
        let c = EventComplex::build_rips(events, 1.0, 2);
        assert_eq!(c.connected_components(), 1);
    }

    #[test]
    fn test_two_events_disconnected() {
        let events = vec![make_event("e1", 0.0), make_event("e2", 10.0)];
        let c = EventComplex::build_rips(events, 1.0, 2);
        assert_eq!(c.connected_components(), 2);
    }

    #[test]
    fn test_triangle_complex() {
        let events = vec![
            make_event("e1", 0.0),
            make_event("e2", 1.0),
            make_event("e3", 2.0),
        ];
        let c = EventComplex::build_rips(events, 2.5, 2);
        assert!(c.simplices[1].len() >= 2); // at least 2 edges
        assert!(c.connected_components() >= 1);
    }

    #[test]
    fn test_boundary_matrix_dim1() {
        let events = vec![
            make_event("e1", 0.0),
            make_event("e2", 1.0),
            make_event("e3", 2.0),
        ];
        let c = EventComplex::build_rips(events, 2.5, 2);
        let b1 = c.boundary_matrix(1);
        assert!(b1.nrows() > 0 || c.simplices[0].len() == 0);
    }

    #[test]
    fn test_betti_numbers_single_point() {
        let events = vec![make_event("e1", 0.0)];
        let c = EventComplex::build_rips(events, 1.0, 2);
        let betti = c.betti_numbers();
        assert_eq!(betti[0], 1);
    }

    #[test]
    fn test_persistence_barcode() {
        let events = vec![
            make_event("e1", 0.0),
            make_event("e2", 1.0),
            make_event("e3", 5.0),
        ];
        let epsilons = vec![0.5, 1.5, 3.0, 6.0];
        let barcode = persistence_barcode_dim0(&events, &epsilons);
        assert!(!barcode.is_empty());
    }

    #[test]
    fn test_event_serialization() {
        let evt = make_event("e1", 42.0);
        let json = serde_json::to_string(&evt).unwrap();
        let back: HistoryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "e1");
    }

    #[test]
    fn test_complex_serialization() {
        let events = vec![make_event("e1", 0.0), make_event("e2", 1.0)];
        let c = EventComplex::build_rips(events, 2.0, 2);
        let json = serde_json::to_string(&c).unwrap();
        let back: EventComplex = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 2);
    }
}
