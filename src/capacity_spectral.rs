//! Spectral gap analysis: how fast does the system converge to equilibrium?
//!
//! The spectral gap (1 - λ₂) of the system's transition graph determines
//! mixing time. A large spectral gap means fast convergence; a small one
//! means the system is slow to equilibrate.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A state in the system's transition graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub id: String,
    pub energy: f64,
    pub is_equilibrium: bool,
}

/// The transition graph between system states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionGraph {
    pub states: Vec<SystemState>,
    /// Adjacency: transition_graph[i][j] = transition rate from i to j.
    pub transition_matrix: DMatrix<f64>,
}

impl TransitionGraph {
    /// Build from a raw transition matrix.
    pub fn new(states: Vec<SystemState>, transition_matrix: DMatrix<f64>) -> Self {
        Self { states, transition_matrix }
    }

    /// Build a random walk transition matrix from an adjacency matrix.
    pub fn from_adjacency(adj: &DMatrix<f64>) -> Self {
        let n = adj.nrows();
        let states = (0..n)
            .map(|i| SystemState {
                id: format!("s{}", i),
                energy: 0.0,
                is_equilibrium: false,
            })
            .collect();

        let mut trans = DMatrix::zeros(n, n);
        for i in 0..n {
            let row_sum: f64 = adj.row(i).iter().sum();
            if row_sum > 0.0 {
                for j in 0..n {
                    trans[(i, j)] = adj[(i, j)] / row_sum;
                }
            } else {
                trans[(i, i)] = 1.0; // absorbing state
            }
        }
        Self { states, transition_matrix: trans }
    }

    /// Compute eigenvalues of the transition matrix.
    /// Returns eigenvalues sorted by magnitude (descending).
    pub fn eigenvalues(&self) -> Vec<f64> {
        let n = self.transition_matrix.nrows();
        if n == 0 {
            return Vec::new();
        }

        // Use power iteration for the top eigenvalue, then deflate
        let mut eigenvalues = Vec::new();
        let mut mat = self.transition_matrix.clone();

        for _ in 0..n {
            let (val, vec) = Self::power_iteration(&mat, 100, 1e-10);
            eigenvalues.push(val);
            // Deflate
            let outer = &vec * &vec.transpose();
            mat = &mat - val * outer;
        }

        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        eigenvalues
    }

    /// Power iteration for dominant eigenvalue.
    fn power_iteration(mat: &DMatrix<f64>, max_iter: usize, tol: f64) -> (f64, DVector<f64>) {
        let n = mat.nrows();
        let mut v = DVector::from_element(n, 1.0 / n as f64);

        for _ in 0..max_iter {
            let new_v = mat * &v;
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            let new_v = new_v / norm;
            let diff: f64 = (&new_v - &v).iter().map(|x| x.abs()).sum();
            v = new_v;
            if diff < tol {
                break;
            }
        }

        let Av = mat * &v;
        let eigenvalue = v.dot(&Av);
        (eigenvalue, v)
    }

    /// Compute the spectral gap: 1 - |λ₂|.
    /// This measures how fast the system converges to equilibrium.
    pub fn spectral_gap(&self) -> f64 {
        let eigenvalues = self.eigenvalues();
        if eigenvalues.len() < 2 {
            return 1.0; // trivial graph
        }
        // First eigenvalue should be ~1, second is λ₂
        1.0 - eigenvalues[1].abs()
    }

    /// Compute mixing time (approximate).
    /// t_mix(ε) ≈ (1/γ*) ln(1/ε), where γ* is the spectral gap.
    pub fn mixing_time(&self, epsilon: f64) -> f64 {
        let gap = self.spectral_gap();
        if gap <= 0.0 {
            return f64::INFINITY;
        }
        (1.0 / gap) * (1.0 / epsilon).ln()
    }

    /// Compute the stationary distribution (left eigenvector of eigenvalue 1).
    pub fn stationary_distribution(&self) -> DVector<f64> {
        let n = self.transition_matrix.nrows();
        if n == 0 {
            return DVector::zeros(0);
        }
        // Solve π P = π, i.e., P^T π = π
        // Use power iteration on P^T
        let pt = self.transition_matrix.transpose();
        let mut pi = DVector::from_element(n, 1.0 / n as f64);

        for _ in 0..200 {
            let new_pi = &pt * &pi;
            let sum: f64 = new_pi.iter().sum();
            if sum < 1e-15 {
                break;
            }
            let new_pi = new_pi / sum;
            let diff: f64 = (&new_pi - &pi).iter().map(|x| x.abs()).sum();
            pi = new_pi;
            if diff < 1e-10 {
                break;
            }
        }
        pi
    }

    /// Compute conductance (Cheeger constant approximation).
    pub fn conductance(&self) -> f64 {
        let pi = self.stationary_distribution();
        let n = self.transition_matrix.nrows();
        if n <= 1 {
            return 1.0;
        }

        let mut min_conductance = f64::MAX;
        // Check splits: first k states vs rest
        for k in 1..n {
            let mass_s: f64 = pi.rows(0, k).iter().sum();
            let mass_sc: f64 = pi.rows(k, n - k).iter().sum();
            if mass_s <= 0.0 || mass_sc <= 0.0 {
                continue;
            }
            // Flow across the cut
            let mut flow = 0.0;
            for i in 0..k {
                for j in k..n {
                    flow += pi[i] * self.transition_matrix[(i, j)];
                    flow += pi[j] * self.transition_matrix[(j, i)];
                }
            }
            let conductance = flow / mass_s.min(mass_sc);
            min_conductance = min_conductance.min(conductance);
        }
        if min_conductance == f64::MAX {
            0.0
        } else {
            min_conductance
        }
    }

    /// Total variation distance between current and stationary distribution.
    pub fn total_variation_distance(&self, current: &DVector<f64>) -> f64 {
        let pi = self.stationary_distribution();
        if pi.nrows() != current.nrows() {
            return f64::MAX;
        }
        let diff = current - &pi;
        0.5_f64 * diff.iter().map(|x| x.abs()).sum::<f64>()
    }

    /// Number of states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adj(n: usize) -> DMatrix<f64> {
        let mut adj = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[(i, j)] = 1.0;
                }
            }
        }
        adj
    }

    #[test]
    fn test_from_adjacency_complete() {
        let adj = make_adj(4);
        let tg = TransitionGraph::from_adjacency(&adj);
        assert_eq!(tg.len(), 4);
        // Each row should sum to 1
        for i in 0..4 {
            let row_sum: f64 = tg.transition_matrix.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spectral_gap_complete_graph() {
        let adj = make_adj(4);
        let tg = TransitionGraph::from_adjacency(&adj);
        let gap = tg.spectral_gap();
        // Complete graph should have a large spectral gap
        assert!(gap > 0.5, "spectral gap = {}", gap);
    }

    #[test]
    fn test_spectral_gap_path_graph() {
        // Path graph: 0-1-2-3
        let mut adj = DMatrix::zeros(4, 4);
        adj[(0, 1)] = 1.0;
        adj[(1, 0)] = 1.0;
        adj[(1, 2)] = 1.0;
        adj[(2, 1)] = 1.0;
        adj[(2, 3)] = 1.0;
        adj[(3, 2)] = 1.0;
        let tg = TransitionGraph::from_adjacency(&adj);
        let gap = tg.spectral_gap();
        // Path graph has smaller spectral gap than complete
        assert!(gap > 0.0, "spectral gap = {}", gap);
    }

    #[test]
    fn test_mixing_time() {
        let adj = make_adj(4);
        let tg = TransitionGraph::from_adjacency(&adj);
        let t = tg.mixing_time(0.01);
        assert!(t > 0.0);
        assert!(t.is_finite());
    }

    #[test]
    fn test_stationary_distribution_uniform() {
        let adj = make_adj(3);
        let tg = TransitionGraph::from_adjacency(&adj);
        let pi = tg.stationary_distribution();
        assert_eq!(pi.nrows(), 3);
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // For complete graph, stationary should be uniform
        for &v in pi.iter() {
            assert!((v - 1.0 / 3.0).abs() < 0.05, "v = {}", v);
        }
    }

    #[test]
    fn test_stationary_distribution_sums_to_one() {
        let mut adj = DMatrix::zeros(3, 3);
        adj[(0, 1)] = 1.0;
        adj[(1, 0)] = 0.5;
        adj[(1, 2)] = 0.5;
        adj[(2, 1)] = 1.0;
        let tg = TransitionGraph::from_adjacency(&adj);
        let pi = tg.stationary_distribution();
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_conductance() {
        let adj = make_adj(4);
        let tg = TransitionGraph::from_adjacency(&adj);
        let c = tg.conductance();
        assert!(c > 0.0);
    }

    #[test]
    fn test_total_variation_distance() {
        let adj = make_adj(3);
        let tg = TransitionGraph::from_adjacency(&adj);
        let current = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let tv = tg.total_variation_distance(&current);
        assert!(tv > 0.0);
    }

    #[test]
    fn test_total_variation_same() {
        let adj = make_adj(3);
        let tg = TransitionGraph::from_adjacency(&adj);
        let pi = tg.stationary_distribution();
        let tv = tg.total_variation_distance(&pi);
        assert!(tv < 0.01, "tv = {}", tv);
    }

    #[test]
    fn test_empty_graph() {
        let tg = TransitionGraph::new(vec![], DMatrix::zeros(0, 0));
        assert!(tg.is_empty());
        assert_eq!(tg.spectral_gap(), 1.0);
    }

    #[test]
    fn test_serialization() {
        let adj = make_adj(2);
        let tg = TransitionGraph::from_adjacency(&adj);
        let json = serde_json::to_string(&tg).unwrap();
        let back: TransitionGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 2);
    }
}
