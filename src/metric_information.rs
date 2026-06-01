//! Information geometry of PLATO metrics: Fisher metric on metric distributions.
//!
//! PLATO metrics are modeled as probability distributions over outcomes.
//! The Fisher information metric gives a Riemannian metric on the manifold
//! of these distributions, allowing us to measure distances between metric states.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A categorical probability distribution over metric states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDistribution {
    /// Category labels.
    pub labels: Vec<String>,
    /// Probabilities (must sum to 1).
    pub probabilities: DVector<f64>,
}

impl MetricDistribution {
    /// Create a new distribution, normalizing if needed.
    pub fn new(labels: Vec<String>, probabilities: Vec<f64>) -> Self {
        let total: f64 = probabilities.iter().sum();
        let probs = if total > 0.0 {
            DVector::from_vec(probabilities.into_iter().map(|p| p / total).collect())
        } else {
            let n = probabilities.len();
            DVector::from_element(n, 1.0 / n as f64)
        };
        Self { labels, probabilities: probs }
    }

    /// Uniform distribution over n categories.
    pub fn uniform(n: usize) -> Self {
        Self {
            labels: (0..n).map(|i| format!("cat-{}", i)).collect(),
            probabilities: DVector::from_element(n, 1.0 / n as f64),
        }
    }

    /// Shannon entropy.
    pub fn entropy(&self) -> f64 {
        -self
            .probabilities
            .iter()
            .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f64>()
    }

    /// KL divergence from self to other: KL(self || other).
    pub fn kl_divergence(&self, other: &MetricDistribution) -> f64 {
        self.probabilities
            .iter()
            .zip(other.probabilities.iter())
            .map(|(&p, &q)| {
                if p > 0.0 && q > 0.0 {
                    p * (p / q).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    }

    /// Compute the Fisher information matrix (diagonal approximation for categorical).
    pub fn fisher_information(&self) -> DMatrix<f64> {
        let n = self.probabilities.nrows();
        let mut fim = DMatrix::zeros(n, n);
        for (i, &p) in self.probabilities.iter().enumerate() {
            if p > 0.0 {
                fim[(i, i)] = 1.0 / p;
            }
        }
        fim
    }
}

/// Compute the Fisher-Rao distance between two distributions (approximation via KL).
pub fn fisher_rao_distance(p: &MetricDistribution, q: &MetricDistribution) -> f64 {
    let kl_pq = p.kl_divergence(q);
    let kl_qp = q.kl_divergence(p);
    // Symmetrized KL (Jeffreys divergence) as approximation
    (kl_pq + kl_qp).sqrt()
}

/// Compute the Bhattacharyya coefficient between two distributions.
pub fn bhattacharyya_coefficient(p: &MetricDistribution, q: &MetricDistribution) -> f64 {
    p.probabilities
        .iter()
        .zip(q.probabilities.iter())
        .map(|(&a, &b)| (a * b).sqrt())
        .sum()
}

/// Bhattacharyya distance.
pub fn bhattacharyya_distance(p: &MetricDistribution, q: &MetricDistribution) -> f64 {
    let bc = bhattacharyya_coefficient(p, q);
    if bc > 0.0 {
        -bc.ln()
    } else {
        f64::INFINITY
    }
}

/// Compute the Rao geodesic path between two distributions (simplified).
/// Returns intermediate distributions along the Fisher-Rao geodesic.
pub fn rao_geodesic(
    p: &MetricDistribution,
    q: &MetricDistribution,
    steps: usize,
) -> Vec<MetricDistribution> {
    let mut path = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let interp: Vec<f64> = p
            .probabilities
            .iter()
            .zip(q.probabilities.iter())
            .map(|(&a, &b)| {
                // Square root interpolation on the probability simplex
                let sa = a.sqrt();
                let sb = b.sqrt();
                let v = (1.0 - t) * sa + t * sb;
                v * v
            })
            .collect();
        path.push(MetricDistribution::new(p.labels.clone(), interp));
    }
    path
}

/// Natural gradient: transform a Euclidean gradient using the Fisher information matrix.
pub fn natural_gradient(fim: &DMatrix<f64>, euclidean_grad: &DVector<f64>) -> DVector<f64> {
    // Use pseudo-inverse for numerical stability
    let svd = fim.clone().svd(true, true);
    match svd.solve(&euclidean_grad.clone(), 1e-10) {
        Ok(sol) => sol,
        Err(_) => euclidean_grad.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let d = MetricDistribution::uniform(4);
        assert_eq!(d.probabilities.nrows(), 4);
        let sum: f64 = d.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization() {
        let d = MetricDistribution::new(
            vec!["a".into(), "b".into()],
            vec![2.0, 3.0],
        );
        let sum: f64 = d.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((d.probabilities[0] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform() {
        let d = MetricDistribution::uniform(4);
        let expected = 4.0_f64.ln();
        assert!((d.entropy() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_deterministic() {
        let d = MetricDistribution::new(
            vec!["a".into(), "b".into()],
            vec![1.0, 0.0],
        );
        assert!((d.entropy()).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_same() {
        let d = MetricDistribution::new(
            vec!["a".into(), "b".into()],
            vec![0.5, 0.5],
        );
        assert!((d.kl_divergence(&d)).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.8, 0.2]);
        let q = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        assert!(p.kl_divergence(&q) > 0.0);
    }

    #[test]
    fn test_fisher_information_diagonal() {
        let d = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let fim = d.fisher_information();
        assert!((fim[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((fim[(1, 1)] - 2.0).abs() < 1e-10);
        assert!((fim[(0, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_distance() {
        let p = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.9, 0.1]);
        let q = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.1, 0.9]);
        let dist = fisher_rao_distance(&p, &q);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_fisher_rao_distance_same() {
        let d = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let dist = fisher_rao_distance(&d, &d);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_bhattacharyya() {
        let p = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let q = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let bc = bhattacharyya_coefficient(&p, &q);
        assert!((bc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bhattacharyya_distance_orthogonal() {
        let p = MetricDistribution::new(vec!["a".into(), "b".into()], vec![1.0, 0.0]);
        let q = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.0, 1.0]);
        let dist = bhattacharyya_distance(&p, &q);
        assert!(dist.is_infinite());
    }

    #[test]
    fn test_rao_geodesic() {
        let p = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.9, 0.1]);
        let q = MetricDistribution::new(vec!["a".into(), "b".into()], vec![0.1, 0.9]);
        let path = rao_geodesic(&p, &q, 5);
        assert_eq!(path.len(), 6);
        // Each point should sum to ~1
        for pt in &path {
            let sum: f64 = pt.probabilities.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "sum = {}", sum);
        }
    }

    #[test]
    fn test_natural_gradient() {
        let fim = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]);
        let grad = DVector::from_vec(vec![1.0, 1.0]);
        let nat = natural_gradient(&fim, &grad);
        assert!((nat[0] - 0.5).abs() < 1e-10);
        assert!((nat[1] - 0.5).abs() < 1e-10);
    }
}
