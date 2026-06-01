//! Knowledge distillation as optimal transport between teacher/student distributions.
//!
//! In PLATO, distillation moves knowledge from teacher models to student models.
//! This module frames it as optimal transport: finding the minimum-cost mapping
//! between teacher and student distributions.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A knowledge distribution (teacher or student).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDistribution {
    /// Name (e.g., "teacher-v1", "student-light").
    pub name: String,
    /// Support points (concepts/topics).
    pub concepts: Vec<String>,
    /// Probability mass over concepts.
    pub masses: DVector<f64>,
}

impl KnowledgeDistribution {
    pub fn new(name: &str, concepts: Vec<String>, masses: Vec<f64>) -> Self {
        let total: f64 = masses.iter().sum();
        let normalized = if total > 0.0 {
            DVector::from_vec(masses.into_iter().map(|m| m / total).collect())
        } else {
            DVector::from_element(concepts.len(), 1.0 / concepts.len() as f64)
        };
        Self {
            name: name.to_string(),
            concepts,
            masses: normalized,
        }
    }

    /// Uniform distribution.
    pub fn uniform(name: &str, n: usize) -> Self {
        Self {
            name: name.to_string(),
            concepts: (0..n).map(|i| format!("concept-{}", i)).collect(),
            masses: DVector::from_element(n, 1.0 / n as f64),
        }
    }
}

/// Cost matrix between teacher and student concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMatrix {
    pub teacher_concepts: Vec<String>,
    pub student_concepts: Vec<String>,
    pub costs: DMatrix<f64>,
}

impl CostMatrix {
    /// Euclidean cost: cost(i,j) = |teacher_i - student_j|.
    /// Uses concept index as a proxy position.
    pub fn euclidean(teacher: &KnowledgeDistribution, student: &KnowledgeDistribution) -> Self {
        let n_t = teacher.concepts.len();
        let n_s = student.concepts.len();
        let mut costs = DMatrix::zeros(n_t, n_s);
        for i in 0..n_t {
            for j in 0..n_s {
                costs[(i, j)] = (teacher.masses[i] - student.masses[j]).abs();
            }
        }
        Self {
            teacher_concepts: teacher.concepts.clone(),
            student_concepts: student.concepts.clone(),
            costs,
        }
    }

    /// KL-based cost: cost(i,j) = p_i * ln(p_i / q_j).
    pub fn kl_cost(teacher: &KnowledgeDistribution, student: &KnowledgeDistribution) -> Self {
        let n_t = teacher.concepts.len();
        let n_s = student.concepts.len();
        let mut costs = DMatrix::zeros(n_t, n_s);
        for i in 0..n_t {
            for j in 0..n_s {
                let p = teacher.masses[i];
                let q = student.masses[j];
                costs[(i, j)] = if p > 0.0 && q > 0.0 {
                    p * (p / q).ln()
                } else {
                    f64::MAX
                };
            }
        }
        Self {
            teacher_concepts: teacher.concepts.clone(),
            student_concepts: student.concepts.clone(),
            costs,
        }
    }
}

/// Transport plan: mapping from teacher to student.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportPlan {
    /// The transport matrix T[i,j] = amount transported from i to j.
    pub plan: DMatrix<f64>,
    /// Total transport cost.
    pub total_cost: f64,
    /// Teacher distribution name.
    pub teacher: String,
    /// Student distribution name.
    pub student: String,
}

/// Solve the optimal transport problem using Sinkhorn iterations (entropic regularization).
pub fn sinkhorn(
    teacher: &KnowledgeDistribution,
    student: &KnowledgeDistribution,
    cost: &CostMatrix,
    regularization: f64,
    max_iterations: usize,
    tolerance: f64,
) -> TransportPlan {
    let n_t = teacher.masses.nrows();
    let n_s = student.masses.nrows();

    let a = teacher.masses.clone();
    let b = student.masses.clone();

    // Kernel: K = exp(-C / epsilon)
    let mut k = DMatrix::zeros(n_t, n_s);
    for i in 0..n_t {
        for j in 0..n_s {
            k[(i, j)] = (-cost.costs[(i, j)] / regularization).exp();
        }
    }

    // Sinkhorn iterations: u, v vectors
    let mut u = DVector::from_element(n_t, 1.0 / n_t as f64);
    let mut v = DVector::from_element(n_s, 1.0 / n_s as f64);

    for _ in 0..max_iterations {
        // u = a / (K * v)
        let kv = &k * &v;
        let u_new = a.component_div(&kv);

        // v = b / (K^T * u)
        let ktu = k.transpose() * &u_new;
        let v_new = b.component_div(&ktu);

        // Check convergence
        let u_diff: f64 = (&u_new - &u).iter().map(|x| x.abs()).sum();
        let v_diff: f64 = (&v_new - &v).iter().map(|x| x.abs()).sum();

        u = u_new;
        v = v_new;

        if u_diff + v_diff < tolerance {
            break;
        }
    }

    // Transport plan: diag(u) * K * diag(v)
    let mut plan = DMatrix::zeros(n_t, n_s);
    for i in 0..n_t {
        for j in 0..n_s {
            plan[(i, j)] = u[i] * k[(i, j)] * v[j];
        }
    }

    let total_cost = (&plan.component_mul(&cost.costs)).iter().sum();

    TransportPlan {
        plan,
        total_cost,
        teacher: teacher.name.clone(),
        student: student.name.clone(),
    }
}

/// Compute the Wasserstein-1 distance (approximate via Sinkhorn).
pub fn wasserstein_distance(
    teacher: &KnowledgeDistribution,
    student: &KnowledgeDistribution,
    regularization: f64,
) -> f64 {
    let cost = CostMatrix::euclidean(teacher, student);
    let transport = sinkhorn(teacher, student, &cost, regularization, 100, 1e-6);
    transport.total_cost
}

/// Distillation result: how much knowledge was transferred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationResult {
    pub transport: TransportPlan,
    pub teacher_retention: f64,
    pub student_gain: f64,
    pub efficiency: f64,
}

/// Perform knowledge distillation via optimal transport.
pub fn distill(
    teacher: &KnowledgeDistribution,
    student: &KnowledgeDistribution,
    regularization: f64,
) -> DistillationResult {
    let cost = CostMatrix::kl_cost(teacher, student);
    let transport = sinkhorn(teacher, student, &cost, regularization, 200, 1e-8);

    let teacher_mass: f64 = teacher.masses.iter().sum();
    let student_mass: f64 = student.masses.iter().sum();
    let transport_mass: f64 = transport.plan.iter().sum();

    let teacher_retention = if teacher_mass > 0.0 {
        1.0 - (transport.total_cost / teacher_mass).min(1.0)
    } else {
        0.0
    };

    let student_gain = if student_mass > 0.0 {
        transport_mass / student_mass
    } else {
        0.0
    };

    let efficiency = if transport.total_cost > 0.0 {
        transport_mass / transport.total_cost
    } else {
        f64::INFINITY
    };

    DistillationResult {
        transport,
        teacher_retention,
        student_gain,
        efficiency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let d = KnowledgeDistribution::uniform("test", 3);
        assert_eq!(d.concepts.len(), 3);
        let sum: f64 = d.masses.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_distribution() {
        let d = KnowledgeDistribution::new("t", vec!["a".into(), "b".into()], vec![3.0, 7.0]);
        assert!((d.masses[0] - 0.3).abs() < 1e-10);
        assert!((d.masses[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_cost_matrix() {
        let t = KnowledgeDistribution::new("t", vec!["a".into()], vec![1.0]);
        let s = KnowledgeDistribution::new("s", vec!["b".into()], vec![1.0]);
        let cost = CostMatrix::euclidean(&t, &s);
        assert_eq!(cost.costs.nrows(), 1);
        assert_eq!(cost.costs.ncols(), 1);
        assert!((cost.costs[(0, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_sinkhorn_converges() {
        let t = KnowledgeDistribution::new("t", vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let s = KnowledgeDistribution::new("s", vec!["c".into(), "d".into()], vec![0.5, 0.5]);
        let cost = CostMatrix::euclidean(&t, &s);
        let transport = sinkhorn(&t, &s, &cost, 0.1, 100, 1e-6);
        assert!(transport.total_cost >= 0.0);
        // Plan should have positive entries
        assert!(transport.plan.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_sinkhorn_mass_conservation() {
        let t = KnowledgeDistribution::new("t", vec!["a".into(), "b".into()], vec![0.6, 0.4]);
        let s = KnowledgeDistribution::new("s", vec!["c".into(), "d".into()], vec![0.3, 0.7]);
        let cost = CostMatrix::euclidean(&t, &s);
        let transport = sinkhorn(&t, &s, &cost, 0.01, 200, 1e-8);
        let row_sums: Vec<f64> = (0..2).map(|i| transport.plan.row(i).iter().sum()).collect();
        for (i, rs) in row_sums.iter().enumerate() {
            assert!((rs - t.masses[i]).abs() < 0.1, "row {} sum = {}, expected {}", i, rs, t.masses[i]);
        }
    }

    #[test]
    fn test_wasserstein_distance_same() {
        let d = KnowledgeDistribution::new("d", vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let dist = wasserstein_distance(&d, &d, 0.01);
        assert!(dist.abs() < 0.1, "distance to self should be ~0, got {}", dist);
    }

    #[test]
    fn test_wasserstein_distance_different() {
        let t = KnowledgeDistribution::new("t", vec!["a".into(), "b".into()], vec![0.9, 0.1]);
        let s = KnowledgeDistribution::new("s", vec!["a".into(), "b".into()], vec![0.1, 0.9]);
        let dist = wasserstein_distance(&t, &s, 0.01);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_distill() {
        let t = KnowledgeDistribution::new("t", vec!["a".into(), "b".into()], vec![0.8, 0.2]);
        let s = KnowledgeDistribution::new("s", vec!["a".into(), "b".into()], vec![0.5, 0.5]);
        let result = distill(&t, &s, 0.1);
        assert!(result.teacher_retention >= 0.0 && result.teacher_retention <= 1.0);
        assert!(result.student_gain > 0.0);
    }

    #[test]
    fn test_transport_plan_serialization() {
        let t = KnowledgeDistribution::new("t", vec!["a".into()], vec![1.0]);
        let s = KnowledgeDistribution::new("s", vec!["b".into()], vec![1.0]);
        let cost = CostMatrix::euclidean(&t, &s);
        let transport = sinkhorn(&t, &s, &cost, 0.1, 50, 1e-6);
        let json = serde_json::to_string(&transport).unwrap();
        let back: TransportPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back.teacher, "t");
    }

    #[test]
    fn test_cost_matrix_serialization() {
        let t = KnowledgeDistribution::new("t", vec!["a".into()], vec![1.0]);
        let s = KnowledgeDistribution::new("s", vec!["b".into()], vec![1.0]);
        let cost = CostMatrix::euclidean(&t, &s);
        let json = serde_json::to_string(&cost).unwrap();
        assert!(!json.is_empty());
    }
}
