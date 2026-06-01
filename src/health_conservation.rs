//! Conservation laws for system health: energy, entropy, capacity budgets.
//!
//! The health of a PLATO system obeys conservation principles:
//! - Energy: total resource usage is conserved across rooms
//! - Entropy: system disorder tends to increase (2nd law analog)
//! - Capacity: total capacity is bounded

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Energy type for conservation tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergyType {
    Cpu,
    Memory,
    Network,
    Disk,
}

/// A health measurement for a room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMeasurement {
    pub room_id: String,
    pub timestamp: f64,
    pub energy: HashMap<EnergyType, f64>,
    pub entropy: f64,
    pub capacity_used: f64,
    pub capacity_total: f64,
}

impl HealthMeasurement {
    /// Total energy across all types.
    pub fn total_energy(&self) -> f64 {
        self.energy.values().sum()
    }

    /// Capacity utilization ratio [0, 1].
    pub fn utilization(&self) -> f64 {
        if self.capacity_total > 0.0 {
            self.capacity_used / self.capacity_total
        } else {
            0.0
        }
    }

    /// Free energy (available capacity).
    pub fn free_energy(&self) -> f64 {
        (self.capacity_total - self.capacity_used).max(0.0)
    }
}

/// Conservation tracker for the entire PLATO system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationTracker {
    /// Total system capacity budget.
    pub total_capacity: f64,
    /// Per-room measurements.
    pub rooms: HashMap<String, Vec<HealthMeasurement>>,
}

impl ConservationTracker {
    pub fn new(total_capacity: f64) -> Self {
        Self {
            total_capacity,
            rooms: HashMap::new(),
        }
    }

    /// Record a health measurement.
    pub fn record(&mut self, measurement: HealthMeasurement) {
        self.rooms
            .entry(measurement.room_id.clone())
            .or_default()
            .push(measurement);
    }

    /// Check energy conservation: sum of room energies should equal total.
    pub fn check_energy_conservation(&self) -> ConservationReport {
        let mut total_energy = 0.0;
        let mut room_energies = HashMap::new();
        for (room_id, measurements) in &self.rooms {
            if let Some(last) = measurements.last() {
                let e = last.total_energy();
                total_energy += e;
                room_energies.insert(room_id.clone(), e);
            }
        }
        ConservationReport {
            total_energy,
            room_energies,
            budget: self.total_capacity,
            deficit: (self.total_capacity - total_energy).max(0.0),
            surplus: (total_energy - self.total_capacity).max(0.0),
        }
    }

    /// Compute total system entropy.
    pub fn total_entropy(&self) -> f64 {
        self.rooms
            .values()
            .filter_map(|m| m.last())
            .map(|m| m.entropy)
            .sum()
    }

    /// Compute entropy production rate (time derivative).
    pub fn entropy_production_rate(&self) -> HashMap<String, f64> {
        let mut rates = HashMap::new();
        for (room_id, measurements) in &self.rooms {
            if measurements.len() >= 2 {
                let last = &measurements[measurements.len() - 1];
                let prev = &measurements[measurements.len() - 2];
                let dt = last.timestamp - prev.timestamp;
                if dt > 0.0 {
                    rates.insert(room_id.clone(), (last.entropy - prev.entropy) / dt);
                }
            }
        }
        rates
    }

    /// Check capacity conservation: sum of used <= total.
    pub fn check_capacity_conservation(&self) -> bool {
        let total_used: f64 = self
            .rooms
            .values()
            .filter_map(|m| m.last())
            .map(|m| m.capacity_used)
            .sum();
        total_used <= self.total_capacity
    }

    /// Compute the health efficiency: useful work per unit entropy.
    pub fn health_efficiency(&self) -> f64 {
        let total_entropy = self.total_entropy();
        if total_entropy > 0.0 {
            let total_used: f64 = self
                .rooms
                .values()
                .filter_map(|m| m.last())
                .map(|m| m.capacity_used)
                .sum();
            total_used / total_entropy
        } else {
            f64::INFINITY
        }
    }

    /// Compute a Carnot-like efficiency bound.
    pub fn carnot_efficiency(&self) -> f64 {
        let max_entropy = self.rooms.values().filter_map(|m| m.last()).map(|m| m.entropy).fold(0.0f64, f64::max);
        let min_entropy = self.rooms.values().filter_map(|m| m.last()).map(|m| m.entropy).fold(f64::MAX, f64::min);
        if max_entropy > 0.0 {
            1.0 - min_entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Get latest measurements as a vector for analysis.
    pub fn latest_energy_vector(&self) -> DVector<f64> {
        let vals: Vec<f64> = self
            .rooms
            .values()
            .filter_map(|m| m.last())
            .map(|m| m.total_energy())
            .collect();
        if vals.is_empty() {
            DVector::zeros(0)
        } else {
            DVector::from_vec(vals)
        }
    }

    /// Count rooms.
    pub fn len(&self) -> usize {
        self.rooms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rooms.is_empty()
    }
}

/// Report on energy conservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    pub total_energy: f64,
    pub room_energies: HashMap<String, f64>,
    pub budget: f64,
    pub deficit: f64,
    pub surplus: f64,
}

impl ConservationReport {
    pub fn is_balanced(&self, tolerance: f64) -> bool {
        self.surplus <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_measurement(room: &str, ts: f64, energy: f64, entropy: f64, used: f64, total: f64) -> HealthMeasurement {
        HealthMeasurement {
            room_id: room.into(),
            timestamp: ts,
            energy: [(EnergyType::Cpu, energy)].into(),
            entropy,
            capacity_used: used,
            capacity_total: total,
        }
    }

    #[test]
    fn test_total_energy() {
        let m = make_measurement("r1", 0.0, 10.0, 1.0, 5.0, 20.0);
        assert!((m.total_energy() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_utilization() {
        let m = make_measurement("r1", 0.0, 10.0, 1.0, 15.0, 20.0);
        assert!((m.utilization() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_free_energy() {
        let m = make_measurement("r1", 0.0, 10.0, 1.0, 15.0, 20.0);
        assert!((m.free_energy() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_tracker_record() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 5.0, 50.0));
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_energy_conservation_balanced() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 30.0, 1.0, 30.0, 50.0));
        tracker.record(make_measurement("r2", 0.0, 40.0, 1.0, 40.0, 50.0));
        let report = tracker.check_energy_conservation();
        assert!((report.total_energy - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_capacity_conservation_ok() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 40.0, 50.0));
        tracker.record(make_measurement("r2", 0.0, 10.0, 1.0, 50.0, 50.0));
        assert!(tracker.check_capacity_conservation());
    }

    #[test]
    fn test_capacity_conservation_exceeded() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 60.0, 50.0));
        tracker.record(make_measurement("r2", 0.0, 10.0, 1.0, 50.0, 50.0));
        assert!(!tracker.check_capacity_conservation());
    }

    #[test]
    fn test_entropy_production_rate() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 5.0, 50.0));
        tracker.record(make_measurement("r1", 1.0, 10.0, 3.0, 5.0, 50.0));
        let rates = tracker.entropy_production_rate();
        assert!((rates["r1"] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_health_efficiency() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 2.0, 10.0, 50.0));
        let eff = tracker.health_efficiency();
        assert!((eff - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_latest_energy_vector() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 5.0, 50.0));
        tracker.record(make_measurement("r2", 0.0, 20.0, 1.0, 5.0, 50.0));
        let v = tracker.latest_energy_vector();
        assert_eq!(v.nrows(), 2);
    }

    #[test]
    fn test_conservation_report_serialization() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 5.0, 50.0));
        let report = tracker.check_energy_conservation();
        let json = serde_json::to_string(&report).unwrap();
        let back: ConservationReport = serde_json::from_str(&json).unwrap();
        assert!((back.total_energy - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_carnot_efficiency() {
        let mut tracker = ConservationTracker::new(100.0);
        tracker.record(make_measurement("r1", 0.0, 10.0, 1.0, 5.0, 50.0));
        tracker.record(make_measurement("r2", 0.0, 10.0, 4.0, 5.0, 50.0));
        let eff = tracker.carnot_efficiency();
        assert!(eff >= 0.0 && eff <= 1.0);
    }
}
