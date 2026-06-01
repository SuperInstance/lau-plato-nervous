//! Unified event bus connecting all PLATO rooms to math analysis.
//!
//! The nervous system is a publish-subscribe event bus that connects PLATO rooms
//! to the mathematical analysis modules. Events flow from rooms through the bus
//! to analyzers, and analysis results flow back as signals.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Event types in the nervous system.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NervousEventKind {
    /// A room emitted an alert.
    Alert,
    /// A metric was updated.
    MetricUpdate,
    /// Configuration changed.
    ConfigChange,
    /// Health check result.
    HealthCheck,
    /// Distillation result.
    Distillation,
    /// Analysis result from a math module.
    AnalysisResult,
    /// Custom event.
    Custom(String),
}

/// An event on the nervous system bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NervousEvent {
    pub id: u64,
    pub kind: NervousEventKind,
    pub source_room: String,
    pub timestamp: f64,
    pub payload: String,
    pub severity: f64,
}

impl NervousEvent {
    pub fn new(kind: NervousEventKind, source_room: &str, timestamp: f64, payload: &str, severity: f64) -> Self {
        Self {
            id: EVENT_COUNTER.fetch_add(1, Ordering::Relaxed),
            kind,
            source_room: source_room.to_string(),
            timestamp,
            payload: payload.to_string(),
            severity,
        }
    }
}

/// A subscriber that receives events matching a filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscriber {
    pub id: String,
    pub name: String,
    /// Event kinds this subscriber cares about.
    pub subscriptions: Vec<NervousEventKind>,
    /// Room filter (None = all rooms).
    pub room_filter: Option<String>,
}

impl Subscriber {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            subscriptions: Vec::new(),
            room_filter: None,
        }
    }

    /// Subscribe to an event kind.
    pub fn subscribe(&mut self, kind: NervousEventKind) {
        if !self.subscriptions.contains(&kind) {
            self.subscriptions.push(kind);
        }
    }

    /// Check if this subscriber should receive an event.
    pub fn matches(&self, event: &NervousEvent) -> bool {
        let kind_match = self.subscriptions.is_empty()
            || self.subscriptions.contains(&event.kind);
        let room_match = self.room_filter.is_none()
            || self.room_filter.as_ref() == Some(&event.source_room);
        kind_match && room_match
    }
}

/// The nervous system event bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NervousSystem {
    pub rooms: Vec<String>,
    pub subscribers: HashMap<String, Subscriber>,
    pub event_log: Vec<NervousEvent>,
    /// Max events to keep in the log.
    pub max_log_size: usize,
    /// Routing table: event kind → subscriber IDs.
    pub routing: HashMap<String, Vec<String>>,
}

impl NervousSystem {
    pub fn new() -> Self {
        Self {
            rooms: Vec::new(),
            subscribers: HashMap::new(),
            event_log: Vec::new(),
            max_log_size: 10000,
            routing: HashMap::new(),
        }
    }

    /// Register a room.
    pub fn register_room(&mut self, room_id: &str) {
        if !self.rooms.contains(&room_id.to_string()) {
            self.rooms.push(room_id.to_string());
        }
    }

    /// Register a subscriber.
    pub fn register_subscriber(&mut self, subscriber: Subscriber) {
        for kind in &subscriber.subscriptions {
            let key = format!("{:?}", kind);
            self.routing
                .entry(key)
                .or_default()
                .push(subscriber.id.clone());
        }
        self.subscribers.insert(subscriber.id.clone(), subscriber);
    }

    /// Emit an event to the bus.
    pub fn emit(&mut self, event: NervousEvent) -> Vec<String> {
        let mut matched_subscribers = Vec::new();
        for (_, sub) in &self.subscribers {
            if sub.matches(&event) {
                matched_subscribers.push(sub.id.clone());
            }
        }

        self.event_log.push(event);
        if self.event_log.len() > self.max_log_size {
            self.event_log.remove(0);
        }

        matched_subscribers
    }

    /// Emit an alert event.
    pub fn emit_alert(&mut self, room: &str, timestamp: f64, message: &str, severity: f64) -> Vec<String> {
        let event = NervousEvent::new(NervousEventKind::Alert, room, timestamp, message, severity);
        self.emit(event)
    }

    /// Emit a metric update.
    pub fn emit_metric(&mut self, room: &str, timestamp: f64, payload: &str) -> Vec<String> {
        let event = NervousEvent::new(NervousEventKind::MetricUpdate, room, timestamp, payload, 0.0);
        self.emit(event)
    }

    /// Emit a health check.
    pub fn emit_health(&mut self, room: &str, timestamp: f64, payload: &str, severity: f64) -> Vec<String> {
        let event = NervousEvent::new(NervousEventKind::HealthCheck, room, timestamp, payload, severity);
        self.emit(event)
    }

    /// Emit an analysis result.
    pub fn emit_analysis(&mut self, room: &str, timestamp: f64, payload: &str) -> Vec<String> {
        let event = NervousEvent::new(NervousEventKind::AnalysisResult, room, timestamp, payload, 0.0);
        self.emit(event)
    }

    /// Get events for a specific room.
    pub fn events_for_room(&self, room_id: &str) -> Vec<&NervousEvent> {
        self.event_log
            .iter()
            .filter(|e| e.source_room == room_id)
            .collect()
    }

    /// Get events of a specific kind.
    pub fn events_of_kind(&self, kind: &NervousEventKind) -> Vec<&NervousEvent> {
        self.event_log
            .iter()
            .filter(|e| &e.kind == kind)
            .collect()
    }

    /// Get high-severity events (above threshold).
    pub fn high_severity_events(&self, threshold: f64) -> Vec<&NervousEvent> {
        self.event_log
            .iter()
            .filter(|e| e.severity > threshold)
            .collect()
    }

    /// Count events.
    pub fn event_count(&self) -> usize {
        self.event_log.len()
    }

    /// Count rooms.
    pub fn room_count(&self) -> usize {
        self.rooms.len()
    }

    /// Count subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Clear the event log.
    pub fn clear_log(&mut self) {
        self.event_log.clear();
    }
}

impl Default for NervousSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_nervous_system() {
        let ns = NervousSystem::new();
        assert_eq!(ns.room_count(), 0);
        assert_eq!(ns.subscriber_count(), 0);
        assert_eq!(ns.event_count(), 0);
    }

    #[test]
    fn test_register_room() {
        let mut ns = NervousSystem::new();
        ns.register_room("room-1");
        ns.register_room("room-2");
        assert_eq!(ns.room_count(), 2);
    }

    #[test]
    fn test_register_room_no_duplicate() {
        let mut ns = NervousSystem::new();
        ns.register_room("room-1");
        ns.register_room("room-1");
        assert_eq!(ns.room_count(), 1);
    }

    #[test]
    fn test_subscriber_matches_kind() {
        let mut sub = Subscriber::new("s1", "alert-handler");
        sub.subscribe(NervousEventKind::Alert);
        let event = NervousEvent::new(NervousEventKind::Alert, "r1", 0.0, "test", 0.5);
        assert!(sub.matches(&event));
    }

    #[test]
    fn test_subscriber_no_match_kind() {
        let mut sub = Subscriber::new("s1", "alert-handler");
        sub.subscribe(NervousEventKind::Alert);
        let event = NervousEvent::new(NervousEventKind::MetricUpdate, "r1", 0.0, "test", 0.0);
        assert!(!sub.matches(&event));
    }

    #[test]
    fn test_subscriber_room_filter() {
        let mut sub = Subscriber::new("s1", "room1-handler");
        sub.subscribe(NervousEventKind::Alert);
        sub.room_filter = Some("room-1".into());
        let event = NervousEvent::new(NervousEventKind::Alert, "room-2", 0.0, "test", 0.5);
        assert!(!sub.matches(&event));
    }

    #[test]
    fn test_subscriber_room_filter_match() {
        let mut sub = Subscriber::new("s1", "room1-handler");
        sub.subscribe(NervousEventKind::Alert);
        sub.room_filter = Some("room-1".into());
        let event = NervousEvent::new(NervousEventKind::Alert, "room-1", 0.0, "test", 0.5);
        assert!(sub.matches(&event));
    }

    #[test]
    fn test_emit_event() {
        let mut ns = NervousSystem::new();
        ns.register_room("r1");
        let mut sub = Subscriber::new("s1", "handler");
        sub.subscribe(NervousEventKind::Alert);
        ns.register_subscriber(sub);
        let matched = ns.emit_alert("r1", 0.0, "fire!", 0.9);
        assert_eq!(matched, vec!["s1"]);
        assert_eq!(ns.event_count(), 1);
    }

    #[test]
    fn test_emit_metric() {
        let mut ns = NervousSystem::new();
        ns.register_room("r1");
        ns.emit_metric("r1", 1.0, "cpu=90%");
        assert_eq!(ns.event_count(), 1);
    }

    #[test]
    fn test_events_for_room() {
        let mut ns = NervousSystem::new();
        ns.register_room("r1");
        ns.register_room("r2");
        ns.emit_alert("r1", 0.0, "a1", 0.5);
        ns.emit_alert("r2", 1.0, "a2", 0.5);
        ns.emit_alert("r1", 2.0, "a3", 0.5);
        assert_eq!(ns.events_for_room("r1").len(), 2);
    }

    #[test]
    fn test_events_of_kind() {
        let mut ns = NervousSystem::new();
        ns.emit_alert("r1", 0.0, "a1", 0.5);
        ns.emit_metric("r1", 1.0, "m1");
        assert_eq!(ns.events_of_kind(&NervousEventKind::Alert).len(), 1);
        assert_eq!(ns.events_of_kind(&NervousEventKind::MetricUpdate).len(), 1);
    }

    #[test]
    fn test_high_severity_events() {
        let mut ns = NervousSystem::new();
        ns.emit_alert("r1", 0.0, "low", 0.3);
        ns.emit_alert("r1", 1.0, "high", 0.9);
        ns.emit_alert("r1", 2.0, "mid", 0.5);
        let high = ns.high_severity_events(0.7);
        assert_eq!(high.len(), 1);
    }

    #[test]
    fn test_event_log_truncation() {
        let mut ns = NervousSystem::new();
        ns.max_log_size = 5;
        for i in 0..10 {
            ns.emit_alert("r1", i as f64, "msg", 0.5);
        }
        assert_eq!(ns.event_count(), 5);
    }

    #[test]
    fn test_clear_log() {
        let mut ns = NervousSystem::new();
        ns.emit_alert("r1", 0.0, "msg", 0.5);
        ns.clear_log();
        assert_eq!(ns.event_count(), 0);
    }

    #[test]
    fn test_event_serialization() {
        let evt = NervousEvent::new(NervousEventKind::Alert, "r1", 0.0, "test", 0.5);
        let json = serde_json::to_string(&evt).unwrap();
        let back: NervousEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source_room, "r1");
        assert_eq!(back.kind, NervousEventKind::Alert);
    }

    #[test]
    fn test_subscriber_subscribe_no_duplicate() {
        let mut sub = Subscriber::new("s1", "handler");
        sub.subscribe(NervousEventKind::Alert);
        sub.subscribe(NervousEventKind::Alert);
        assert_eq!(sub.subscriptions.len(), 1);
    }

    #[test]
    fn test_wildcard_subscriber() {
        let mut sub = Subscriber::new("s1", "all-handler");
        // No subscriptions = receives all events
        let event = NervousEvent::new(NervousEventKind::Custom("test".into()), "r1", 0.0, "x", 0.0);
        assert!(sub.matches(&event));
    }

    #[test]
    fn test_nervous_system_serialization() {
        let mut ns = NervousSystem::new();
        ns.register_room("r1");
        ns.emit_alert("r1", 0.0, "test", 0.5);
        let json = serde_json::to_string(&ns).unwrap();
        let back: NervousSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(back.room_count(), 1);
        assert_eq!(back.event_count(), 1);
    }
}
