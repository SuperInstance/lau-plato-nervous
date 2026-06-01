# lau-plato-nervous

**The nervous system connecting PLATO rooms to the deep math ecosystem.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests: 117](https://img.shields.io/badge/tests-117-brightgreen.svg)]()

---

## What This Does

PLATO is the monitoring and distillation system for SuperInstance. It has **rooms** (monitoring targets), **alerts** (severity-tagged events), **metrics** (continuous measurements), **config** (settings), **history** (event logs), and **health** (system status).

`lau-plato-nervous` is the **nervous system** that connects these PLATO concepts to deep mathematical analysis. It provides 10 modules:

| Module | Mathematical Lens | PLATO Concept |
|--------|------------------|---------------|
| `room_sheaf` | Sheaf theory | Rooms as open sets with local data |
| `alert_spectral` | Fourier analysis | Alert time series → frequency decomposition |
| `alert_cohomology` | Sheaf cohomology | Alert dependency graphs → missed alerts as H¹ classes |
| `metric_information` | Information geometry | Metric distributions → Fisher metric, KL divergence |
| `capacity_spectral` | Markov chains | System transitions → spectral gap, mixing time |
| `health_conservation` | Thermodynamics | Health measurements → energy/entropy conservation |
| `history_homology` | Topological data analysis | Event history → Vietoris-Rips complex, persistent homology |
| `distillation_transport` | Optimal transport | Knowledge distillation → Sinkhorn, Wasserstein distance |
| `config_category` | Category theory | Configurations as objects, reconfigurations as morphisms |
| `nervous_system` | Event bus | Pub/sub backbone connecting rooms to all analyzers |

Every PLATO concept gets a rigorous mathematical treatment. Alerts aren't just logged — they're decomposed into frequencies. Rooms aren't just containers — they're open sets in a sheaf. Config changes aren't just applied — they're morphisms in a category with composability proofs.

---

## Key Idea

> **"The nervous system is the mathematical brain of PLATO."**

PLATO rooms emit events (alerts, metric updates, health checks). The nervous system routes these events through a mathematical analysis pipeline:

```
Room emits alert →
  ├─ Fourier decomposition (is it periodic?)
  ├─ Cohomology check (did a dependent alert fail to fire?)
  ├─ Entropy computation (how surprising is this?)
  └─ Routing to subscribers
```

The result: every PLATO observation is simultaneously a data point *and* a mathematical object.

---

## Install

```toml
[dependencies]
lau-plato-nervous = "0.1.0"
```

Requires **Rust 2021 edition**. Dependencies: `serde`, `nalgebra` (with `serde-serialize`), `serde_json`.

---

## Quick Start

```rust
use lau_plato_nervous::{NervousSystem, NervousEventKind, Subscriber};

// Create the nervous system
let mut ns = NervousSystem::new();
ns.register_room("server-1");
ns.register_room("server-2");

// Subscribe an alert handler
let mut handler = Subscriber::new("alert-handler", "Handle alerts");
handler.subscribe(NervousEventKind::Alert);
handler.room_filter = Some("server-1".into());
ns.register_subscriber(handler);

// Emit events
let matched = ns.emit_alert("server-1", 0.0, "CPU at 95%", 0.9);
// → matched = ["alert-handler"]

let matched2 = ns.emit_metric("server-2", 1.0, "memory=4.2GB");

// Query the event log
let alerts = ns.events_of_kind(&NervousEventKind::Alert);
let high = ns.high_severity_events(0.8);
let room_events = ns.events_for_room("server-1");
```

### Room Sheaves

```rust
use lau_plato_nervous::room_sheaf::{RoomSheaf, RoomSection};
use nalgebra::DVector;

let mut sheaf = RoomSheaf::new();
sheaf.add_section(RoomSection {
    room_id: "server-1".into(),
    metrics: DVector::from_vec(vec![0.95, 0.80, 0.60]), // CPU, memory, disk
    labels: HashMap::new(),
});

let global = sheaf.global_section().unwrap(); // Aggregate all rooms
sheaf.check_gluing_axiom(); // Are rooms consistent?
```

### Alert Spectral Analysis

```rust
use lau_plato_nervous::alert_spectral::{spectral_decompose, detect_periodicity};

let samples = vec![0.1, 0.3, 0.1, 0.3, 0.1, 0.3]; // Periodic alerts
let decomp = spectral_decompose(&samples, 1.0);
println!("Dominant frequency: {} Hz", decomp.frequencies[decomp.dominant_index]);

let (freq, strength) = detect_periodicity(&samples, 1.0);
```

### Distillation via Optimal Transport

```rust
use lau_plato_nervous::distillation_transport::{distill, KnowledgeDistribution};

let teacher = KnowledgeDistribution::new("teacher-v1",
    vec!["topic-a".into(), "topic-b".into()], vec![0.8, 0.2]);
let student = KnowledgeDistribution::new("student-light",
    vec!["topic-a".into(), "topic-b".into()], vec![0.5, 0.5]);

let result = distill(&teacher, &student, 0.1);
println!("Teacher retention: {:.2}", result.teacher_retention);
println!("Student gain: {:.2}", result.student_gain);
println!("Efficiency: {:.2}", result.efficiency);
```

---

## API Reference

### `nervous_system` — Event Bus

The central pub/sub backbone.

| Type | Description |
|------|-------------|
| `NervousSystem` | Event bus with rooms, subscribers, and event log |
| `NervousEvent` | Timestamped event with kind, source, payload, severity |
| `NervousEventKind` | Alert, MetricUpdate, ConfigChange, HealthCheck, Distillation, AnalysisResult, Custom |
| `Subscriber` | Filtered consumer with kind subscriptions and optional room filter |

```rust
ns.register_room("room-1");
ns.register_subscriber(sub);
ns.emit_alert(room, ts, msg, severity) → Vec<matched_ids>
ns.emit_metric(room, ts, payload) → Vec<matched_ids>
ns.emit_health(room, ts, payload, severity) → Vec<matched_ids>
ns.emit_analysis(room, ts, payload) → Vec<matched_ids>
ns.events_for_room(room) → Vec<&NervousEvent>
ns.events_of_kind(&kind) → Vec<&NervousEvent>
ns.high_severity_events(threshold) → Vec<&NervousEvent>
ns.clear_log();
```

### `room_sheaf` — Rooms as Sheaves

| Type | Description |
|------|-------------|
| `RoomSection` | Local data at a room: metrics vector + labels |
| `RoomSheaf` | Sheaf of room sections with restriction maps |

```rust
sheaf.add_section(section);
sheaf.restrict("room-1", RoomSheaf::aggregate_restriction); // Project
sheaf.global_section();     // Aggregate all rooms
sheaf.check_gluing_axiom(); // Consistency check
sheaf.stalk("room-1");      // Direct limit at room
```

### `alert_spectral` — Fourier Analysis of Alerts

```rust
spectral_decompose(samples, sample_rate) → SpectralDecomposition
detect_periodicity(samples, sample_rate) → (dominant_freq, strength)
build_alert_timeline(events, window, step) → AlertTimeline
```

Returns frequencies, amplitudes, phases, dominant frequency, and SNR.

### `alert_cohomology` — Sheaf Cohomology on Alerts

| Type | Description |
|------|-------------|
| `Alert` | An alert with severity, dependencies, and fired status |
| `AlertGraph` | Dependency graph of alerts |
| `CohomologyResult` | H⁰ (components) + H¹ (missed alerts) |

```rust
let graph = AlertGraph::new();
graph.add_alert(alert);
graph.add_edge(src, tgt, weight);

graph.compute_h0();  // Connected components
graph.compute_h1();  // Missed alerts: deps fired but alert didn't
compute_cohomology(&graph) → CohomologyResult
```

**Key insight**: H¹ measures "holes" in the alert dependency chain — alerts that *should* have fired (all dependencies fired) but *didn't*. These are bugs or gaps in monitoring.

### `metric_information` — Information Geometry

```rust
let dist = MetricDistribution::new(labels, probabilities);
dist.entropy();                        // Shannon entropy
dist.kl_divergence(&other);            // KL(self || other)
dist.fisher_information();             // Fisher information matrix (diagonal)

fisher_rao_distance(&p, &q);          // √(KL(p||q) + KL(q||p))
bhattacharyya_coefficient(&p, &q);    // Σ √(pᵢqᵢ)
bhattacharyya_distance(&p, &q);       // -ln(BC)
rao_geodesic(&p, &q, steps);          // Fisher-Rao geodesic path
natural_gradient(&fim, &grad);        // F⁻¹ ∇ (via SVD pseudo-inverse)
```

### `capacity_spectral` — Spectral Gap & Mixing

```rust
let tg = TransitionGraph::from_adjacency(&adj);
tg.spectral_gap();              // 1 - |λ₂| — convergence speed
tg.mixing_time(epsilon);       // t_mix(ε) ≈ (1/γ) ln(1/ε)
tg.stationary_distribution();  // Left eigenvector of eigenvalue 1
tg.conductance();              // Cheeger constant approximation
tg.total_variation_distance(&current); // Distance to equilibrium
```

### `health_conservation` — Thermodynamic Health Laws

```rust
let mut tracker = ConservationTracker::new(total_capacity);
tracker.record(measurement);
tracker.check_energy_conservation() → ConservationReport;
tracker.check_capacity_conservation() → bool;
tracker.total_entropy();
tracker.entropy_production_rate();   // dS/dt per room
tracker.health_efficiency();         // work per unit entropy
tracker.carnot_efficiency();         // Thermodynamic bound
tracker.latest_energy_vector();      // nalgebra DVector for analysis
```

### `history_homology` — Topological Data Analysis

```rust
let events = vec![make_event("e1", 0.0), make_event("e2", 0.5), ...];
let complex = EventComplex::build_rips(events, epsilon, max_dim);
complex.betti_numbers();           // [β₀, β₁, β₂, ...]
complex.connected_components();
complex.boundary_matrix(k);        // ∂ₖ: k-simplices → (k-1)-simplices

persistence_barcode_dim0(&events, &epsilons); // Birth/death times
```

### `distillation_transport` — Knowledge Distillation as OT

```rust
let teacher = KnowledgeDistribution::new("teacher", concepts, masses);
let student = KnowledgeDistribution::new("student", concepts, masses);

// Cost matrices
CostMatrix::euclidean(&teacher, &student);
CostMatrix::kl_cost(&teacher, &student);

// Sinkhorn algorithm (entropic regularization)
sinkhorn(&teacher, &student, &cost, reg, max_iter, tol) → TransportPlan

// One-shot distillation
distill(&teacher, &student, regularization) → DistillationResult
// Returns: teacher_retention, student_gain, efficiency

// Wasserstein distance
wasserstein_distance(&teacher, &student, regularization) → f64
```

### `config_category` — Configuration as Category Theory

```rust
let mut cat = ConfigCategory::new();
cat.add_config(config);

let morphism = ConfigCategory::compose(&f, &g); // g ∘ f
ConfigCategory::check_associativity(&f, &g, &h);
cat.find_cheapest_path("cfg-v1", "cfg-v3"); // Dijkstra on reconfigurations
```

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PLATO Rooms                           │
│  server-1, server-2, database, cache, ...               │
└────────────────────────┬────────────────────────────────┘
                         │ Events (alerts, metrics, health)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Nervous System (Event Bus)                   │
│  Pub/sub routing, event log, severity filtering          │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  ┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
  │  Room   │ │ Alert  │ │ Alert  │ │ Metric   │
  │  Sheaf  │ │Spectral│ │Cohomol.│ │Information│
  │(local   │ │(Fourier│ │(H¹=    │ │(Fisher,  │
  │ data)   │ │ decomp)│ │ missed)│ │ KL, Rao) │
  └─────────┘ └────────┘ └────────┘ └──────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Capacity │ │  Health  │ │ History  │ │ Distill- │
  │ Spectral │ │Conserv-  │ │Homology  │ │ ation    │
  │(mixing,  │ │ation     │ │(Rips,    │ │Transport │
  │ Cheeger) │ │(thermo)  │ │ barcode) │ │(Sinkhorn)│
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
       │
       ▼
  ┌──────────┐
  │  Config  │
  │ Category │
  │(Dijkstra │
  │ reconfig)│
  └──────────┘
```

### Event Flow

1. A PLATO room emits an event (alert, metric update, health check)
2. The `NervousSystem` routes to matching subscribers
3. Each mathematical module analyzes its aspect:
   - **Room sheaf**: updates local sections, checks gluing
   - **Alert spectral**: decomposes time series into frequencies
   - **Alert cohomology**: detects missed alerts (H¹ classes)
   - **Metric information**: computes Fisher distances between metric states
   - **Capacity spectral**: checks convergence speed
   - **Health conservation**: verifies energy/entropy budgets
   - **History homology**: builds Vietoris-Rips complex from temporal proximity
4. Results flow back through the event bus to other rooms and analyzers

---

## The Math

### Sheaf Cohomology on Alerts

Given an alert dependency graph, we compute sheaf cohomology:

- **H⁰** = connected components of the alert graph (clusters of related alerts)
- **H¹** = "missed alerts" — alerts whose dependencies all fired but which didn't fire themselves

This is a genuine topological invariant applied to monitoring: H¹ measures the **obstruction** to having a complete alert chain. If H¹ ≠ 0, something is wrong with the monitoring configuration.

### Persistent Homology of Event History

Events in PLATO history are modeled as points in a temporal metric space. For varying proximity thresholds `ε`, we build **Vietoris-Rips complexes** and compute their homology. The resulting **persistence barcode** reveals:

- Long-lived H⁰ features → stable event clusters (groups of events that always co-occur)
- Long-lived H¹ features → recurring cycles (periodic patterns of events)

### Information Geometry of Metrics

PLATO metrics are modeled as probability distributions over outcome categories. The **Fisher information metric** gives a Riemannian structure on the manifold of these distributions:

$$g_{ij} = \mathbb{E}\left[\frac{\partial \ln p}{\partial \theta_i} \frac{\partial \ln p}{\partial \theta_j}\right]$$

Distances (Fisher-Rao, Bhattacharyya) and geodesics are computed in this intrinsic geometry, providing **coordinate-free** comparisons between metric states.

### Knowledge Distillation as Optimal Transport

Distillation from teacher to student is framed as finding the minimum-cost transport plan between their knowledge distributions:

$$W(μ_T, μ_S) = \inf_{\gamma \in \Pi(μ_T, μ_S)} \sum_{i,j} c_{ij} \gamma_{ij}$$

Solved efficiently via **Sinkhorn iterations** with entropic regularization:

$$\min_\gamma \sum c_{ij} \gamma_{ij} + \epsilon \sum \gamma_{ij} \ln \gamma_{ij}$$

### Configuration as Category

PLATO configurations form a category where:
- **Objects** = configuration states (key-value parameter maps)
- **Morphisms** = valid reconfigurations (partial updates with costs)
- **Composition** = chaining reconfigurations (changes merge, costs add)
- **Identity** = no-op reconfiguration

Finding the cheapest reconfiguration path is Dijkstra's algorithm on the category's morphism graph.

---

## Test Coverage

**117 tests** across all 10 modules:

| Module | Tests | Key Verifications |
|--------|-------|-------------------|
| `room_sheaf` | 10 | Sections, restrictions, gluing axiom, serialization |
| `alert_spectral` | 12 | DFT, periodicity detection, timeline, SNR |
| `alert_cohomology` | 12 | H⁰ components, H¹ missed alerts, coboundary |
| `metric_information` | 13 | Entropy, KL, Fisher, Bhattacharyya, geodesic, natural gradient |
| `capacity_spectral` | 11 | Spectral gap, mixing time, stationary distribution, conductance |
| `health_conservation` | 11 | Energy/capacity conservation, entropy production, Carnot |
| `history_homology` | 10 | Rips complex, Betti numbers, persistence barcode, boundary matrices |
| `distillation_transport` | 10 | Sinkhorn convergence, mass conservation, KL cost, distillation |
| `config_category` | 11 | Identity, composition, associativity, Dijkstra path finding |
| `nervous_system` | 17 | Event emission, routing, filtering, log truncation, serialization |

---

## License

MIT
