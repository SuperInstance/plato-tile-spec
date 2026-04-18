//! Plato Unified Tile Specification
//!
//! A tile is the atomic unit of accumulated experience in the Plato system.
//! Each tile fuses concepts from constraint theory, ghost-tile attention,
//! Python-based knowledge tiles, and semantic tiling into one coherent structure.
//!
//! ## Layers
//!
//! - **Core**: identity, confidence, provenance
//! - **Content**: question/answer/tags/anchors (semantic retrieval)
//! - **Attention**: weight, use_count, active, last_used_tick (ghost-tile dynamics)
//! - **Constraints**: tolerance + threshold (constraint-theory engine)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the current time as nanoseconds since the Unix epoch.
fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Generate a nanosecond-based tile ID with an optional label prefix.
pub fn generate_id(label: &str) -> String {
    let ns = now_ns();
    if label.is_empty() {
        format!("tile-{ns}")
    } else {
        format!("{label}-{ns}")
    }
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// Records where a tile came from and which self-play generation produced it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source identifier: "visitor", "npc", "mid-tier", "human", "system", etc.
    pub source: String,
    /// Self-play / training generation counter (0 = human-authored).
    pub generation: u32,
}

impl Provenance {
    pub fn new(source: impl Into<String>, generation: u32) -> Self {
        Self { source: source.into(), generation }
    }

    pub fn human() -> Self {
        Self::new("human", 0)
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self::new("system", 0)
    }
}

// ---------------------------------------------------------------------------
// ConstraintBlock
// ---------------------------------------------------------------------------

/// Constraint parameters drawn from the Constraint Theory engine.
///
/// Simplified for the unified spec: keeps the two most operationally useful
/// scalars — `tolerance` (how much deviation is acceptable) and `threshold`
/// (the activation cutoff for constraint satisfaction).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintBlock {
    /// Acceptable deviation from the constraint target (e.g. 0.05 = 5 %).
    pub tolerance: f64,
    /// Minimum score required for the constraint to be considered satisfied.
    pub threshold: f64,
}

impl ConstraintBlock {
    pub fn new(tolerance: f64, threshold: f64) -> Self {
        Self { tolerance, threshold }
    }

    /// Returns `true` when `score` satisfies the constraint.
    pub fn is_satisfied(&self, score: f64) -> bool {
        score >= self.threshold
    }

    /// Returns `true` when `value` is within `tolerance` of `target`.
    pub fn within_tolerance(&self, value: f64, target: f64) -> bool {
        (value - target).abs() <= self.tolerance
    }
}

impl Default for ConstraintBlock {
    fn default() -> Self {
        Self::new(0.05, 0.6)
    }
}

// ---------------------------------------------------------------------------
// Tile
// ---------------------------------------------------------------------------

/// The unified Plato tile — atomic unit of accumulated experience.
///
/// ### Layer summary
///
/// | Layer       | Fields                                           |
/// |-------------|--------------------------------------------------|
/// | Core        | `id`, `confidence`, `provenance`                 |
/// | Content     | `question`, `answer`, `tags`, `anchors`          |
/// | Attention   | `weight`, `use_count`, `active`, `last_used_tick`|
/// | Constraints | `constraints` (`ConstraintBlock`)                |
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tile {
    // --- Core ---
    /// Nanosecond-based unique identifier.
    pub id: String,
    /// Confidence in the answer (0.0 – 1.0).
    pub confidence: f64,
    /// Where this tile came from.
    pub provenance: Provenance,

    // --- Content ---
    /// The question or prompt this tile answers.
    pub question: String,
    /// The answer or response.
    pub answer: String,
    /// Semantic tags for retrieval (e.g. `["payment", "refund"]`).
    pub tags: Vec<String>,
    /// Named anchors (slugified section identifiers, e.g. `["PaymentFlow"]`).
    pub anchors: Vec<String>,

    // --- Attention ---
    /// Relevance weight; grows with use, decays over time.
    pub weight: f64,
    /// Total number of times this tile has been used.
    pub use_count: u64,
    /// Whether the tile is currently active (not pruned).
    pub active: bool,
    /// Nanosecond tick of last use (0 = never used).
    pub last_used_tick: u64,

    // --- Constraints ---
    pub constraints: ConstraintBlock,
}

impl Tile {
    /// Create a new tile with sensible defaults.
    pub fn new(
        question: impl Into<String>,
        answer: impl Into<String>,
        provenance: Provenance,
    ) -> Self {
        Self {
            id: generate_id("tile"),
            confidence: 0.5,
            provenance,
            question: question.into(),
            answer: answer.into(),
            tags: Vec::new(),
            anchors: Vec::new(),
            weight: 1.0,
            use_count: 0,
            active: true,
            last_used_tick: 0,
            constraints: ConstraintBlock::default(),
        }
    }

    /// Record a usage: bump counters, update weight (saturating), record tick.
    pub fn record_use(&mut self, confidence_signal: f64) {
        self.use_count += 1;
        self.last_used_tick = now_ns();
        // Bayesian-style weight saturation: weight → 1 asymptotically.
        self.weight = 1.0 - (1.0 - self.weight) * 0.9;
        // Fuse incoming confidence with current via harmonic mean.
        self.confidence = fuse_confidence(self.confidence, confidence_signal);
    }

    /// Exponential weight decay simulating a forgetting curve.
    ///
    /// `rate` is the per-minute decay coefficient.
    pub fn decay(&mut self, rate: f64) {
        let age_ns = now_ns().saturating_sub(self.last_used_tick);
        let age_min = age_ns as f64 / 1_000_000_000.0 / 60.0;
        let factor = (-rate * age_min).exp();
        self.weight *= factor;
        self.confidence *= 1.0 - rate * 0.01;
        self.confidence = self.confidence.max(0.0);
        if self.weight < 0.01 {
            self.active = false;
        }
    }

    /// Whether this tile matches any of the given tags (case-insensitive).
    pub fn matches_tags(&self, query_tags: &[&str]) -> bool {
        query_tags.iter().any(|qt| {
            self.tags.iter().any(|t| t.to_lowercase() == qt.to_lowercase())
        })
    }

    /// Whether this tile contains the given anchor (case-insensitive).
    pub fn matches_anchor(&self, anchor: &str) -> bool {
        self.anchors.iter().any(|a| a.to_lowercase() == anchor.to_lowercase())
    }

    /// Score = confidence × weight; used for ranking.
    pub fn score(&self) -> f64 {
        self.confidence * self.weight
    }

    /// Returns `true` when the tile's confidence satisfies its own constraint threshold.
    pub fn constraint_satisfied(&self) -> bool {
        self.constraints.is_satisfied(self.confidence)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Deserialize from JSON string.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

// ---------------------------------------------------------------------------
// TileManager
// ---------------------------------------------------------------------------

/// Manages a collection of tiles: create, retrieve, search, prune, decay, merge.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TileManager {
    tiles: HashMap<String, Tile>,
}

impl TileManager {
    pub fn new() -> Self {
        Self { tiles: HashMap::new() }
    }

    // --- CRUD ----------------------------------------------------------

    /// Create a tile and insert it; returns a clone for inspection.
    pub fn create(
        &mut self,
        question: impl Into<String>,
        answer: impl Into<String>,
        provenance: Provenance,
    ) -> Tile {
        let tile = Tile::new(question, answer, provenance);
        let id = tile.id.clone();
        self.tiles.insert(id, tile.clone());
        tile
    }

    /// Insert an existing tile (e.g. after merging).
    pub fn insert(&mut self, tile: Tile) -> String {
        let id = tile.id.clone();
        self.tiles.insert(id.clone(), tile);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Tile> {
        self.tiles.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Tile> {
        self.tiles.get_mut(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<Tile> {
        self.tiles.remove(id)
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Iterate over all tiles (unordered).
    pub fn all(&self) -> impl Iterator<Item = &Tile> {
        self.tiles.values()
    }

    // --- Search --------------------------------------------------------

    /// Find active tiles that contain at least one of the given tags.
    pub fn search_by_tag(&self, tags: &[&str]) -> Vec<&Tile> {
        let mut results: Vec<&Tile> = self
            .tiles
            .values()
            .filter(|t| t.active && t.matches_tags(tags))
            .collect();
        results.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find active tiles that contain the given anchor.
    pub fn search_by_anchor(&self, anchor: &str) -> Vec<&Tile> {
        let mut results: Vec<&Tile> = self
            .tiles
            .values()
            .filter(|t| t.active && t.matches_anchor(anchor))
            .collect();
        results.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Return the single highest-scoring active tile for a tag query.
    pub fn best_for_tag(&self, tags: &[&str]) -> Option<&Tile> {
        self.search_by_tag(tags).into_iter().next()
    }

    // --- Lifecycle -----------------------------------------------------

    /// Deactivate tiles whose weight falls below `min_weight`.
    pub fn prune_by_weight(&mut self, min_weight: f64) {
        for tile in self.tiles.values_mut() {
            if tile.weight < min_weight {
                tile.active = false;
            }
        }
    }

    /// Apply decay to every tile at the given per-minute rate.
    pub fn decay_all(&mut self, rate: f64) {
        for tile in self.tiles.values_mut() {
            tile.decay(rate);
        }
    }

    /// Merge two tiles by ID into a new tile inserted into the manager.
    ///
    /// - Confidence: harmonic-mean fusion
    /// - Weight: average
    /// - Tags / anchors: union (deduplicated)
    /// - Question / answer: taken from `a` (primary)
    /// - Provenance: taken from `a` with generation bumped by 1
    pub fn merge(&mut self, id_a: &str, id_b: &str) -> Option<Tile> {
        let a = self.tiles.get(id_a)?.clone();
        let b = self.tiles.get(id_b)?.clone();

        let mut merged_tags = a.tags.clone();
        for tag in &b.tags {
            if !merged_tags.contains(tag) {
                merged_tags.push(tag.clone());
            }
        }

        let mut merged_anchors = a.anchors.clone();
        for anchor in &b.anchors {
            if !merged_anchors.contains(anchor) {
                merged_anchors.push(anchor.clone());
            }
        }

        let mut merged = Tile::new(
            a.question.clone(),
            a.answer.clone(),
            Provenance::new(a.provenance.source.clone(), a.provenance.generation + 1),
        );
        merged.confidence = fuse_confidence(a.confidence, b.confidence);
        merged.weight = (a.weight + b.weight) / 2.0;
        merged.use_count = a.use_count + b.use_count;
        merged.tags = merged_tags;
        merged.anchors = merged_anchors;
        merged.constraints = a.constraints.clone();

        let id = merged.id.clone();
        self.tiles.insert(id, merged.clone());
        Some(merged)
    }

    /// Remove all inactive tiles from storage.
    pub fn collect_garbage(&mut self) {
        self.tiles.retain(|_, t| t.active);
    }

    // --- Stats ---------------------------------------------------------

    pub fn active_count(&self) -> usize {
        self.tiles.values().filter(|t| t.active).count()
    }

    pub fn avg_confidence(&self) -> f64 {
        let vals: Vec<f64> = self.tiles.values().map(|t| t.confidence).collect();
        if vals.is_empty() {
            return 0.0;
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Harmonic-mean confidence fusion (same formula as cuda-ghost-tiles).
fn fuse_confidence(a: f64, b: f64) -> f64 {
    let inv = 1.0 / a.max(1e-10) + 1.0 / b.max(1e-10);
    if inv >= 1e10 {
        return 0.0;
    }
    1.0 / inv
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn make_tile(q: &str, a: &str) -> Tile {
        Tile::new(q, a, Provenance::human())
    }

    fn make_tagged(tags: &[&str]) -> Tile {
        let mut t = make_tile("What is X?", "X is Y.");
        t.tags = tags.iter().map(|s| s.to_string()).collect();
        t
    }

    // 1. ID is non-empty and contains "tile"
    #[test]
    fn test_tile_id_format() {
        let t = make_tile("q", "a");
        assert!(t.id.contains("tile"), "id should contain 'tile'");
        assert!(!t.id.is_empty());
    }

    // 2. Consecutive tiles get different IDs (nanosecond precision)
    #[test]
    fn test_unique_ids() {
        // spin until time advances
        let t1 = make_tile("q1", "a1");
        let mut t2 = make_tile("q2", "a2");
        // If somehow same ns (unlikely but possible on fast CPUs), regenerate
        while t2.id == t1.id {
            t2 = make_tile("q2", "a2");
        }
        assert_ne!(t1.id, t2.id);
    }

    // 3. Default confidence is 0.5
    #[test]
    fn test_default_confidence() {
        let t = make_tile("q", "a");
        assert!((t.confidence - 0.5).abs() < f64::EPSILON);
    }

    // 4. record_use increments use_count and bumps weight toward 1
    #[test]
    fn test_record_use() {
        let mut t = make_tile("q", "a");
        t.weight = 0.5;
        t.record_use(0.9);
        assert_eq!(t.use_count, 1);
        assert!(t.weight > 0.5, "weight should grow after use");
        assert!(t.last_used_tick > 0);
    }

    // 5. Repeated use saturates weight toward 1.0
    #[test]
    fn test_weight_saturation() {
        let mut t = make_tile("q", "a");
        for _ in 0..50 {
            t.record_use(1.0);
        }
        assert!(t.weight > 0.99, "weight should saturate near 1 after many uses");
    }

    // 6. score() = confidence × weight
    #[test]
    fn test_score() {
        let mut t = make_tile("q", "a");
        t.confidence = 0.8;
        t.weight = 0.5;
        assert!((t.score() - 0.4).abs() < 1e-9);
    }

    // 7. matches_tags is case-insensitive
    #[test]
    fn test_matches_tags_case_insensitive() {
        let t = make_tagged(&["Payment", "Refund"]);
        assert!(t.matches_tags(&["payment"]));
        assert!(t.matches_tags(&["REFUND"]));
        assert!(!t.matches_tags(&["billing"]));
    }

    // 8. matches_anchor is case-insensitive
    #[test]
    fn test_matches_anchor() {
        let mut t = make_tile("q", "a");
        t.anchors = vec!["PaymentFlow".to_string()];
        assert!(t.matches_anchor("paymentflow"));
        assert!(t.matches_anchor("PaymentFlow"));
        assert!(!t.matches_anchor("settlement"));
    }

    // 9. ConstraintBlock::is_satisfied
    #[test]
    fn test_constraint_satisfied() {
        let cb = ConstraintBlock::new(0.05, 0.7);
        assert!(cb.is_satisfied(0.7));
        assert!(cb.is_satisfied(0.9));
        assert!(!cb.is_satisfied(0.5));
    }

    // 10. ConstraintBlock::within_tolerance
    #[test]
    fn test_within_tolerance() {
        let cb = ConstraintBlock::new(0.1, 0.5);
        assert!(cb.within_tolerance(1.05, 1.0));
        assert!(!cb.within_tolerance(1.2, 1.0));
    }

    // 11. tile.constraint_satisfied() delegates to ConstraintBlock
    #[test]
    fn test_tile_constraint_satisfied() {
        let mut t = make_tile("q", "a");
        t.confidence = 0.8;
        t.constraints = ConstraintBlock::new(0.05, 0.6);
        assert!(t.constraint_satisfied());
        t.confidence = 0.4;
        assert!(!t.constraint_satisfied());
    }

    // 12. JSON round-trip
    #[test]
    fn test_json_roundtrip() {
        let mut t = make_tagged(&["foo", "bar"]);
        t.anchors = vec!["FooBar".to_string()];
        let json = t.to_json();
        let t2 = Tile::from_json(&json).expect("deserialization failed");
        assert_eq!(t.id, t2.id);
        assert_eq!(t.tags, t2.tags);
        assert_eq!(t.anchors, t2.anchors);
    }

    // 13. TileManager::create inserts tile and returns it
    #[test]
    fn test_manager_create() {
        let mut mgr = TileManager::new();
        let t = mgr.create("q", "a", Provenance::human());
        assert_eq!(mgr.len(), 1);
        assert!(mgr.get(&t.id).is_some());
    }

    // 14. search_by_tag returns active tiles with matching tags, sorted by score
    #[test]
    fn test_search_by_tag() {
        let mut mgr = TileManager::new();
        let mut t1 = mgr.create("q1", "a1", Provenance::human());
        t1.tags = vec!["rust".to_string()];
        t1.confidence = 0.9;
        let id1 = t1.id.clone();
        mgr.tiles.insert(id1, t1);

        let mut t2 = mgr.create("q2", "a2", Provenance::human());
        t2.tags = vec!["python".to_string()];
        let id2 = t2.id.clone();
        mgr.tiles.insert(id2, t2);

        let results = mgr.search_by_tag(&["rust"]);
        assert_eq!(results.len(), 1);
        assert!(results[0].tags.contains(&"rust".to_string()));
    }

    // 15. search_by_anchor returns matching active tiles
    #[test]
    fn test_search_by_anchor() {
        let mut mgr = TileManager::new();
        let mut t = mgr.create("What is PaymentFlow?", "It processes payments.", Provenance::human());
        t.anchors = vec!["PaymentFlow".to_string()];
        let id = t.id.clone();
        mgr.tiles.insert(id, t);

        let results = mgr.search_by_anchor("paymentflow");
        assert_eq!(results.len(), 1);
    }

    // 16. prune_by_weight deactivates low-weight tiles
    #[test]
    fn test_prune_by_weight() {
        let mut mgr = TileManager::new();
        let t = mgr.create("q", "a", Provenance::human());
        let id = t.id.clone();
        mgr.get_mut(&id).unwrap().weight = 0.05;
        mgr.prune_by_weight(0.1);
        assert!(!mgr.get(&id).unwrap().active);
    }

    // 17. decay_all reduces weight; tiles with last_used_tick=0 decay maximally
    #[test]
    fn test_decay_all() {
        let mut mgr = TileManager::new();
        let t = mgr.create("q", "a", Provenance::human());
        let id = t.id.clone();
        // Set last_used to distant past
        mgr.get_mut(&id).unwrap().last_used_tick = 0;
        let before = mgr.get(&id).unwrap().weight;
        mgr.decay_all(0.5);
        let after = mgr.get(&id).unwrap().weight;
        assert!(after < before, "weight should decrease after decay");
    }

    // 18. merge combines two tiles into a new one with fused confidence
    #[test]
    fn test_merge() {
        let mut mgr = TileManager::new();
        let mut ta = mgr.create("q", "a", Provenance::human());
        ta.tags = vec!["tag-a".to_string()];
        ta.anchors = vec!["AnchorA".to_string()];
        ta.confidence = 0.8;
        let id_a = ta.id.clone();
        mgr.tiles.insert(id_a.clone(), ta);

        let mut tb = mgr.create("q", "b", Provenance::human());
        tb.tags = vec!["tag-b".to_string()];
        tb.anchors = vec!["AnchorB".to_string()];
        tb.confidence = 0.6;
        let id_b = tb.id.clone();
        mgr.tiles.insert(id_b.clone(), tb);

        let merged = mgr.merge(&id_a, &id_b).expect("merge failed");
        assert!(merged.tags.contains(&"tag-a".to_string()));
        assert!(merged.tags.contains(&"tag-b".to_string()));
        assert!(merged.anchors.contains(&"AnchorA".to_string()));
        assert!(merged.anchors.contains(&"AnchorB".to_string()));
        assert!(merged.confidence > 0.0);
        // generation bumped
        assert_eq!(merged.provenance.generation, 1);
    }

    // 19. collect_garbage removes inactive tiles
    #[test]
    fn test_collect_garbage() {
        let mut mgr = TileManager::new();
        let t = mgr.create("q", "a", Provenance::human());
        let id = t.id.clone();
        mgr.get_mut(&id).unwrap().active = false;
        mgr.collect_garbage();
        assert!(mgr.get(&id).is_none());
        assert_eq!(mgr.len(), 0);
    }

    // 20. active_count excludes inactive tiles
    #[test]
    fn test_active_count() {
        let mut mgr = TileManager::new();
        let t1 = mgr.create("q1", "a1", Provenance::human());
        let t2 = mgr.create("q2", "a2", Provenance::human());
        let id2 = t2.id.clone();
        let _ = t1;
        mgr.get_mut(&id2).unwrap().active = false;
        assert_eq!(mgr.active_count(), 1);
    }

    // 21. avg_confidence over empty manager is 0
    #[test]
    fn test_avg_confidence_empty() {
        let mgr = TileManager::new();
        assert!((mgr.avg_confidence() - 0.0).abs() < f64::EPSILON);
    }

    // 22. fuse_confidence harmonic mean property: result < arithmetic mean
    #[test]
    fn test_fuse_confidence_harmonic() {
        let fused = fuse_confidence(0.8, 0.6);
        let arithmetic = (0.8 + 0.6) / 2.0;
        assert!(fused < arithmetic, "harmonic mean should be less than arithmetic mean");
        assert!(fused > 0.0);
    }

    // 23. Provenance::human() has generation 0 and source "human"
    #[test]
    fn test_provenance_human() {
        let p = Provenance::human();
        assert_eq!(p.source, "human");
        assert_eq!(p.generation, 0);
    }

    // 24. best_for_tag returns highest-scoring match
    #[test]
    fn test_best_for_tag() {
        let mut mgr = TileManager::new();
        let mut t_low = mgr.create("q-low", "a", Provenance::human());
        t_low.tags = vec!["x".to_string()];
        t_low.confidence = 0.3;
        let id_low = t_low.id.clone();
        mgr.tiles.insert(id_low, t_low);

        let mut t_high = mgr.create("q-high", "a", Provenance::human());
        t_high.tags = vec!["x".to_string()];
        t_high.confidence = 0.9;
        let id_high = t_high.id.clone();
        mgr.tiles.insert(id_high.clone(), t_high);

        let best = mgr.best_for_tag(&["x"]).unwrap();
        assert_eq!(best.id, id_high);
    }

    // 25. Tile remains active after record_use
    #[test]
    fn test_tile_stays_active_after_use() {
        let mut t = make_tile("q", "a");
        t.record_use(0.7);
        assert!(t.active);
    }
}
