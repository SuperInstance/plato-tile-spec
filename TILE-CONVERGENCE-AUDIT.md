# Tile Convergence Audit — 4 Incompatible Tile Types

**Date:** 2026-04-18 18:05 AKDT
**Author:** Forgemaster ⚒️

## The Problem

Four independent Tile definitions across the fleet. None interoperate.

## Tile Type Comparison

### 1. plato-tile-spec::Tile (CANONICAL — 13 fields)
```rust
pub struct Tile {
    pub id: String, pub confidence: f64, pub provenance: Provenance,
    pub question: String, pub answer: String, pub tags: Vec<String>, pub anchors: Vec<String>,
    pub weight: f64, pub use_count: u64, pub active: bool, pub last_used_tick: u64,
    pub constraints: ConstraintBlock,
}
```
**Used by:** plato-tile-spec, plato-genepool-tile (partial match)

### 2. holodeck-rust::plato_bridge::Tile (9 fields)
```rust
pub struct Tile {
    pub room_id: String, pub agent: String, pub action: String, pub outcome: String,
    pub reward: f64, pub timestamp: u64, pub state_hash: String, pub context: HashMap<String, String>,
}
```
**Used by:** holodeck-rust/src/plato_bridge.rs (232 lines)
**Gap:** No question/answer, no tags/anchors, no weight/attention. Different semantic model (event log vs knowledge).

### 3. fleet-simulator SimPattern (15 fields, Python)
```python
@dataclass
class SimPattern:
    pattern_id: str, pattern_type: str, trigger: str, response: str, outcome: str,
    quality: float, sentiment_before: List[float], sentiment_after: List[float],
    ships_involved: List[str], rooms_involved: List[str], duration_ticks: int,
    auto_resolved: bool, big_model_needed: bool, source_scenario: str, tick_range: Tuple[int, int]
```
**Used by:** fleet-simulator/sim_to_tiles.py → TileConverter → Dict
**Gap:** Python dataclass, no Rust equivalent. Sentiment vectors are fleet-sim-specific.

### 4. plato-kernel::KnowledgeTile (6 fields)
```rust
pub struct KnowledgeTile {
    pub anchor: String, pub header: String, pub body: String,
    pub position: usize, pub word_anchors: Vec<String>,
}
```
**Used by:** plato-kernel/src/tiling/mod.rs (document tiling)
**Gap:** No id, no confidence, no weight. This is a PARSER tile, not a knowledge tile. Different layer.

## Convergence Strategy

### Decision: plato-tile-spec::Tile IS the canonical type
- Most complete (13 fields + Provenance + ConstraintBlock)
- Already used by plato-genepool-tile
- 25 tests pass
- JSON-serializable (serde)

### Mapping Plan

| Field | tile-spec | holodeck | fleet-sim | kernel |
|-------|-----------|----------|-----------|--------|
| id | ✅ String | ❌ → room_id:timestamp | ✅ pattern_id | ❌ → anchor:position |
| confidence | ✅ f64 | ❌ → reward | ✅ quality | ❌ → 1.0 |
| provenance | ✅ enum | ❌ → room_id | ✅ source_scenario | ❌ → Document |
| question | ✅ String | ❌ → action | ✅ trigger | ❌ → header |
| answer | ✅ String | ❌ → outcome | ✅ response | ❌ → body |
| tags | ✅ Vec<String> | ❌ → [room_id] | ❌ → [pattern_type] | ✅ word_anchors |
| anchors | ✅ Vec<String> | ❌ → [] | ❌ → [] | ✅ [anchor] |
| weight | ✅ f64 | ❌ → reward | ❌ → quality | ❌ → 1.0 |
| use_count | ✅ u64 | ❌ → 0 | ❌ → 0 | ❌ → 0 |
| active | ✅ bool | ❌ → true | ❌ → true | ❌ → true |
| constraints | ✅ Block | ❌ → empty | ❌ → empty | ❌ → empty |

### holodeck-rust Migration Path
1. Add `plato-tile-spec` as dependency
2. Create `holodeck_bridge.rs` conversion: `holodeck::Tile → tile_spec::Tile`
3. Keep holodeck::Tile internally (event log semantics), convert at boundary
4. plato_bridge.rs output uses tile_spec::Tile for fleet interchange

### fleet-simulator Migration Path  
1. TileConverter produces JSON matching tile_spec::Tile serde format
2. No Rust dependency — just agree on the JSON wire format
3. Add roundtrip test: Python dict → JSON → Rust Tile (in plato-tile-spec tests)

### plato-kernel Migration Path
1. KnowledgeTile stays as-is (it's a parser artifact, not a knowledge tile)
2. Add conversion: `KnowledgeTile → tile_spec::Tile` in plato-kernel/src/tiling/
3. plato-kernel's Pillar 5 runtime uses tile_spec::Tile for query/response tiles

### plato-genepool-tile Alignment
1. Already close — remove redundant Tile struct, import from plato-tile-spec
2. GeneTile wraps tile_spec::Tile instead of embedding fields
3. gene_to_tile() returns tile_spec::Tile directly
