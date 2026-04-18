# plato-tile-spec

Unified Plato tile specification — the atomic unit of accumulated experience.

## Overview

A **tile** fuses four design lineages:

| Reference | Contribution |
|-----------|-------------|
| `reference-constraint-theory-tile.rs` | `ConstraintBlock` (tolerance, threshold) |
| `reference-ghost-tiles.rs` | Attention layer: weight, decay, prune, merge |
| `reference-plato-python-tiles.py` | Content layer: question/answer/tags, provenance |
| `reference-plato-tiling.rs` | Anchors as first-class semantic identifiers |

## Tile Structure

```
Tile
├── Core
│   ├── id            — nanosecond-based string ("tile-<ns>")
│   ├── confidence    — f64 in [0, 1]
│   └── provenance    — source: String, generation: u32
│
├── Content
│   ├── question      — String
│   ├── answer        — String
│   ├── tags          — Vec<String>  (keyword retrieval)
│   └── anchors       — Vec<String>  (slug-based retrieval)
│
├── Attention
│   ├── weight        — f64  (grows with use, decays over time)
│   ├── use_count     — u64
│   ├── active        — bool
│   └── last_used_tick — u64 (nanoseconds)
│
└── Constraints
    └── ConstraintBlock
        ├── tolerance — f64  (acceptable deviation)
        └── threshold — f64  (minimum satisfying score)
```

## Quick Start

### Create a tile

```rust
use plato_tile_spec::{Tile, Provenance};

let mut tile = Tile::new(
    "What is the refund policy?",
    "Refunds are processed within 5 business days.",
    Provenance::human(),
);
tile.tags = vec!["refund".to_string(), "policy".to_string()];
tile.anchors = vec!["RefundPolicy".to_string()];
```

### Use TileManager

```rust
use plato_tile_spec::{TileManager, Provenance};

let mut mgr = TileManager::new();

// Create tiles
let t1 = mgr.create("What is X?", "X is Y.", Provenance::human());
let t2 = mgr.create("What is Z?", "Z is W.", Provenance::new("npc", 1));

// Add tags directly
mgr.get_mut(&t1.id).unwrap().tags = vec!["x".to_string()];
mgr.get_mut(&t2.id).unwrap().tags = vec!["z".to_string()];

// Search
let results = mgr.search_by_tag(&["x"]);
let by_anchor = mgr.search_by_anchor("RefundPolicy");
let best = mgr.best_for_tag(&["x"]);

// Lifecycle
mgr.decay_all(0.1);            // apply forgetting curve (rate = per-minute)
mgr.prune_by_weight(0.05);     // deactivate low-weight tiles
mgr.collect_garbage();         // drop inactive tiles from memory

// Merge two tiles
let merged = mgr.merge(&t1.id, &t2.id);
```

### Record usage

```rust
tile.record_use(0.85);   // confidence signal from retrieval feedback
println!("weight: {:.3}", tile.weight);   // grows toward 1.0
println!("uses:   {}", tile.use_count);
```

### JSON serialization

```rust
let json = tile.to_json();
let restored = Tile::from_json(&json).unwrap();
assert_eq!(tile.id, restored.id);
```

### Constraints

```rust
use plato_tile_spec::ConstraintBlock;

tile.constraints = ConstraintBlock::new(0.05, 0.7); // tolerance=5%, threshold=0.7
println!("satisfied: {}", tile.constraint_satisfied()); // confidence >= 0.7?
println!("in range:  {}", tile.constraints.within_tolerance(1.03, 1.0)); // |1.03-1.0| <= 0.05
```

## Running Tests

```sh
cargo test
```

## Design Notes

- **IDs** use `SystemTime` nanoseconds to avoid external dependencies. Collisions
  are astronomically unlikely in single-threaded usage; add a sequence counter if
  needed for concurrent tile creation.

- **Confidence fusion** uses the harmonic mean (`1/(1/a + 1/b)`), identical to
  the formula in `cuda-ghost-tiles`. This penalises low-confidence signals more
  aggressively than the arithmetic mean.

- **Decay** is an exponential forgetting curve parameterised by a per-minute rate.
  A tile last used at tick 0 (never used) decays maximally.

- **Merge** takes question/answer from the primary tile and bumps `generation` by 1,
  making provenance traceable across synthetic tile creation rounds.
