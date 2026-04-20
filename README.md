# plato-tile-spec

Canonical tile format for the PLATO ecosystem. 14 domain types, validation schema, and (de)serialization.

## Install

```bash
pip install plato-tile-spec
```

## Usage

```python
from plato_tile_spec.spec import TileSpec, TileDomain, TileSpecValidator

# Create a tile
tile = TileSpec(
    id="tile-001",
    content="Aggressive play in late position with strong hole cards yields positive EV.",
    domain=TileDomain.FLEET,
    confidence=0.87,
    priority="P1",
    tags=["poker", "strategy", "position"],
    provenance="reinforce-room/poker-sim",
)

# Validate
valid, errors = TileSpecValidator.validate(tile)
assert valid, errors

# Serialize
json_str = TileSpecValidator.to_json(tile)
restored = TileSpecValidator.from_json(json_str)
assert restored.id == tile.id
```

## Domains

| Domain | Purpose |
|--------|---------|
| `CONSTRAINT_THEORY` | Geometric constraint solving |
| `TILES` | Tile metadata and indexing |
| `GOVERNANCE` | Policy and access control |
| `FORGE` | Training and fine-tuning |
| `FLEET` | Multi-agent coordination |
| `RESEARCH` | Research notes and findings |
| `BOUNDARY` | System boundary definitions |
| `EDGE` | Edge computing and deployment |
| `MUD` | Multi-user domain interactions |
| `NEGATIVE_SPACE` | Anti-patterns and failure modes |
| `META_COGNITION` | Self-awareness and reflection |
| `CROSS_POLLINATION` | Cross-domain knowledge transfer |
| `SENTIMENT` | Room and agent sentiment |
| `GENERAL` | Default catch-all |

## API

- `TileSpecValidator.validate(spec)` → `(bool, list[str])`
- `TileSpecValidator.to_dict(spec)` → `dict`
- `TileSpecValidator.from_dict(data)` → `TileSpec`
- `TileSpecValidator.to_json(spec)` → `str`
- `TileSpecValidator.from_json(raw)` → `TileSpec`
