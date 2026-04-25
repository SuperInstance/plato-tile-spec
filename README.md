# plato-tile-spec

[![PyPI](https://img.shields.io/pypi/v/plato-tile-spec)](https://pypi.org/project/plato-tile-spec/) [![Python](https://img.shields.io/pypi/pyversions/plato-tile-spec)](https://pypi.org/project/plato-tile-spec/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Canonical tile format — TileSpec, TileDomain (14 types), validation schema.

The lingua franca of the PLATO knowledge system. Every piece of knowledge in the fleet is expressed as a TileSpec — a structured question-answer pair with metadata.

## TileSpec Format

```python
{
    "domain": "fleet_orchestration",  # PLATO room
    "question": "How do agents coordinate?",
    "answer": "Via Bottle Protocol...",
    "confidence": 0.85,  # 0-1 quality score
    "source": "oracle1",  # originating agent
    "timestamp": "2026-04-24T20:11:00Z",
    "hash": "sha256:..."  # content hash for dedup
}
```

## 14 Tile Domains

fleet, neural, architecture, security, grammar, arena, mud, theory, research, context, training, evaluation, constraint_theory, general

## Installation

```bash
pip install plato-tile-spec
```

## Part of the Cocapn Fleet

The foundational format that all PLATO components read and write.

## License

MIT