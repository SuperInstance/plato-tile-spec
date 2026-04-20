"""Canonical PLATO tile specification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TileDomain(Enum):
    """The fourteen canonical PLATO tile domains."""

    CONTRAINT_THEORY = "CONTRAINT_THEORY"
    TILES = "TILES"
    GOVERNANCE = "GOVERNANCE"
    FORGE = "FORGE"
    FLEET = "FLEET"
    RESEARCH = "RESEARCH"
    BOUNDARY = "BOUNDARY"
    EDGE = "EDGE"
    MUD = "MUD"
    NEGATIVE_SPACE = "NEGATIVE_SPACE"
    META_COGNITION = "META_COGNITION"
    CROSS_POLLINATION = "CROSS_POLLINATION"
    SENTIMENT = "SENTIMENT"
    GENERAL = "GENERAL"


@dataclass
class TileSpec:
    """A single PLATO tile specification.

    Attributes:
        id: Unique tile identifier (non-empty string).
        content: Human-readable tile content (10–100_000 characters).
        domain: Canonical domain from :class:`TileDomain`.
        confidence: Model confidence in the tile, bounded ``[0.0, 1.0]``.
        priority: Urgency level — one of ``P0``, ``P1``, or ``P2``.
        tags: Optional categorical labels.
        provenance: Source or origin description.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
        usage_count: Number of times the tile has been invoked.
        success_rate: Fraction of successful invocations ``[0.0, 1.0]``.
        dependencies: IDs of tiles this tile depends on.
        version: Monotonically increasing revision number.
    """

    id: str
    content: str
    domain: TileDomain
    confidence: float
    priority: str
    tags: list[str] = field(default_factory=list)
    provenance: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    version: int = 1


class TileSpecValidator:
    """Validation and (de)serialization utilities for :class:`TileSpec`."""

    _VALID_PRIORITIES = {"P0", "P1", "P2"}

    @classmethod
    def validate(cls, spec: TileSpec) -> tuple[bool, list[str]]:
        """Validate a ``TileSpec`` against canonical rules.

        Args:
            spec: The tile specification to validate.

        Returns:
            A 2-tuple ``(is_valid, errors)`` where *errors* is a list of human-
            readable violation messages (empty when *is_valid* is ``True``).
        """
        errors: list[str] = []

        if not isinstance(spec.id, str) or not spec.id.strip():
            errors.append("id must be a non-empty string.")

        if not isinstance(spec.content, str):
            errors.append("content must be a string.")
        else:
            content_len = len(spec.content)
            if content_len < 10:
                errors.append(f"content must be at least 10 characters (got {content_len}).")
            if content_len > 100_000:
                errors.append(f"content must not exceed 100,000 characters (got {content_len}).")

        if not isinstance(spec.domain, TileDomain):
            errors.append(f"domain must be a TileDomain enum member (got {type(spec.domain).__name__}).")

        if not isinstance(spec.confidence, (int, float)):
            errors.append("confidence must be a number.")
        else:
            if not 0.0 <= spec.confidence <= 1.0:
                errors.append(f"confidence must be in [0.0, 1.0] (got {spec.confidence}).")

        if spec.priority not in cls._VALID_PRIORITIES:
            errors.append(f"priority must be one of {sorted(cls._VALID_PRIORITIES)} (got {spec.priority!r}).")

        return (not errors, errors)

    @classmethod
    def to_dict(cls, spec: TileSpec) -> dict[str, Any]:
        """Serialize a ``TileSpec`` to a plain dictionary.

        The *domain* is rendered as its value string for portability.
        """
        data = asdict(spec)
        data["domain"] = spec.domain.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TileSpec:
        """Deserialize a ``TileSpec`` from a dictionary.

        Args:
            data: Mapping produced by :meth:`to_dict` or equivalent.

        Returns:
            A fully instantiated ``TileSpec``.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If the *domain* or *priority* value is unrecognised.
        """
        data = dict(data)  # shallow copy so we can mutate safely

        domain_raw = data.pop("domain")
        try:
            domain = TileDomain(domain_raw)
        except ValueError as exc:
            raise ValueError(f"Unrecognised TileDomain: {domain_raw!r}") from exc

        # Coerce optional list fields to list[str] if they arrive as something else
        for list_key in ("tags", "dependencies"):
            if list_key in data and not isinstance(data[list_key], list):
                data[list_key] = list(data[list_key])

        return TileSpec(
            id=data.pop("id"),
            content=data.pop("content"),
            domain=domain,
            confidence=data.pop("confidence"),
            priority=data.pop("priority"),
            tags=data.pop("tags", []),
            provenance=data.pop("provenance", ""),
            created_at=data.pop("created_at", 0.0),
            updated_at=data.pop("updated_at", 0.0),
            usage_count=data.pop("usage_count", 0),
            success_rate=data.pop("success_rate", 0.0),
            dependencies=data.pop("dependencies", []),
            version=data.pop("version", 1),
        )

    @classmethod
    def to_json(cls, spec: TileSpec) -> str:
        """Serialize a ``TileSpec`` to a JSON string."""
        return json.dumps(cls.to_dict(spec), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> TileSpec:
        """Deserialize a ``TileSpec`` from a JSON string."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON payload must decode to a JSON object.")
        return cls.from_dict(data)
