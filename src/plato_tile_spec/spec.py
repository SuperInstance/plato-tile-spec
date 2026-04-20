"""Tile specification — schema definition, validation rules, type registry, and compatibility checking."""
import time
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
from collections import defaultdict

class FieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    JSON = "json"
    LIST = "list"
    ENUM = "enum"

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class FieldSpec:
    name: str
    field_type: FieldType
    required: bool = False
    default: Any = None
    min_value: float = 0.0
    max_value: float = 0.0
    pattern: str = ""
    enum_values: list[str] = field(default_factory=list)
    description: str = ""

@dataclass
class ValidationError:
    field: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    value: Any = None

@dataclass
class TileSpec:
    name: str
    version: str = "1.0.0"
    description: str = ""
    domain: str = ""
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    custom_validators: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    deprecated: bool = False
    deprecation_message: str = ""
    extends: str = ""  # parent spec name

class SpecRegistry:
    def __init__(self):
        self._specs: dict[str, TileSpec] = {}
        self._validators: dict[str, Callable] = {}
        self._validation_log: list[dict] = []

    def define(self, name: str, version: str = "1.0.0", description: str = "",
               domain: str = "", extends: str = "") -> TileSpec:
        spec = TileSpec(name=name, version=version, description=description,
                       domain=domain, extends=extends)
        self._specs[name] = spec
        return spec

    def add_field(self, spec_name: str, field_name: str, field_type: str = "string",
                  required: bool = False, default: Any = None, min_value: float = 0.0,
                  max_value: float = 0.0, pattern: str = "", enum_values: list[str] = None,
                  description: str = "") -> FieldSpec:
        spec = self._specs.get(spec_name)
        if not spec:
            raise ValueError(f"Spec '{spec_name}' not found")
        fs = FieldSpec(name=field_name, field_type=FieldType(field_type),
                      required=required, default=default, min_value=min_value,
                      max_value=max_value, pattern=pattern,
                      enum_values=enum_values or [], description=description)
        spec.fields[field_name] = fs
        if required:
            spec.required_fields.append(field_name)
        return fs

    def register_validator(self, name: str, fn: Callable):
        self._validators[name] = fn

    def validate(self, spec_name: str, tile: dict) -> list[ValidationError]:
        spec = self._specs.get(spec_name)
        if not spec:
            return [ValidationError("spec", f"Spec '{spec_name}' not found")]
        if spec.deprecated:
            return [ValidationError("spec", f"Spec '{spec_name}' is deprecated: {spec.deprecation_message}",
                                   ValidationSeverity.WARNING)]
        errors = []
        # Check required fields
        for req in spec.required_fields:
            if req not in tile or tile[req] is None or tile[req] == "":
                errors.append(ValidationError(req, f"Required field '{req}' is missing"))

        # Check field types and constraints
        for field_name, field_spec in spec.fields.items():
            value = tile.get(field_name)
            if value is None:
                continue
            # Type check
            if field_spec.field_type == FieldType.STRING:
                if not isinstance(value, str):
                    errors.append(ValidationError(field_name, f"Expected string, got {type(value).__name__}"))
                elif field_spec.pattern and not re.match(field_spec.pattern, value):
                    errors.append(ValidationError(field_name,
                        f"Value '{value}' does not match pattern '{field_spec.pattern}'"))
                elif field_spec.max_value > 0 and len(value) > field_spec.max_value:
                    errors.append(ValidationError(field_name,
                        f"String length {len(value)} exceeds max {field_spec.max_value}",
                        ValidationSeverity.WARNING))
            elif field_spec.field_type == FieldType.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(ValidationError(field_name, f"Expected integer, got {type(value).__name__}"))
                elif field_spec.min_value and value < field_spec.min_value:
                    errors.append(ValidationError(field_name,
                        f"Value {value} below minimum {field_spec.min_value}"))
            elif field_spec.field_type == FieldType.FLOAT:
                try:
                    fval = float(value)
                    if field_spec.min_value and fval < field_spec.min_value:
                        errors.append(ValidationError(field_name,
                            f"Value {fval} below minimum {field_spec.min_value}"))
                except (ValueError, TypeError):
                    errors.append(ValidationError(field_name, f"Expected float"))
            elif field_spec.field_type == FieldType.ENUM:
                if value not in field_spec.enum_values:
                    errors.append(ValidationError(field_name,
                        f"Value '{value}' not in allowed values: {field_spec.enum_values}"))
            elif field_spec.field_type == FieldType.LIST:
                if not isinstance(value, list):
                    errors.append(ValidationError(field_name, f"Expected list"))
                elif field_spec.max_value and len(value) > field_spec.max_value:
                    errors.append(ValidationError(field_name,
                        f"List length {len(value)} exceeds max {field_spec.max_value}",
                        ValidationSeverity.WARNING))

        # Run custom validators
        for vname in spec.custom_validators:
            vfn = self._validators.get(vname)
            if vfn:
                try:
                    result = vfn(tile)
                    if isinstance(result, list):
                        errors.extend(result)
                    elif isinstance(result, str):
                        errors.append(ValidationError("custom", result))
                except Exception as e:
                    errors.append(ValidationError(vname, f"Validator error: {e}",
                                               ValidationSeverity.WARNING))

        self._validation_log.append({"spec": spec_name, "errors": len(errors),
                                      "timestamp": time.time()})
        return errors

    def check_compatibility(self, spec_a: str, spec_b: str) -> dict:
        """Check backward compatibility between two spec versions."""
        sa = self._specs.get(spec_a)
        sb = self._specs.get(spec_b)
        if not sa or not sb:
            return {"compatible": False, "error": "Spec not found"}
        breaking = []
        additive = []
        # Check removed required fields
        for req in sa.required_fields:
            if req not in sb.fields:
                breaking.append(f"Required field '{req}' removed")
        # Check type changes
        for fname, fs in sa.fields.items():
            if fname in sb.fields:
                if sb.fields[fname].field_type != fs.field_type:
                    breaking.append(f"Field '{fname}' type changed: {fs.field_type.value} -> {sb.fields[fname].field_type.value}")
        # Check new required fields
        for req in sb.required_fields:
            if req not in sa.fields:
                breaking.append(f"New required field '{req}' added")
        # Check new optional fields (additive, not breaking)
        for fname in sb.fields:
            if fname not in sa.fields:
                additive.append(f"New field '{fname}' added")
        return {"compatible": len(breaking) == 0,
                "breaking_changes": breaking,
                "additive_changes": additive}

    def get_spec(self, name: str) -> Optional[TileSpec]:
        return self._specs.get(name)

    def list_specs(self, domain: str = "") -> list[TileSpec]:
        specs = list(self._specs.values())
        if domain:
            specs = [s for s in specs if s.domain == domain]
        return specs

    @property
    def stats(self) -> dict:
        domains = {}
        for s in self._specs.values():
            domains[s.domain or "general"] = domains.get(s.domain or "general", 0) + 1
        return {"specs": len(self._specs), "validators": len(self._validators),
                "validations": len(self._validation_log),
                "domains": domains, "deprecated": sum(1 for s in self._specs.values() if s.deprecated)}
