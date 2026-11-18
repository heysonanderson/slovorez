"""Typed view of a model's config.json (schema 2).

Every problem with the file is reported as ConfigError whose message starts
with the JSON path of the bad field, e.g. ``config.io.inputs[0].dtype: missing``.

The config knows nothing about the model directory: ``ResourceSpec.file`` is a
plain file name, and the caller joins it with the directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from slovorez.hub import CONFIG_NAME, resolve_model_dir
from slovorez.io.loaders import load_json
from slovorez.paths import check_part

SUPPORTED_SCHEMA_VERSION = 2
KNOWN_SCOPES = ("lemmas", "wordforms")


class ConfigError(ValueError):
    """config.json has a missing field, a wrong type or an unsupported value."""

_JSON_TYPE_NAMES = {str: "string", int: "integer", float: "number", list: "array", dict: "object"}


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def _check(value: Any, expected: type, path: str) -> Any:
    """Return value if it has the JSON type `expected`, else raise ConfigError.

    An integer passes as float and is converted, since JSON writes 1.0 as 1.
    """
    if isinstance(value, bool):
        ok = False
    elif expected is float:
        ok = isinstance(value, (int, float))
    else:
        ok = isinstance(value, expected)
    if not ok:
        raise ConfigError(
            f"{path}: expected {_JSON_TYPE_NAMES[expected]}, got {_json_type_name(value)}"
        )
    return float(value) if expected is float else value


def _require(d: dict, key: str, path: str, expected: type) -> Any:
    if key not in d:
        raise ConfigError(f"{path}.{key}: missing")
    return _check(d[key], expected, f"{path}.{key}")


def _optional(d: dict, key: str, path: str, expected: type) -> Any:
    value = d.get(key)
    return None if value is None else _check(value, expected, f"{path}.{key}")


def _list_of(d: dict, key: str, path: str, item_type: type) -> list:
    items = _require(d, key, path, list)
    return [_check(item, item_type, f"{path}.{key}[{i}]") for i, item in enumerate(items)]


def _dict_of(d: dict, key: str, path: str, value_type: type) -> dict:
    items = _require(d, key, path, dict)
    return {k: _check(v, value_type, f"{path}.{key}[{k!r}]") for k, v in items.items()}


def _section(cls, d: dict, key: str, path: str):
    return cls.from_dict(_require(d, key, path, dict), f"{path}.{key}")


def _optional_section(cls, d: dict, key: str, path: str):
    value = _optional(d, key, path, dict)
    return None if value is None else cls.from_dict(value, f"{path}.{key}")


def _section_list(cls, d: dict, key: str, path: str) -> list:
    items = _require(d, key, path, list)
    result = []
    for i, item in enumerate(items):
        item_path = f"{path}.{key}[{i}]"
        result.append(cls.from_dict(_check(item, dict, item_path), item_path))
    return result




@dataclass(frozen=True)
class ResourceSpec:
    file: str
    sha256: str | None

    @classmethod
    def from_dict(cls, d: dict, path: str) -> ResourceSpec:
        file = _require(d, "file", path, str)
        try:
            check_part(file, "file")
        except ValueError as e:
            raise ConfigError(f"{path}.file: {e}") from e
        return cls(file=file, sha256=_optional(d, "sha256", path, str))


@dataclass(frozen=True)
class ResourcesSpec:
    weights: ResourceSpec
    base_dict: ResourceSpec | None
    passport: ResourceSpec | None

    @classmethod
    def from_dict(cls, d: dict, path: str) -> ResourcesSpec:
        return cls(
            weights=_section(ResourceSpec, d, "weights", path),
            base_dict=_optional_section(ResourceSpec, d, "base_dict", path),
            passport=_optional_section(ResourceSpec, d, "passport", path),
        )


@dataclass(frozen=True)
class InputSpec:
    name: str
    dtype: str
    shape: list[str]

    @classmethod
    def from_dict(cls, d: dict, path: str) -> InputSpec:
        return cls(
            name=_require(d, "name", path, str),
            dtype=_require(d, "dtype", path, str),
            shape=_list_of(d, "shape", path, str),
        )


@dataclass(frozen=True)
class OutputSpec:
    name: str
    kind: str
    dtype: str
    shape: list[str]

    @classmethod
    def from_dict(cls, d: dict, path: str) -> OutputSpec:
        return cls(
            name=_require(d, "name", path, str),
            kind=_require(d, "kind", path, str),
            dtype=_require(d, "dtype", path, str),
            shape=_list_of(d, "shape", path, str),
        )


@dataclass(frozen=True)
class IOSpec:
    minlen: int
    maxlen: int
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    padding_side: str

    @classmethod
    def from_dict(cls, d: dict, path: str) -> IOSpec:
        return cls(
            minlen=_require(d, "minlen", path, int),
            maxlen=_require(d, "maxlen", path, int),
            inputs=_section_list(InputSpec, d, "inputs", path),
            outputs=_section_list(OutputSpec, d, "outputs", path),
            padding_side=_require(d, "padding_side", path, str),
        )


@dataclass(frozen=True)
class MappingSpec:
    label2id: dict[str, int]
    tokenizer_vocab: dict[str, int]
    vocab_fingerprint: str
    char_pad_id: int
    label_pad_id: int
    morpheme_types: dict[str, int]

    @classmethod
    def from_dict(cls, d: dict, path: str) -> MappingSpec:
        return cls(
            label2id=_dict_of(d, "label2id", path, int),
            tokenizer_vocab=_dict_of(d, "tokenizer_vocab", path, int),
            vocab_fingerprint=_require(d, "vocab_fingerprint", path, str),
            char_pad_id=_require(d, "char_pad_id", path, int),
            label_pad_id=_require(d, "label_pad_id", path, int),
            morpheme_types=_dict_of(d, "morpheme_types", path, int),
        )


@dataclass(frozen=True)
class ConstraintsSpec:
    forbidden_starts: list[str]
    forbidden_ends: list[str]

    @classmethod
    def from_dict(cls, d: dict, path: str) -> ConstraintsSpec:
        return cls(
            forbidden_starts=_list_of(d, "forbidden_starts", path, str),
            forbidden_ends=_list_of(d, "forbidden_ends", path, str),
        )


@dataclass(frozen=True)
class TransitionsSpec:
    """Learned CRF weights (C x C, C, C), added to the Viterbi mask."""

    matrix: list[list[float]]
    start: list[float]
    end: list[float]

    @classmethod
    def from_dict(cls, d: dict, path: str) -> TransitionsSpec:
        rows = _require(d, "matrix", path, list)
        matrix = []
        for i, row in enumerate(rows):
            row_path = f"{path}.matrix[{i}]"
            row = _check(row, list, row_path)
            matrix.append([_check(x, float, f"{row_path}[{j}]") for j, x in enumerate(row)])
        return cls(
            matrix=matrix,
            start=_list_of(d, "start", path, float),
            end=_list_of(d, "end", path, float),
        )


@dataclass(frozen=True)
class DecoderSpec:
    method: str
    constraints: ConstraintsSpec
    transitions: TransitionsSpec | None

    @classmethod
    def from_dict(cls, d: dict, path: str) -> DecoderSpec:
        return cls(
            method=_require(d, "method", path, str),
            constraints=_section(ConstraintsSpec, d, "constraints", path),
            transitions=_optional_section(TransitionsSpec, d, "transitions", path),
        )


@dataclass(frozen=True)
class CalibrationSpec:
    method: str
    a: float

    @classmethod
    def from_dict(cls, d: dict, path: str) -> CalibrationSpec:
        return cls(
            method=_require(d, "method", path, str),
            a=_require(d, "a", path, float),
        )




@dataclass(frozen=True)
class SlovorezConfig:
    schema_version: int
    min_slovorez: str
    name: str
    version: str
    scope: str
    domain: str
    resources: ResourcesSpec
    normalization: list[str]
    io: IOSpec
    mapping: MappingSpec
    decoder: DecoderSpec
    calibration: CalibrationSpec | None

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, Path]) -> SlovorezConfig:
        """Load config.json of a bundled model name or a model directory."""
        return cls.from_file(resolve_model_dir(name_or_path) / CONFIG_NAME)

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> SlovorezConfig:
        """Load a config.json; errors are prefixed with the file path."""
        try:
            return cls.from_dict(load_json(config_path), "config")
        except json.JSONDecodeError as e:
            raise ConfigError(f"{config_path}: not valid JSON: {e}") from e
        except ConfigError as e:
            raise ConfigError(f"{config_path}: {e}") from e

    @classmethod
    def from_dict(cls, data: dict, path: str) -> SlovorezConfig:
        data = _check(data, dict, path)

        schema_version = _require(data, "schema_version", path, int)
        if not (schema_version == SUPPORTED_SCHEMA_VERSION):
            raise ConfigError(
                f"{path}.schema_version: unsupported value {schema_version!r}, "
                f"expected {SUPPORTED_SCHEMA_VERSION}"
            )

        scope = _require(data, "scope", path, str)
        if scope not in KNOWN_SCOPES:
            raise ConfigError(
                f"{path}.scope: unknown value {scope!r}, expected one of {KNOWN_SCOPES}"
            )

        config = cls(
            schema_version=schema_version,
            min_slovorez=_require(data, "min_slovorez", path, str),
            name=_require(data, "name", path, str),
            version=_require(data, "version", path, str),
            scope=scope,
            domain=_require(data, "domain", path, str),
            resources=_section(ResourcesSpec, data, "resources", path),
            normalization=_list_of(data, "normalization", path, str),
            io=_section(IOSpec, data, "io", path),
            mapping=_section(MappingSpec, data, "mapping", path),
            decoder=_section(DecoderSpec, data, "decoder", path),
            calibration=_optional_section(CalibrationSpec, data, "calibration", path),
        )
        _check_transitions(config, path)
        return config


def _check_transitions(config: SlovorezConfig, path: str) -> None:
    """Cross-field checks: transitions match label count and output kind."""
    transitions = config.decoder.transitions
    if transitions is None:
        return
    t_path = f"{path}.decoder.transitions"

    kind = config.io.outputs[0].kind if config.io.outputs else None
    if kind != "emissions":
        raise ConfigError(
            f"{t_path}: only valid when io.outputs[0].kind is 'emissions', got {kind!r}"
        )

    n = len(config.mapping.label2id)
    if len(transitions.matrix) != n or any(len(row) != n for row in transitions.matrix):
        raise ConfigError(f"{t_path}.matrix: expected {n}x{n} (labels in mapping.label2id)")
    for key in ("start", "end"):
        if len(getattr(transitions, key)) != n:
            raise ConfigError(f"{t_path}.{key}: expected length {n} (labels in mapping.label2id)")
