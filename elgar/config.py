"""Inference-only configuration for a separately trained backbone."""
from __future__ import annotations
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')


def boolean(name, value):
    if type(value) is not bool:
        raise ValueError(f'{name} must be boolean.')


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = True
    weight: float = 0.5
    neighbors: int = 128
    max_len: int = 20
    horizon: int = 5

    def __post_init__(self):
        boolean('retrieval.enabled', self.enabled)
        nonnegative('retrieval.weight', self.weight)
        for key in ('neighbors', 'max_len', 'horizon'):
            positive_int(f'retrieval.{key}', getattr(self, key))

    def index_dict(self):
        return {key: getattr(self, key) for key in ('neighbors', 'max_len', 'horizon')}


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    weight: float = 0.5

    def __post_init__(self):
        boolean('memory.enabled', self.enabled)
        nonnegative('memory.weight', self.weight)


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    threads: int = 2
    device: str = 'auto'
    inference_batch_size: int = 512
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be an integer in 0..2**31-1.')
        positive_int('threads', self.threads)
        positive_int('inference_batch_size', self.inference_batch_size)
        if self.device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda.')
        for key, cls in SECTIONS.items():
            if not isinstance(getattr(self, key), cls):
                raise ValueError(f'Invalid section {key}.')

    def to_dict(self):
        return asdict(self)


SECTIONS = {'retrieval': RetrievalConfig, 'memory': MemoryConfig}


def checked(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f'{cls.__name__} must be an object.')
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} keys: {sorted(unknown)}')
    return cls(**value)


def from_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError('Configuration must be an object.')
    raw = dict(raw)
    for key, cls in SECTIONS.items():
        raw[key] = checked(cls, raw.get(key, {}))
    return checked(Config, raw)


def load_config(path: str | Path):
    return from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
