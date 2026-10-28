"""Configuration: dataclasses with defaults, TOML loading, durations and XDG paths."""

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
import re
import tomllib


class ConfigError(ValueError):
    """A config file or value that cannot be used."""


_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(value):
    """Seconds from a number or a string such as "500ms", "10s", "2m", "1h", "1d"."""
    if isinstance(value, bool):
        raise ConfigError(f"Invalid duration: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)\s*", str(value))
    if not match:
        raise ConfigError(f"Invalid duration: {value!r} (use e.g. 500ms, 10s, 2m, 1h, 1d)")
    return float(match.group(1)) * _UNITS[match.group(2)]


def _xdg(variable, fallback):
    base = os.environ.get(variable)
    return (Path(base) if base else Path.home() / fallback) / "senses"


def config_dir():
    return _xdg("XDG_CONFIG_HOME", ".config")


def data_dir():
    return _xdg("XDG_DATA_HOME", ".local/share")


@dataclass
class ModelConfig:
    url: str | None = None
    server_bin: str = "llama-server"
    hf: str = "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
    device: str = "auto"
    port: int | None = None
    alias: str = "senses"
    timeout: float = 30.0
    slots: int = 4


@dataclass
class ScreenConfig:
    backend: str = "auto"
    command: str | None = None
    interval: float = 2.0
    min_change: float = 0.002
    ocr_per_minute: int = 30


@dataclass
class CameraConfig:
    enabled: bool = True
    device: int | str | None = None


@dataclass
class AudioConfig:
    enabled: bool = True
    mic: str | None = None
    desktop: str | None = None
    max_utterance: float = 30.0


@dataclass
class PresenceConfig:
    question: str = "Is a person sitting in front of the computer?"
    enter: float = 0.7
    leave: float = 0.3
    enter_after: float = 2.0
    leave_after: float = 5.0
    fast_hz: float = 1.0
    slow_hz: float = 0.3


@dataclass
class SummaryConfig:
    min_interval: float = 30.0
    scene_interval: float = 60.0


@dataclass
class StoreConfig:
    path: Path | None = None
    retention_days: int = 30

    def resolved_path(self):
        return Path(self.path).expanduser() if self.path else data_dir() / "senses.db"


@dataclass
class Rule:
    on: str
    run: list[str]
    after: float = 0.0
    match: str | None = None


@dataclass
class PrivacyConfig:
    exclude: list[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


def _default_rules():
    return [Rule(on="presence.away", after=10.0, run=["@media.pause", "@screen.lock"])]


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    rules: list[Rule] = field(default_factory=_default_rules)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


_DURATIONS = {"interval", "after", "min_interval", "scene_interval", "enter_after", "leave_after",
              "max_utterance", "timeout"}
_SECTIONS = {f.name: f.default_factory for f in fields(Config) if f.name != "rules"}


def _build(cls, values, where):
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"Unknown config key: {where}.{key}")
        kwargs[key] = parse_duration(value) if key in _DURATIONS else value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Invalid [{where}]: {exc}") from exc


def load_config(path=None):
    path = Path(path) if path else config_dir() / "config.toml"
    if not path.exists():
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    sections = {}
    for name, values in data.items():
        if name == "rules":
            if not isinstance(values, list):
                raise ConfigError("rules must be an array of tables ([[rules]])")
            sections["rules"] = [_build(Rule, rule, "rules") for rule in values]
        elif name in _SECTIONS:
            if not isinstance(values, dict):
                raise ConfigError(f"[{name}] must be a table")
            sections[name] = _build(type(_SECTIONS[name]()), values, name)
        else:
            raise ConfigError(f"Unknown config section: [{name}]")
    return Config(**sections)
