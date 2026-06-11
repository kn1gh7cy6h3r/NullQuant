"""
config.py — load and validate the YAML configuration.

A single Config object is threaded through the whole pipeline so that the
universe, strategy parameters, cost assumptions, and validation settings all
come from one auditable place. Access is attribute-style for nested sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class _Section(dict):
    """A dict that also supports attribute access for nested config sections."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict):
            return _Section(value)
        return value


@dataclass
class Config:
    """Top-level configuration wrapper."""

    raw: dict
    path: Path

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def __getattr__(self, item: str) -> Any:
        # dataclass fields (raw, path) are handled normally; everything else
        # falls through to the parsed YAML sections.
        raw = self.__dict__.get("raw", {})
        if item in raw:
            value = raw[item]
            return _Section(value) if isinstance(value, dict) else value
        raise AttributeError(item)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value via a dotted path, e.g. cfg.get('risk.target_portfolio_vol')."""
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(path: str | Path | None = None) -> Config:
    """Load the project configuration (defaults to config/config.yaml)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a mapping.")
    return Config(raw=raw, path=cfg_path)
