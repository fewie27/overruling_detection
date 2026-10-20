from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

VALID_REASONING_EFFORT: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)

_PRICING_CONNECTORS: frozenset[str] = frozenset({"openai", "claude", "gemini", "deepseek"})
_warned_missing_pricing: set[tuple[str, str]] = set()


def models_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "models.yaml"


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, Any]:
    path = models_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model catalog not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise ValueError(f"Invalid models catalog in {path}: missing 'providers' mapping")
    return providers


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in {"anthropic"}:
        return "claude"
    return normalized


def _provider_entry(provider: str) -> dict[str, Any] | None:
    entry = _load_catalog().get(_normalize_provider(provider))
    return entry if isinstance(entry, dict) else None


def resolve_model_name(provider: str, model: str | None) -> str:
    """Resolve model: explicit CLI value, else provider default from YAML."""
    if model and model.strip():
        return model.strip()
    entry = _provider_entry(provider)
    if entry is None:
        raise ValueError(f"Unknown provider {provider!r} in models catalog")
    default = entry.get("default_model")
    if not isinstance(default, str) or not default.strip():
        raise ValueError(f"No default_model configured for provider {provider!r}")
    return default.strip()


def model_entry(provider: str, model: str) -> dict[str, Any] | None:
    entry = _provider_entry(provider)
    if entry is None:
        return None
    models = entry.get("models")
    if not isinstance(models, dict):
        return None
    spec = models.get(model)
    return spec if isinstance(spec, dict) else None


def list_known_models(provider: str) -> list[str]:
    entry = _provider_entry(provider)
    if entry is None:
        return []
    models = entry.get("models")
    if not isinstance(models, dict):
        return []
    return sorted(models.keys())


def _rate_from_spec(spec: dict[str, Any], key: str) -> Optional[float]:
    val = spec.get(key)
    if val is None:
        return None
    try:
        rate = float(val)
    except (TypeError, ValueError):
        return None
    if rate < 0:
        return None
    return rate


def lookup_pricing(
    provider: str,
    model: str,
    *,
    use_batch: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    """Return (input_usd_per_1m, output_usd_per_1m) from the catalog, or (None, None)."""
    spec = model_entry(provider, model)
    if spec is None:
        return None, None

    if use_batch:
        batch_in = _rate_from_spec(spec, "batch_input_usd_per_1m")
        batch_out = _rate_from_spec(spec, "batch_output_usd_per_1m")
        if batch_in is not None and batch_out is not None:
            return batch_in, batch_out
        std_in = _rate_from_spec(spec, "input_usd_per_1m")
        std_out = _rate_from_spec(spec, "output_usd_per_1m")
        if std_in is not None and std_out is not None:
            return std_in * 0.5, std_out * 0.5

    inp = _rate_from_spec(spec, "input_usd_per_1m")
    out = _rate_from_spec(spec, "output_usd_per_1m")
    return inp, out


def warn_if_missing_pricing(provider: str, model: str) -> None:
    """Emit a one-time warning when a priced provider has no rates for the model."""
    normalized = _normalize_provider(provider)
    if normalized not in _PRICING_CONNECTORS:
        return
    key = (normalized, model)
    if key in _warned_missing_pricing:
        return
    in_rate, out_rate = lookup_pricing(normalized, model, use_batch=False)
    if in_rate is not None and out_rate is not None:
        return
    _warned_missing_pricing.add(key)
    spec = model_entry(normalized, model)
    if spec is None:
        msg = (
            f"Model {model!r} is not listed under provider {normalized!r} in "
            f"{models_config_path()}. Cost estimates will be omitted."
        )
    else:
        msg = (
            f"No pricing configured for {normalized}/{model} in {models_config_path()}. "
            "Cost estimates will be omitted."
        )
    warnings.warn(msg, stacklevel=3)


def parse_reasoning_effort(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip().lower()
    if raw not in VALID_REASONING_EFFORT:
        raise ValueError(
            f"Invalid reasoning effort {value!r}; expected one of: {sorted(VALID_REASONING_EFFORT)}"
        )
    return raw  # type: ignore[return-value]
