# common_config.py
import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

# Keys that are common across all scripts
COMMON_KEYS = {
    "outputavgtime",
    "ssps",
    "experiment_era",
    "baseline_era",
    "input_length",
    "in_res",
    "out_res",
    "time_range",
    "file_front",
    "model_file_front",
    "input_var",
    "output_var",
    "n_train",
    "n_val",
    "test",
    "data_dir",
}

# Optional constraints for light validation
_ALLOWED_SSP = {"126", "245", "370", "585"}

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    Do light validation so we fail fast if config is malformed.
    """
    def _expect_list_of_len2(name: str) -> None:
        if name in cfg:
            v = cfg[name]
            if not isinstance(v, list) or len(v) != 2 or not all(isinstance(x, int) for x in v):
                raise ValueError(f'"{name}" must be a list of two integers, got: {v!r}')

    def _expect_int(name: str) -> None:
        if name in cfg and not isinstance(cfg[name], int):
            raise ValueError(f'"{name}" must be an integer, got: {cfg[name]!r}')

    def _expect_str(name: str) -> None:
        if name in cfg and not isinstance(cfg[name], str):
            raise ValueError(f'"{name}" must be a string, got: {cfg[name]!r}')

    def _expect_list_of_str(name: str) -> None:
        if name in cfg:
            v = cfg[name]
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f'"{name}" must be a list of strings, got: {v!r}')

    # Validate types / shapes
    _expect_int("outputavgtime")
    _expect_list_of_str("ssps")
    _expect_list_of_len2("experiment_era")
    _expect_list_of_len2("baseline_era")
    _expect_int("input_length")
    _expect_int("in_res")
    _expect_int("out_res")
    _expect_list_of_len2("time_range")
    _expect_str("file_front")
    _expect_str("model_file_front")
    _expect_str("input_var")
    _expect_str("output_var")
    _expect_int("n_train")
    _expect_int("n_val")
    _expect_list_of_len2("test")
    _expect_str("data_dir")

    # Value constraints (optional but helpful)
    if "ssps" in cfg:
        invalid = [s for s in cfg["ssps"] if s not in _ALLOWED_SSP]
        if invalid:
            raise ValueError(f'Invalid "ssps" values {invalid!r}; allowed: {_ALLOWED_SSP}')

def _subset_to_known_keys(cfg: Dict[str, Any], known: Iterable[str]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k in known}

def _normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize small inconsistencies (e.g., ensure data_dir has a trailing slash if you want it).
    Adjust as you see fit.
    """
    out = dict(cfg)
    # Example: normalize data_dir to always end with a slash
    if "data_dir" in out and isinstance(out["data_dir"], str):
        if not out["data_dir"].endswith("/"):
            out["data_dir"] = out["data_dir"] + "/"
    return out

def apply_common_config_and_parse_args(
    parser: argparse.ArgumentParser,
    *,
    config_arg: str = "--config",
    default_config_path: str = "common_params.json",
    known_keys: Iterable[str] = COMMON_KEYS,
    validate: bool = True
) -> argparse.Namespace:
    """
    Two-phase parse:
      1) Parse known args to discover --config (if present).
      2) Load JSON and set parser defaults for common keys.
      3) Parse args again so CLI overrides JSON.

    Usage:
        parser.add_argument("--config", default="common_params.json", help="Path to common JSON config")
        # ... define the rest of your args ...
        args = apply_common_config_and_parse_args(parser)
    """
    # Ensure the parser has a --config argument (idempotent)
    if not any(a.option_strings and config_arg in a.option_strings for a in parser._actions):
        parser.add_argument(config_arg, default=default_config_path, help="Path to common JSON config")

    # First pass: only to get the config path
    prelim, _ = parser.parse_known_args()
    cfg_path = getattr(prelim, config_arg.lstrip("-").replace("-", "_"), default_config_path)

    # Load & validate JSON if it exists
    config_data: Dict[str, Any] = {}
    if cfg_path and os.path.exists(cfg_path):
        config_data = _load_json(cfg_path)
        if validate:
            _validate_config(config_data)
        config_data = _normalize(config_data)

    # Apply defaults from config for known keys that the parser recognizes
    # (We don't try to set defaults for args absent from a script.)
    config_subset = _subset_to_known_keys(config_data, known_keys)
    if config_subset:
        parser.set_defaults(**config_subset)

    # Second pass: final parse with JSON defaults; CLI overrides JSON
    return parser.parse_args()