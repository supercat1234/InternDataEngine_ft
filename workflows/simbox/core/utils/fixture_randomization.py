"""Episode-level height changes for upright fixtures rooted at floor level."""
import math
import random


def sample_fixture_height(fixture_cfg, rng=None):
    cfg = fixture_cfg["height_randomization"]
    nominal = float(cfg["nominal_height"])
    low, high = map(float, cfg["height_range"])
    if not all(math.isfinite(v) for v in (nominal, low, high)) or not 0 < low <= high or nominal <= 0:
        raise ValueError("Fixture nominal height and ordered height bounds must be positive and finite")
    scale = list(map(float, fixture_cfg.get("scale", [1., 1., 1.])))
    if len(scale) != 3 or any(not math.isfinite(v) or v <= 0 for v in scale):
        raise ValueError("Fixture baseline scale must have three positive finite components")
    height = (rng or random).uniform(low, high) if cfg.get("enabled", False) else nominal
    # Always start from the configured scale, never the previous episode's scale.
    scale[2] *= height / nominal
    return {"height_m": height, "nominal_height_m": nominal, "scale": scale}


def apply_fixture_height(fixture, fixture_cfg, use_mem=False):
    cfg = fixture_cfg["height_randomization"]
    state = cfg.get("mem_height") if use_mem and cfg.get("enabled", False) else None
    state = state or sample_fixture_height(fixture_cfg)
    fixture.set_local_scale(state["scale"])
    cfg["mem_height"] = state
    return state
