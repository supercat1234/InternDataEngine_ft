"""Episode-level lighting and texture choices; no image or asset-file edits."""
from pathlib import Path
import math
import random


def sample_lighting(cfg, rng=None):
    rng = rng or random
    nominal = float(cfg.get("nominal_intensity", 1000.))
    low, high = map(float, cfg.get("intensity_scale_range", [.8, 1.2]))
    if nominal <= 0 or not 0 < low <= high:
        raise ValueError("Lighting intensity and ordered scale bounds must be positive")
    modes = cfg.get("color_temperature_ranges", {"white": [5500, 6500], "warm": [3000, 4000]})
    if not modes:
        raise ValueError("At least one light color mode is required")
    for bounds in modes.values():
        if len(bounds) != 2 or not 1000 <= bounds[0] <= bounds[1] <= 10000:
            raise ValueError("Color temperature bounds must be ordered within 1000..10000 K")
    base = list(map(float, cfg.get("base_rotation_deg", [0, 0, 0])))
    jitter = list(map(float, cfg.get("rotation_jitter_deg", [5, 5, 20])))
    if len(base) != 3 or len(jitter) != 3 or any(value < 0 for value in jitter):
        raise ValueError("Rotation base/jitter must have three values; jitter must be nonnegative")
    scale = rng.uniform(low, high)
    mode = rng.choice(list(modes))
    return {"intensity": nominal * scale, "nominal_intensity": nominal, "intensity_scale": scale,
            "color_mode": mode, "temperature_kelvin": rng.uniform(*modes[mode]),
            "rotation_deg": [value + rng.uniform(-delta, delta) for value, delta in zip(base, jitter)]}


def sample_texture_option(options, asset_root, randomized=True, rng=None, allow_solid=False):
    rng = rng or random
    woods = [entry for entry in options if entry.get("enabled", True)]
    if not woods:
        raise ValueError("At least one enabled texture is required")
    weights = [float(wood.get("weight", 1.)) for wood in woods]
    if any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("Texture weights must be finite, nonnegative and have a positive sum")
    # Validate all enabled files, so a missing variant never silently reduces diversity.
    for wood in woods:
        if allow_solid and "color" in wood:
            color = wood["color"]
            if len(color) != 3 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in color):
                raise ValueError(f"Solid color must have three finite components within 0..1: {wood['name']}")
            continue
        if not wood.get("file") or not (Path(asset_root) / wood["file"]).is_file():
            raise ValueError(f"Missing enabled texture: {wood['name']}: {wood.get('file')}")
    wood = woods[0]
    if randomized:
        wood = (rng.choices(woods, weights=weights, k=1)[0] if any("weight" in entry for entry in woods)
                else rng.choice(woods))
    if allow_solid and "color" in wood:
        return {"name": wood["name"], "color": list(map(float, wood["color"]))}
    return {"name": wood["name"], "texture_file": str((Path(asset_root) / wood["file"]).resolve())}


def sample_fixture_appearance(cfg, asset_root, rng=None):
    rng = rng or random
    tones = cfg["color_options"]
    if not tones:
        raise ValueError("At least one color option is required")
    texture = sample_texture_option(cfg["texture_options"], asset_root,
                                    cfg.get("apply_randomization", False), rng, allow_solid=True)
    if "color" in texture:
        return {"wood": texture["name"], "tone": "solid", "surface_type": "solid",
                "color": texture["color"], "texture_file": None}
    tone = rng.choice(tones) if cfg.get("apply_randomization", False) else tones[0]
    scale, bias = tone.get("scale", [1, 1, 1, 1]), tone.get("bias", [0, 0, 0, 0])
    if len(scale) != 4 or len(bias) != 4:
        raise ValueError("UsdUVTexture scale and bias must each have four components")
    return {"wood": texture["name"], "tone": tone["name"],
            "texture_file": texture["texture_file"],
            "scale": list(map(float, scale)), "bias": list(map(float, bias))}


def apply_fixture_appearance(stage, fixture_path, cfg, asset_root, use_mem=False):
    from pxr import Gf, Sdf, UsdShade
    state = cfg.get("mem_appearance") if use_mem else None
    state = state or sample_fixture_appearance(cfg, asset_root)
    shader_path = str(fixture_path).rstrip("/") + "/" + cfg["texture_shader"].lstrip("/")
    shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
    if not shader or shader.GetIdAttr().Get() != "UsdUVTexture":
        raise ValueError(f"Expected an existing UsdUVTexture shader: {shader_path}")
    # Older saved configurations omit surface_shader; the original table uses
    # Surface next to WoodTexture. Restore its connection when replaying them.
    surface_path = cfg.get("surface_shader", cfg["texture_shader"].rsplit("/", 1)[0] + "/Surface")
    surface = UsdShade.Shader(stage.GetPrimAtPath(str(fixture_path).rstrip("/") + "/" + surface_path.lstrip("/")))
    if state.get("surface_type") == "solid" or cfg.get("surface_shader") or surface:
        if not surface or surface.GetIdAttr().Get() != "UsdPreviewSurface":
            raise ValueError(f"Expected an existing UsdPreviewSurface shader: {surface_path}")
        diffuse = surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if state.get("surface_type") == "solid":
            diffuse.DisconnectSource()
            diffuse.Set(Gf.Vec3f(*state["color"]))
            cfg["mem_appearance"] = state
            return state
        # Restore the texture connection after a previous solid-color episode.
        diffuse.ConnectToSource(shader.ConnectableAPI(), "rgb")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(state["texture_file"]))
    shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*state["scale"]))
    shader.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*state["bias"]))
    cfg["mem_appearance"] = state
    return state
