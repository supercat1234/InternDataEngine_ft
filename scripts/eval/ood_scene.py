"""Opt-in evaluation-only OOD configuration and session-local material tinting."""
from copy import deepcopy
import json
from pathlib import Path
import random

import yaml

PROFILES = ('positions', 'objects', 'table', 'floor', 'combined')


def configure_ood(task, arena, settings, repo):
    spec = settings.get('ood')
    if not spec or not spec.get('enabled', False):
        return
    profile = spec['profile']
    if profile not in PROFILES:
        raise ValueError(f'Unknown OOD profile: {profile}')
    params = yaml.safe_load((Path(repo) / spec['parameters_file']).read_text())
    active = set(PROFILES[:-1]) if profile == 'combined' else {profile}
    if 'positions' in active:
        if task.get('shared_region_sampling'):
            task['shared_region_sampling']['enabled'] = False  # Honor the explicit OOD position boxes.
        for region in task['regions']:
            region['random_config']['pos_range'] = deepcopy(params['positions'][region['object']])
        if not settings.get('randomization', {}).get('layout', False):
            raise ValueError('OOD positions require randomization.layout=true')
    for fixture in arena['fixtures']:
        name = fixture['name']
        if name == 'table' and 'table' in active:
            appearance = fixture['appearance_randomization']
            appearance['texture_options'] = deepcopy(params['table_textures'])
            appearance['color_options'] = [{'name': 'original', 'scale': [1,1,1,1], 'bias': [0,0,0,0]}]
        elif name == 'floor' and 'floor' in active:
            fixture['texture']['texture_options'] = deepcopy(params['floor_colors'])
        else:
            continue
        key = 'appearance_randomization' if name == 'table' else 'texture'
        for option in fixture[key]['texture_options']:
            path = (Path(repo) / option['file']).resolve()
            if not path.is_file():
                raise FileNotFoundError(f'OOD texture missing: {path}')
            option['file'] = str(path)
    # Archive the effective spec inside task.yaml, so reproduction does not read
    # a subsequently edited parameters file or modify any original asset.
    task['evaluation_ood'] = {'profile': profile, 'active': sorted(active),
                              'object_colors': deepcopy(params['object_colors'])}


def tint_object(stage, root_path, rgb):
    """Tint existing diffuse maps; retain print/grain, roughness and collision."""
    from pxr import Gf, Sdf, Usd, UsdShade
    modified = []
    # Only color connections are tinted, never normal/roughness texture nodes.
    # Session overrides cannot be saved back into the referenced model files.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            if shader.GetIdAttr().Get() != 'UsdPreviewSurface':
                continue
            diffuse = shader.GetInput('diffuseColor')
            if not diffuse:
                continue
            source = diffuse.GetConnectedSource()
            if source:
                texture = UsdShade.Shader(source[0].GetPrim())
                path = str(texture.GetPath())
                if path in modified:
                    continue
                if texture.GetIdAttr().Get() != 'UsdUVTexture':
                    raise ValueError(f'Unsupported diffuse color network: {path}')
                previous = texture.GetInput('scale').Get()
                if previous is None:
                    previous = (1.,1.,1.,1.)
                texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(
                    Gf.Vec4f(*(float(previous[i])*rgb[i] for i in range(3)), float(previous[3])))
                modified.append(path)
            else:
                previous = diffuse.Get()
                if previous is None:
                    previous = (1.,1.,1.)
                diffuse.Set(Gf.Vec3f(*(float(previous[i])*rgb[i] for i in range(3))))
                modified.append(str(prim.GetPath()))
    if not modified:
        raise ValueError(f'No tintable diffuse material below {root_path}')
    return modified


def apply_ood_visuals(task, task_path, output):
    archived = yaml.safe_load(Path(task_path).read_text())['tasks'][0]
    spec = archived.get('evaluation_ood')
    if not spec:
        return
    # Independent local sampler: material recoloring does not advance the
    # simulation's global random state after the scene has been initialized.
    rng = random.Random(repr(random.getstate()) + ':ood_object_colors')
    state = {'profile': spec['profile'], 'active': spec['active'], 'objects': {}, 'fixtures': {}}
    for name, obj in task.objects.items():
        position, orientation = obj.get_world_pose()
        entry = {'position': list(map(float, position)), 'orientation_wxyz': list(map(float, orientation))}
        if 'objects' in spec['active'] and name in spec['object_colors']:
            color = rng.choice(spec['object_colors'][name])
            entry.update(color=color['name'], tint_rgb=color['rgb'],
                         shaders=tint_object(task.stage, obj.prim_path, color['rgb']))
        state['objects'][name] = entry
    for name, obj in task.fixtures.items():
        cfg = obj.cfg
        state['fixtures'][name] = {
            'appearance': cfg.get('appearance_randomization', {}).get('mem_appearance'),
            'texture': cfg.get('texture', {}).get('mem_texture')}
    (Path(output) / 'ood_state.json').write_text(json.dumps(state, indent=2))
    print(f'OOD_SCENE {json.dumps(state)}', flush=True)
    return state
