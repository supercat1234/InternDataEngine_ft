"""Lighting bounds, reproducibility and non-destructive table material changes."""
import importlib.util
import ast
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import tempfile

import numpy as np
import yaml
from pxr import Usd, UsdShade

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("appearance", ROOT / "workflows/simbox/core/utils/appearance_randomization.py")
appearance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appearance)


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        task = yaml.safe_load((ROOT / "local_configs/cup_plate_desk/task.yaml").read_text())["tasks"][0]
        arena = yaml.safe_load((ROOT / "local_configs/cup_plate_desk/arena.yaml").read_text())
        self.light = task["env_map"]["photometric_randomization"]
        self.table = arena["fixtures"][0]["appearance_randomization"]
        self.floor = arena['fixtures'][1]['texture']
        self.assets = ROOT / task["asset_root"]

    def test_lighting_ranges_modes_and_seed(self):
        rng = random.Random(42)
        samples = [appearance.sample_lighting(self.light, rng) for _ in range(300)]
        self.assertEqual({s["color_mode"] for s in samples}, {"white", "warm"})
        for state in samples:
            self.assertTrue(800 <= state["intensity"] <= 1200)
            low, high = self.light["color_temperature_ranges"][state["color_mode"]]
            self.assertTrue(low <= state["temperature_kelvin"] <= high)
            self.assertTrue(all(abs(x) <= d for x, d in zip(state["rotation_deg"], [5, 5, 20])))
        self.assertEqual(samples[0], appearance.sample_lighting(self.light, random.Random(42)))

    def test_invalid_lighting_configuration_is_rejected(self):
        for change in [{"intensity_scale_range": [1.2, .8]}, {"nominal_intensity": 0},
                       {"color_temperature_ranges": {"warm": [4000, 3000]}},
                       {"rotation_jitter_deg": [5, -5, 20]}]:
            with self.assertRaises(ValueError):
                appearance.sample_lighting({**self.light, **change})

    def test_disabled_second_wood_is_not_sampled_and_both_tones_exist(self):
        self.table["texture_options"] = self.table["texture_options"][:2]
        self.table["texture_options"][1]["enabled"] = False
        rng = random.Random(42)
        samples = [appearance.sample_fixture_appearance(self.table, self.assets, rng) for _ in range(100)]
        self.assertEqual({s["wood"] for s in samples}, {"wood_1"})
        self.assertEqual({s["tone"] for s in samples}, {"standard", "light"})

    def test_enabled_missing_texture_fails_clearly(self):
        self.table["texture_options"][1].update(enabled=True, file='missing_wood2_test.png')
        with self.assertRaisesRegex(ValueError, "wood_2"):
            appearance.sample_fixture_appearance(self.table, self.assets)

    def test_four_table_surfaces_form_six_appearances(self):
        rng = random.Random(42)
        samples = [appearance.sample_fixture_appearance(self.table, self.assets, rng) for _ in range(100)]
        self.assertEqual(len({(s["wood"], s["tone"]) for s in samples}), 6)

    def test_second_wood_has_thirty_percent_total_probability(self):
        from collections import Counter
        rng = random.Random(42)
        counts = Counter()
        for _ in range(10000):
            state = appearance.sample_fixture_appearance(self.table, self.assets, rng)
            counts[(state['wood'], state['tone'])] += 1
        for key, probability in {('wood_1', 'standard'): .20, ('wood_1', 'light'): .20,
                                 ('wood_2', 'standard'): .15, ('wood_2', 'light'): .15,
                                 ('solid_wood', 'solid'): .15, ('solid_white', 'solid'): .15}.items():
            self.assertAlmostEqual(counts[key] / 10000, probability, delta=.015)
        self.assertAlmostEqual(sum(n for (wood, _), n in counts.items() if wood == 'wood_2') / 10000,
                               .3, delta=.015)

    def test_invalid_texture_weights_are_rejected(self):
        self.table["texture_options"] = self.table["texture_options"][:2]
        for weights in [(-1, 1), (0, 0), (float('nan'), 1), (float('inf'), 1)]:
            for wood, weight in zip(self.table['texture_options'], weights):
                wood['weight'] = weight
            with self.assertRaisesRegex(ValueError, 'Texture weights'):
                appearance.sample_fixture_appearance(self.table, self.assets)

    def test_second_wood_can_be_bound_with_standard_and_light_tones(self):
        from PIL import Image
        self.table["texture_options"] = self.table["texture_options"][:2]
        self.table['texture_options'].reverse()
        self.table['apply_randomization'] = False
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/Table', 'Xform').GetReferences().AddReference(
            str(self.assets / 'custom/objects/wood_table/wood_table_v2.usd'))
        for _ in range(2):
            state = appearance.apply_fixture_appearance(stage, '/Table', self.table, self.assets)
            self.assertEqual(state['wood'], 'wood_2')
            with Image.open(state['texture_file']) as texture:
                texture.verify()
            shader = UsdShade.Shader(stage.GetPrimAtPath('/Table/Looks/LightOak/WoodTexture'))
            self.assertEqual(shader.GetInput('file').Get().path, state['texture_file'])
            np.testing.assert_allclose(shader.GetInput('scale').Get(), state['scale'])
            np.testing.assert_allclose(shader.GetInput('bias').Get(), state['bias'])
            self.table['color_options'].reverse()

    def test_shading_changes_preserve_geometry_bindings_and_replay(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/Table', 'Xform').GetReferences().AddReference(
            str(self.assets / 'custom/objects/wood_table/wood_table_v2.usd'))
        legs = stage.GetPrimAtPath('/Table/Leg_FL')
        material_before = UsdShade.MaterialBindingAPI(legs).ComputeBoundMaterial()[0].GetPath()
        points_before = stage.GetPrimAtPath('/Table/Desktop').GetAttribute('points').Get()
        self.table['apply_randomization'] = False
        self.table['color_options'].reverse()  # force light
        state = appearance.apply_fixture_appearance(stage, '/Table', self.table, self.assets)
        self.assertEqual(state['tone'], 'light')
        shader = UsdShade.Shader(stage.GetPrimAtPath('/Table/Looks/LightOak/WoodTexture'))
        np.testing.assert_allclose(shader.GetInput('bias').Get(), self.table['color_options'][0]['bias'])
        self.table['color_options'].reverse()
        self.assertEqual(appearance.apply_fixture_appearance(stage, '/Table', self.table, self.assets, use_mem=True), state)
        appearance.apply_fixture_appearance(stage, '/Table', self.table, self.assets)
        np.testing.assert_allclose(shader.GetInput('scale').Get(), [1, 1, 1, 1])
        np.testing.assert_allclose(shader.GetInput('bias').Get(), [0, 0, 0, 0])
        self.assertEqual(UsdShade.MaterialBindingAPI(legs).ComputeBoundMaterial()[0].GetPath(), material_before)
        self.assertEqual(stage.GetPrimAtPath('/Table/Desktop').GetAttribute('points').Get(), points_before)

    def test_solid_colors_disconnect_grain_and_replay_then_restore_legacy_wood(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/Table', 'Xform').GetReferences().AddReference(
            str(self.assets / 'custom/objects/wood_table/wood_table_v2.usd'))
        surface = UsdShade.Shader(stage.GetPrimAtPath('/Table/Looks/LightOak/Surface'))
        diffuse = surface.GetInput('diffuseColor')
        points = stage.GetPrimAtPath('/Table/Desktop').GetAttribute('points').Get()
        legs = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/Table/Leg_FL')).ComputeBoundMaterial()[0].GetPath()
        original = deepcopy(self.table)
        self.table['apply_randomization'] = False
        for name in ('solid_wood', 'solid_white'):
            self.table['texture_options'] = [next(e for e in original['texture_options'] if e['name'] == name)]
            state = appearance.apply_fixture_appearance(stage, '/Table', self.table, self.assets)
            self.assertFalse(diffuse.HasConnectedSource())
            self.assertEqual(state['surface_type'], 'solid')
            self.assertIsNone(state['texture_file'])
            np.testing.assert_allclose(diffuse.Get(), self.table['texture_options'][0]['color'])
            self.assertEqual(appearance.apply_fixture_appearance(stage, '/Table', self.table, self.assets, use_mem=True), state)
        legacy = deepcopy(original)
        legacy.pop('surface_shader')
        legacy['apply_randomization'] = False
        legacy['texture_options'] = legacy['texture_options'][:2]
        state = appearance.apply_fixture_appearance(stage, '/Table', legacy, self.assets)
        self.assertTrue(diffuse.HasConnectedSource())
        self.assertEqual(state['wood'], 'wood_1')
        self.assertAlmostEqual(surface.GetInput('roughness').Get(), .72)
        self.assertEqual(stage.GetPrimAtPath('/Table/Desktop').GetAttribute('points').Get(), points)
        self.assertEqual(UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/Table/Leg_FL')).ComputeBoundMaterial()[0].GetPath(), legs)

    def test_solid_color_rejects_invalid_rgb(self):
        for rgb in ([1, 1], [1, -1, 0], [float('nan'), 0, 0], [1.1, 0, 0]):
            self.table['texture_options'] = [{'name': 'bad_color', 'color': rgb}]
            with self.assertRaisesRegex(ValueError, 'Solid color'):
                appearance.sample_fixture_appearance(self.table, self.assets)

    def test_episode_save_passes_serializable_randomization_to_logger(self):
        source = ROOT / 'workflows/simbox_dual_workflow.py'
        cls = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'SimBoxDualWorkFlow')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'save')
        namespace = dict(os=os, datetime=datetime, deepcopy=deepcopy)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), namespace)
        light = appearance.sample_lighting(self.light, random.Random(42))
        table = appearance.sample_fixture_appearance(self.table, self.assets, random.Random(42))
        geometry = {'height_m': .742, 'nominal_height_m': .75, 'scale': [.001, .001, .742/750]}
        robot_height = {'nominal_offset_m': .08, 'jitter_m': .01, 'height_offset_m': .09}
        cfg = dict(env_map={'mem_light': light}, arena={'fixtures': [
            {'name': 'table', 'appearance_randomization': {'mem_appearance': table},
             'height_randomization': {'mem_height': geometry}},
            {'name': 'floor', 'texture': {'mem_texture': {'name': 'tile_2',
                'texture_file': 'custom/materials/floor_tiles/tile_02.jpg'}}}]},
            shared_region_sampling={'enabled': True}, mem_shared_layout={'clearance': .01, 'poses': {}},
            robots=[{'name': 'split_aloha', 'base_height_randomization': {'mem_height': robot_height}}],
            cameras=[{'name': 'split_aloha_external', 'mem_pose': {
                'translation': [.001, -.48, 1.44], 'orientation_wxyz': [1, 0, 0, 0]}}],
            mem_grasp_selection={'mode': 'random_top_k', 'object_from_ee': np.eye(4).tolist()})
        logger = SimpleNamespace(proprio_data_logger={'split_aloha': {}}, add_json_data=Mock(), save=Mock())
        workflow = SimpleNamespace(task_cfg=cfg, logger=logger, length=10)
        with tempfile.TemporaryDirectory() as destination:
            self.assertEqual(namespace['save'](workflow, destination), 10)
        robot, key, state = logger.add_json_data.call_args.args
        self.assertEqual((robot, key), ('split_aloha', 'episode_randomization'))
        self.assertEqual(set(json.loads(json.dumps(state))), {'layout', 'lighting', 'fixtures', 'fixture_geometry', 'robot_base_height', 'cameras', 'grasp'})
        self.assertEqual(state['layout']['clearance'], .01)
        self.assertEqual(state['robot_base_height']['split_aloha']['height_offset_m'], .09)
        robot_height['height_offset_m'] = 0
        self.assertEqual(state['robot_base_height']['split_aloha']['height_offset_m'], .09)
        self.assertEqual(state['fixture_geometry']['table']['height_m'], .742)
        geometry['height_m'] = 0
        self.assertEqual(state['fixture_geometry']['table']['height_m'], .742)
        self.assertEqual(state['fixtures']['floor']['name'], 'tile_2')
        self.assertEqual(state['cameras']['split_aloha_external']['translation'], [.001, -.48, 1.44])
        light['intensity'] = -1
        self.assertGreater(state['lighting']['intensity'], 0)  # independent snapshot
        logger.save.assert_called_once()

    def test_floor_weights_and_all_image_assets(self):
        from collections import Counter
        from PIL import Image
        rng = random.Random(42)
        counts = Counter(appearance.sample_texture_option(self.floor['texture_options'], self.assets, rng=rng)['name']
                         for _ in range(10000))
        for entry in self.floor['texture_options']:
            self.assertAlmostEqual(counts[entry['name']] / 10000, entry['weight'], delta=.015)
            with Image.open(self.assets / entry['file']) as texture:
                texture.verify()
        first = appearance.sample_texture_option(self.floor['texture_options'], self.assets, rng=random.Random(42))
        self.assertEqual(first, appearance.sample_texture_option(self.floor['texture_options'], self.assets,
                                                                rng=random.Random(42)))

    def test_plane_texture_switch_and_replay_preserve_tiles_and_grout(self):
        import sys
        from unittest.mock import patch
        from pxr import Sdf, UsdGeom
        tree = ast.parse((ROOT / 'workflows/simbox/core/objects/plane_object.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PlaneObject')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['_create_tiles', 'apply_texture']]
        stage = Usd.Stage.CreateInMemory()
        fixture_path = '/Floor'
        UsdGeom.Xform.Define(stage, fixture_path)
        class Material:
            def __init__(self, prim_path, **kwargs):
                self.material = UsdShade.Material.Define(stage, prim_path)
                self.texture = kwargs.get('texture_path')
            def set_texture(self, path): self.texture = path
            def set_project_uvw(self, value): self.project = value
            def set_reflection_roughness(self, value): self.roughness = value
        namespace = dict(np=np, math=__import__('math'), Sdf=Sdf, UsdGeom=UsdGeom, UsdShade=UsdShade,
                         OmniPBR=Material, get_current_stage=lambda: stage,
                         is_prim_path_valid=lambda path: bool(stage.GetPrimAtPath(path)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), '<floor>', 'exec'), namespace)
        plane = SimpleNamespace(cfg={'size': [1.2, 1.2], 'tile_size': [.6, .6]}, prim_path=fixture_path, name='floor')
        plane.apply_visual_material = lambda mat, **kwargs: UsdShade.MaterialBindingAPI.Apply(
            stage.GetPrimAtPath(fixture_path)).Bind(mat.material)
        namespace['_create_tiles'](plane, stage, fixture_path, plane.cfg)
        points = stage.GetPrimAtPath('/Floor/Tiles').GetAttribute('points').Get()
        with patch.dict(sys.modules, {'core.utils.appearance_randomization': appearance}):
            for entry in self.floor['texture_options']:
                cfg = {**self.floor, 'apply_randomization': False, 'texture_options': [entry]}
                namespace['apply_texture'](plane, self.assets, cfg)
                self.assertEqual(cfg['mem_texture']['name'], entry['name'])
                self.assertEqual(plane.mat.texture, cfg['mem_texture']['texture_file'])
                self.assertFalse(plane.mat.project)
                self.assertEqual(plane.mat.roughness, .25)
                grout = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/Floor/Grout')).ComputeBoundMaterial()[0]
                self.assertEqual(str(grout.GetPath()), '/Floor/Looks/GroutMaterial')
                self.assertEqual(stage.GetPrimAtPath('/Floor/Tiles').GetAttribute('points').Get(), points)
            state = dict(cfg['mem_texture'])
            cfg['texture_options'] = [self.floor['texture_options'][0]]
            namespace['apply_texture'](plane, self.assets, cfg, use_mem=True)
            self.assertEqual(cfg['mem_texture'], state)


if __name__ == '__main__':
    unittest.main()
