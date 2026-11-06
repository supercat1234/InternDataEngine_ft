import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from rollout_cup_plate_pi05 import fixed_config,reproduction_config


class SceneConfigTests(unittest.TestCase):
    def test_reproduction_uses_archived_configs_and_does_not_modify_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'old';source.mkdir()
            output=root/'new';output.mkdir()
            task={'tasks':[{'arena_file':'old/path','regions':[{'saved':True}],'skills':['unused']}]}
            (source/'task.yaml').write_text(yaml.safe_dump(task))
            (source/'arena.yaml').write_text('fixtures: []\n')
            before=(source/'task.yaml').read_bytes()
            path=reproduction_config(source,output)
            restored=yaml.safe_load(path.read_text())['tasks'][0]
            self.assertEqual(restored['regions'],[{'saved':True}])
            self.assertEqual(restored['arena_file'],str(output/'arena.yaml'))
            self.assertEqual((source/'task.yaml').read_bytes(),before)
            self.assertEqual((source/'arena.yaml').read_bytes(),(output/'arena.yaml').read_bytes())
    def build(self, flags):
        settings = yaml.safe_load((ROOT / 'local_configs/cup_plate_pi05_eval/eval.yaml').read_text())
        settings['randomization'] = flags
        source_path = ROOT / settings['source_task']
        before = source_path.read_bytes()
        source = yaml.safe_load(before)['tasks'][0]
        with tempfile.TemporaryDirectory() as directory:
            task = yaml.safe_load(fixed_config(settings, Path(directory)).read_text())['tasks'][0]
            arena = yaml.safe_load(Path(task['arena_file']).read_text())
        self.assertEqual(source_path.read_bytes(), before)
        self.assertEqual(task['skills'], [])
        return settings, source, task, arena

    def test_fixed_default(self):
        settings, source, task, arena = self.build({})
        self.assertFalse(task['shared_region_sampling']['enabled'])
        self.assertEqual(task['regions'][0]['random_config']['pos_range'], [settings['cup_position']] * 2)
        self.assertEqual(task['env_map']['photometric_randomization']['intensity_scale_range'], [1, 1])
        self.assertTrue(all(not c['apply_randomization'] for c in task['cameras']))

    def test_all_randomized_preserves_expert_ranges(self):
        settings, source, task, arena = self.build(dict.fromkeys(('layout','lighting','table','floor','camera'), True))
        self.assertEqual(task['shared_region_sampling'], source['shared_region_sampling'])
        self.assertEqual(task['regions'], source['regions'])
        self.assertEqual(task['env_map']['photometric_randomization'], source['env_map']['photometric_randomization'])
        for fixture in arena['fixtures']:
            for key in ('texture', 'appearance_randomization'):
                if key in fixture:
                    self.assertTrue(fixture[key]['apply_randomization'])
        for camera in task['cameras']:
            self.assertEqual(camera['apply_randomization'], camera['name'] == 'split_aloha_external')

    def test_lighting_only_keeps_positions_and_camera_fixed(self):
        settings, source, task, arena = self.build({'lighting': True})
        self.assertEqual(task['regions'][0]['random_config']['pos_range'], [settings['cup_position']] * 2)
        self.assertEqual(task['env_map']['photometric_randomization'], source['env_map']['photometric_randomization'])
        self.assertTrue(all(not c['apply_randomization'] for c in task['cameras']))

if __name__ == '__main__':
    unittest.main()
