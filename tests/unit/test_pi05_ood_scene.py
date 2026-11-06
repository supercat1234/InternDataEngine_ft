"""Isolation, OOD bounds and archived reproduction; no Isaac runtime needed."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from ood_scene import PROFILES
from rollout_cup_plate_pi05 import fixed_config, reproduction_config


class OODConfigTests(unittest.TestCase):
    def setUp(self):
        self.folder = ROOT / 'local_configs/cup_plate_pi05_ood'
        self.id_settings = yaml.safe_load((ROOT / 'local_configs/cup_plate_pi05_eval/eval.yaml').read_text())
        source_path = ROOT / self.id_settings['source_task']
        self.source = yaml.safe_load(source_path.read_text())['tasks'][0]
        self.id_arena = yaml.safe_load((ROOT / self.source['arena_file']).read_text())

    def build(self, profile, directory):
        settings = yaml.safe_load((self.folder / f'{profile}.yaml').read_text())
        path = fixed_config(settings, directory)
        task = yaml.safe_load(path.read_text())['tasks'][0]
        arena = yaml.safe_load((directory / 'arena.yaml').read_text())
        return settings, task, arena

    def test_profiles_change_only_selected_factors(self):
        originals = [ROOT / self.id_settings['source_task'], ROOT / self.source['arena_file'],
                     ROOT / 'local_configs/cup_plate_pi05_eval/eval.yaml']
        before = [p.read_bytes() for p in originals]
        roots = set()
        for profile in PROFILES:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                settings, task, arena = self.build(profile, Path(directory))
                active = set(PROFILES[:-1]) if profile == 'combined' else {profile}
                self.assertEqual(set(task['evaluation_ood']['active']), active)
                self.assertEqual(task['shared_region_sampling']['enabled'], 'positions' not in active)
                self.assertEqual(settings['success_criteria'], self.id_settings['success_criteria'])
                self.assertEqual(settings['checkpoint'], self.id_settings['checkpoint'])
                self.assertNotEqual(settings['output_root'], self.id_settings['output_root'])
                roots.add(settings['output_root'])
                self.assertEqual(task['env_map'], self.source['env_map'])
                self.assertEqual(task['objects'], self.source['objects'])
                self.assertEqual(task['robots'], self.source['robots'])
                if 'positions' not in active:
                    self.assertEqual(task['regions'], self.source['regions'])
                for source, fixture in zip(self.id_arena['fixtures'], arena['fixtures']):
                    group = source['name']
                    for key in ('appearance_randomization', 'texture'):
                        if key not in source:
                            continue
                        expected = deepcopy(source[key])
                        expected['apply_randomization'] = True
                        if group not in active:
                            self.assertEqual(fixture[key], expected)
                        else:
                            self.assertNotEqual(fixture[key]['texture_options'], expected['texture_options'])
                            self.assertTrue(all(Path(v['file']).is_file() for v in fixture[key]['texture_options']))
        self.assertEqual(len(roots), len(PROFILES))
        self.assertEqual([p.read_bytes() for p in originals], before)

    def test_position_boxes_are_outside_id_inside_table_and_nonoverlapping(self):
        params = yaml.safe_load((self.folder / 'parameters.yaml').read_text())
        regions = {r['object']: r['random_config']['pos_range'] for r in self.source['regions']}
        # Check whole boxes, rather than a finite sample of positions.
        for name, radius in [('paper_cup', .0375), ('ceramic_plate', .077)]:
            lo, hi = params['positions'][name]
            id_lo, id_hi = regions[name]
            self.assertTrue(any(hi[i] < id_lo[i] or lo[i] > id_hi[i] for i in (0,1)))
            for axis, extent in [(0,.7),(1,.3)]:
                self.assertGreaterEqual(lo[axis] - radius, -extent)
                self.assertLessEqual(hi[axis] + radius, extent)
        cup_hi = params['positions']['paper_cup'][1]
        plate_lo = params['positions']['ceramic_plate'][0]
        self.assertGreater(plate_lo[0] - cup_hi[0], .0375 + .077)

    def test_reproduction_keeps_effective_ood_materials_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source', Path(directory)/'output'
            source.mkdir(); output.mkdir()
            _, task, arena = self.build('combined', source)
            # Reproduction must use the archived effective task and arena, even
            # when the original OOD parameter file is unavailable.
            archived = yaml.safe_load((source/'task.yaml').read_text())
            archived['tasks'][0]['evaluation_ood']['object_colors']['paper_cup'][0]['rgb'] = [.1,.2,.3]
            (source/'task.yaml').write_text(yaml.safe_dump(archived))
            path = reproduction_config(source, output)
            reproduced = yaml.safe_load(path.read_text())['tasks'][0]
            self.assertEqual(reproduced['evaluation_ood'], archived['tasks'][0]['evaluation_ood'])
            self.assertEqual(yaml.safe_load((output/'arena.yaml').read_text()), arena)


if __name__ == '__main__':
    unittest.main()
