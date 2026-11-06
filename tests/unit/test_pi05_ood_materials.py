"""USD material isolation; run with scripts/isaac_python.sh for USD coverage."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from ood_scene import tint_object
try:
    from pxr import Gf, Sdf, Usd, UsdShade
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, 'USD libraries require the Isaac Python environment')
class OODMaterialTests(unittest.TestCase):
    def test_actual_assets_are_tinted_only_in_session_layer(self):
        for name in ['paper_cup_001', 'ceramic_plate_001']:
            with self.subTest(asset=name):
                path = ROOT / 'workflows/simbox/assets/custom/objects' / name / 'Aligned_obj.usd'
                before_file = path.read_bytes()
                stage = Usd.Stage.Open(str(path.resolve()))
                before_root = stage.GetRootLayer().ExportToString()
                modified = tint_object(stage, '/World/Aligned', [.18,.45,.9])
                self.assertTrue(modified)
                self.assertEqual(stage.GetRootLayer().ExportToString(), before_root)
                self.assertEqual(path.read_bytes(), before_file)
                self.assertIn('inputs:scale', stage.GetSessionLayer().ExportToString())
                self.assertNotIn('roughness', stage.GetSessionLayer().ExportToString())
                self.assertNotIn('physics:', stage.GetSessionLayer().ExportToString())

    def test_roughness_texture_and_original_color_scale_are_preserved(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/Object')
        surface = UsdShade.Shader.Define(stage, '/Object/Surface')
        surface.CreateIdAttr('UsdPreviewSurface')
        color = UsdShade.Shader.Define(stage, '/Object/Color')
        color.CreateIdAttr('UsdUVTexture')
        color.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.5,.8,1,.7))
        surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
            color.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
        roughness = UsdShade.Shader.Define(stage, '/Object/Roughness')
        roughness.CreateIdAttr('UsdUVTexture')
        roughness.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1,1,1,1))
        surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).ConnectToSource(
            roughness.CreateOutput('r', Sdf.ValueTypeNames.Float))
        tint_object(stage, '/Object', [.2,.5,.8])
        for actual, expected in zip(color.GetInput('scale').Get(), [.1,.4,.8,.7]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(tuple(roughness.GetInput('scale').Get()), (1,1,1,1))


if __name__ == '__main__':
    unittest.main()
