# test_fpc_regression.py>
#
# Golden-output regression test for the fpc.py hot path (see
# docs/optimization_plan.md, Phase 0). This is the first automated test that
# checks FPC *numerical* output rather than loader shape/sanity - it did not
# exist before this effort, and any change to fpc.py, data_dhybridr.py's
# field/particle loaders, or analysis.py's basis-conversion functions should
# be run against this before being considered safe.
#
# Methodology: the checked-in test fixture (tests/testdata/dHybridR/M06_th45/)
# only has field data, not particle data, so we generate deterministic
# synthetic particles (fixed seed, see tests/_synthetic.py) spread across the
# real field domain and run fpc.compute_hist_and_cor on them. The exact
# numbers are not physically meaningful (synthetic particles), but they
# exercise the real trilinear interpolation / histogram / correlation code
# path end to end, and must stay identical across any optimization that
# claims not to change output.
#
# If you intentionally change expected output, regenerate the golden file
# with `python tests/generate_fpc_golden.py` and review the diff/commit
# message explaining why the numbers changed.

import os
import unittest

import numpy as np

import FPCAnalysis as fpca
from _synthetic import make_synthetic_particles, default_box_from_domain

FIXTURE_PATH = 'tests/testdata/dHybridR/M06_th45/'
FIXTURE_NUM = '2000'
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden', 'fpc_regression_golden.npz')

VMAX = 5.0
DV = 1.0

CASES = {
    'ex_box': ('ex', True),
    'ey_box': ('ey', True),
    'ez_box': ('ez', True),
    'epar_box': ('epar', True),
    'epar_local': ('epar', False),
}


class TestFPCRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GOLDEN_PATH):
            raise unittest.SkipTest(
                f"Golden baseline not found at {GOLDEN_PATH}. "
                "Run `python tests/generate_fpc_golden.py` once to create it."
            )
        cls.golden = np.load(GOLDEN_PATH, allow_pickle=True)
        cls.n_particles = int(cls.golden['_meta_n_particles'])
        cls.seed = int(cls.golden['_meta_seed'])

        cls.dfields = fpca.ddhr.field_loader(path=FIXTURE_PATH, num=FIXTURE_NUM)
        cls.dpar = make_synthetic_particles(cls.dfields, cls.n_particles, seed=cls.seed)

    def _check_case(self, name):
        fieldkey, usebox = CASES[name]
        x1, x2, y1, y2, z1, z2 = default_box_from_domain(self.dfields)

        vx, vy, vz, totalptcl, hist, cor = fpca.fpc.compute_hist_and_cor(
            VMAX, DV, x1, x2, y1, y2, z1, z2, self.dpar, self.dfields, fieldkey, useBoxFAC=usebox,
        )

        np.testing.assert_allclose(vx, self.golden[f'{name}_vx'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vy, self.golden[f'{name}_vy'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(vz, self.golden[f'{name}_vz'], rtol=1e-10, atol=1e-12)
        self.assertEqual(totalptcl, float(self.golden[f'{name}_totalptcl']))
        np.testing.assert_allclose(hist, self.golden[f'{name}_hist'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(cor, self.golden[f'{name}_cor'], rtol=1e-8, atol=1e-10)

    def test_ex_box(self):
        self._check_case('ex_box')

    def test_ey_box(self):
        self._check_case('ey_box')

    def test_ez_box(self):
        self._check_case('ez_box')

    def test_epar_box(self):
        self._check_case('epar_box')

    def test_epar_local(self):
        self._check_case('epar_local')


if __name__ == '__main__':
    unittest.main()
