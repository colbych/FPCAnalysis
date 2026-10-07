#!/usr/bin/env python
# generate_fpc_golden.py>
#
# Regenerates tests/golden/fpc_regression_golden.npz, the golden-output
# baseline used by tests/test_fpc_regression.py.
#
# This is a deliberate, manually-run script - NOT part of the pytest suite -
# because regenerating the golden file is exactly the thing that should
# never happen silently. Run it once, by hand, from a version of the code
# you trust (e.g. before starting any optimization work in
# docs/optimization_plan.md), review the diff of the resulting .npz's
# describing metadata, and commit it. Only rerun it if you are intentionally
# changing expected numerical output (and say so in the commit message).
#
# Usage (from repo root, with the FPCAnalysis conda env active):
#   python tests/generate_fpc_golden.py

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))  # repo root, for `import FPCAnalysis`
sys.path.insert(0, _THIS_DIR)  # tests/, for flat `import _synthetic` (no tests/__init__.py)

import FPCAnalysis as fpca
from _synthetic import make_synthetic_particles, default_box_from_domain

FIXTURE_PATH = 'tests/testdata/dHybridR/M06_th45/'
FIXTURE_NUM = '2000'
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), 'golden', 'fpc_regression_golden.npz')

N_PARTICLES = 20000
SEED = 1234
VMAX = 5.0
DV = 1.0


def _run_case(dfields, dpar, fieldkey, usebox):
    x1, x2, y1, y2, z1, z2 = default_box_from_domain(dfields)
    vx, vy, vz, totalptcl, hist, cor = fpca.fpc.compute_hist_and_cor(
        VMAX, DV, x1, x2, y1, y2, z1, z2, dpar, dfields, fieldkey, useBoxFAC=usebox,
    )
    return vx, vy, vz, totalptcl, hist, cor


def main():
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)

    dfields = fpca.ddhr.field_loader(path=FIXTURE_PATH, num=FIXTURE_NUM)
    dpar = make_synthetic_particles(dfields, N_PARTICLES, seed=SEED)

    cases = {
        # (fieldkey, useBoxFAC) -> case name
        'ex_box': ('ex', True),
        'ey_box': ('ey', True),
        'ez_box': ('ez', True),
        'epar_box': ('epar', True),
        'epar_local': ('epar', False),
    }

    out = {}
    for name, (fieldkey, usebox) in cases.items():
        print(f"Running case '{name}' (fieldkey={fieldkey}, useBoxFAC={usebox})...")
        vx, vy, vz, totalptcl, hist, cor = _run_case(dfields, dpar, fieldkey, usebox)
        out[f'{name}_vx'] = vx
        out[f'{name}_vy'] = vy
        out[f'{name}_vz'] = vz
        out[f'{name}_totalptcl'] = totalptcl
        out[f'{name}_hist'] = hist
        out[f'{name}_cor'] = cor

    out['_meta_n_particles'] = N_PARTICLES
    out['_meta_seed'] = SEED
    out['_meta_vmax'] = VMAX
    out['_meta_dv'] = DV
    out['_meta_case_names'] = np.array(list(cases.keys()))

    np.savez(GOLDEN_PATH, **out)
    print(f"Wrote golden baseline to {GOLDEN_PATH}")


if __name__ == '__main__':
    main()
