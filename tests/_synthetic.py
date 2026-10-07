# _synthetic.py>
#
# Deterministic synthetic particle generation used by the Phase 0 regression
# test (tests/test_fpc_regression.py) and the benchmark script
# (scripts/bench_fpc.py). See docs/optimization_plan.md.
#
# The checked-in test fixture (tests/testdata/dHybridR/M06_th45/) only
# contains field data, not particle data, so the FPC numerical hot path
# (fpc.py) has no real particle fixture to regress against. We generate
# reproducible synthetic particles (fixed seed) spatially distributed across
# the real field domain instead, so the regression test still exercises the
# real trilinear-interpolation + histogram + correlation code path end to
# end, just not with physically meaningful particle statistics.
#
# Not a pytest file itself (no test_ prefix) - imported by test and bench code.

import numpy as np


def make_synthetic_particles(dfields, n_particles, seed=0, vscale=1.0, q=1.0):
    """
    Generates a dHybridR-style particle dict with n_particles particles
    uniformly distributed in position across dfields's domain and normally
    distributed in velocity, using a fixed seed for reproducibility.

    Parameters
    ----------
    dfields : dict
        field data dictionary from a field_loader (used only for its domain
        bounds via the 'ex_xx'/'ex_yy'/'ex_zz' keys)
    n_particles : int
        number of synthetic particles to generate
    seed : int
        RNG seed; same seed + same n_particles always produces the same
        particle dict (bit for bit, given a fixed numpy version's PCG64)
    vscale : float
        standard deviation of the (zero-mean, normally distributed)
        synthetic velocity components
    q : float
        charge to assign to the synthetic species

    Returns
    -------
    dpar : dict
        particle data dictionary with keys p1,p2,p3,x1,x2,x3,q,
        Vframe_relative_to_sim, in the same convention as
        FPCAnalysis.data_dhybridr.read_particles
    """

    rng = np.random.default_rng(seed)

    x1lo, x1hi = float(dfields['ex_xx'][0]), float(dfields['ex_xx'][-1])
    x2lo, x2hi = float(dfields['ex_yy'][0]), float(dfields['ex_yy'][-1])
    x3lo, x3hi = float(dfields['ex_zz'][0]), float(dfields['ex_zz'][-1])

    dpar = {}
    dpar['x1'] = rng.uniform(x1lo, x1hi, n_particles)
    dpar['x2'] = rng.uniform(x2lo, x2hi, n_particles)
    dpar['x3'] = rng.uniform(x3lo, x3hi, n_particles)
    dpar['p1'] = rng.normal(0.0, vscale, n_particles)
    dpar['p2'] = rng.normal(0.0, vscale, n_particles)
    dpar['p3'] = rng.normal(0.0, vscale, n_particles)
    dpar['q'] = q
    dpar['Vframe_relative_to_sim'] = 0.0

    return dpar


def default_box_from_domain(dfields, xfrac=0.2):
    """
    Picks a deterministic sub-box of dfields's domain to run a correlation
    over: a slice of width xfrac of the domain in xx, centered in xx, using
    the full yy/zz extent (mirrors the slicing convention used by
    fpc.compute_correlation_over_x and the generateFPCfrom*.py scripts).

    Returns
    -------
    x1,x2,y1,y2,z1,z2 : floats
    """

    x1lo, x1hi = float(dfields['ex_xx'][0]), float(dfields['ex_xx'][-1])
    y1, y2 = float(dfields['ex_yy'][0]), float(dfields['ex_yy'][-1])
    z1, z2 = float(dfields['ex_zz'][0]), float(dfields['ex_zz'][-1])

    xwidth = (x1hi - x1lo) * xfrac
    xcenter = (x1hi + x1lo) / 2.0
    x1 = xcenter - xwidth / 2.0
    x2 = xcenter + xwidth / 2.0

    return x1, x2, y1, y2, z1, z2
