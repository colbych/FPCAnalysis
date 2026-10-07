#!/usr/bin/env python
# bench_fpc.py>
#
# Benchmarks the particle load path and the fpc.py hot path at a few
# synthetic particle-count scales, for Phase 0 of docs/optimization_plan.md.
# Produces timing + peak-RSS numbers to fill in that doc's baseline table
# before Phase 1 (I/O streaming) work starts, and to compare against after
# each later phase.
#
# What it measures per scale N:
#   write  : writing N synthetic particles to a dHybridR-style HDF5 file
#            (models what preslicedata*.py produces)
#   load   : data_dhybridr.read_particles reading that whole file back in
#            (this is the function Phase 1 targets - it currently loads the
#            entire file with no spatial filtering)
#   fpc    : fpc.compute_hist_and_cor on a sub-box of the loaded particles
#   peak_rss_mb : this process's peak resident set size during the above
#
# Peak RSS (resource.getrusage) is a cumulative, monotonically-increasing
# watermark for the lifetime of a process, not a per-call delta. To get an
# accurate isolated reading per scale, each scale runs in its own freshly
# spawned subprocess; results are then collected and printed as one table.
#
# Usage (from repo root, with the FPCAnalysisenv env's python):
#   FPCAnalysisenv/bin/python scripts/bench_fpc.py
#   FPCAnalysisenv/bin/python scripts/bench_fpc.py --scales 1e5,1e6,1e7
#   FPCAnalysisenv/bin/python scripts/bench_fpc.py --single-scale 1e6   # internal, one isolated run

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tests'))  # for flat `import _synthetic`

FIXTURE_PATH = 'tests/testdata/dHybridR/M06_th45/'
FIXTURE_NUM = '2000'
VMAX = 5.0
DV = 1.0


def _peak_rss_mb():
    """Peak RSS of this process so far, in MB. Units differ by platform:
    macOS reports bytes, Linux reports KB."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _run_single_scale(n_particles):
    import FPCAnalysis as fpca
    from _synthetic import make_synthetic_particles, default_box_from_domain

    dfields = fpca.ddhr.field_loader(path=FIXTURE_PATH, num=FIXTURE_NUM)
    dpar = make_synthetic_particles(dfields, n_particles, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        flnm = os.path.join(tmpdir, 'bench_particles.h5')

        t0 = time.perf_counter()
        fpca.ddhr.write_particles_to_hdf5(dpar, flnm)
        write_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        dpar_loaded = fpca.ddhr.read_particles(flnm)
        load_s = time.perf_counter() - t0

    dpar_loaded['q'] = 1.0
    x1, x2, y1, y2, z1, z2 = default_box_from_domain(dfields)

    # warm up numba JIT compilation (compute_cprimew, weighted_field_average,
    # etc. compile once per process) on a trivial subset so the timed call
    # below measures steady-state performance, not one-time compile cost.
    _warmup_dpar = {k: (v[:10] if hasattr(v, '__len__') else v) for k, v in dpar_loaded.items()}
    fpca.fpc.compute_hist_and_cor(
        VMAX, DV, x1, x2, y1, y2, z1, z2, _warmup_dpar, dfields, 'ex', useBoxFAC=True,
    )

    t0 = time.perf_counter()
    _ = fpca.fpc.compute_hist_and_cor(
        VMAX, DV, x1, x2, y1, y2, z1, z2, dpar_loaded, dfields, 'ex', useBoxFAC=True,
    )
    fpc_s = time.perf_counter() - t0

    return {
        'n_particles': n_particles,
        'write_s': write_s,
        'load_s': load_s,
        'fpc_s': fpc_s,
        'peak_rss_mb': _peak_rss_mb(),
    }


def _run_scale_in_subprocess(n_particles):
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--single-scale', str(n_particles)],
        capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
    )
    # the single-scale mode prints progress to stderr-like lines and exactly
    # one JSON line (prefixed) to stdout as its last line
    last_line = [l for l in result.stdout.strip().splitlines() if l.strip()][-1]
    return json.loads(last_line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scales', type=str, default='1e5,1e6,1e7',
                         help="comma-separated particle counts to benchmark")
    parser.add_argument('--single-scale', type=str, default=None,
                         help=argparse.SUPPRESS)  # internal subprocess entrypoint
    args = parser.parse_args()

    if args.single_scale is not None:
        n = int(float(args.single_scale))
        result = _run_single_scale(n)
        print(json.dumps(result))
        return

    scales = [int(float(s)) for s in args.scales.split(',')]

    results = []
    for n in scales:
        print(f"Benchmarking n_particles={n} (isolated subprocess)...", file=sys.stderr)
        results.append(_run_scale_in_subprocess(n))

    print()
    print(f"{'n_particles':>12} | {'write_s':>9} | {'load_s':>9} | {'fpc_s':>9} | {'peak_rss_mb':>12}")
    print('-' * 65)
    for r in results:
        print(f"{r['n_particles']:>12} | {r['write_s']:>9.3f} | {r['load_s']:>9.3f} | "
              f"{r['fpc_s']:>9.3f} | {r['peak_rss_mb']:>12.1f}")


if __name__ == '__main__':
    main()
