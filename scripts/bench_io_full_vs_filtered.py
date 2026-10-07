#!/usr/bin/env python
# bench_io_full_vs_filtered.py>
#
# Part B of the Phase 0 follow-up discussed in docs/optimization_plan.md:
# the original optimization proposal's claim was that full-frame particle
# loads (data_dhybridr.read_particles / data_tristan.load_particles - no
# spatial filtering, O(n_total) peak RAM regardless of how much you need)
# are the real scaling ceiling. scripts/bench_fpc.py's benchmark didn't
# actually demonstrate this (it read back exactly what it had just written,
# so there was no "load everything, use a slice" pattern to measure).
#
# This compares the two code paths that already exist for dHybridR, with NO
# new production code required:
#   - data_dhybridr.read_particles      : current default, always full-load
#   - data_dhybridr.read_box_of_particles : already in the codebase, does a
#     bounds-filtered h5py read
# against the SAME on-disk file (simulating one full simulation frame),
# asking for only a small spatial slice of it.
#
# Each (scale, mode) combination runs in its own subprocess so peak-RSS
# readings aren't contaminated by resource.getrusage's cumulative watermark
# or by one measurement's allocations still being live for the next.
#
# Usage:
#   FPCAnalysisenv/bin/python scripts/bench_io_full_vs_filtered.py
#   FPCAnalysisenv/bin/python scripts/bench_io_full_vs_filtered.py --n-totals 1e6,1e7,1e8 --slice-frac 0.01

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
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tests'))

FIXTURE_PATH = 'tests/testdata/dHybridR/M06_th45/'
FIXTURE_NUM = '2000'


def _peak_rss_mb():
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _write_frame_file(dfields, n_total, flnm, seed=99):
    from _synthetic import make_synthetic_particles
    import FPCAnalysis as fpca

    dpar = make_synthetic_particles(dfields, n_total, seed=seed)
    fpca.ddhr.write_particles_to_hdf5(dpar, flnm)


def _slice_bounds(dfields, slice_frac):
    from _synthetic import default_box_from_domain
    return default_box_from_domain(dfields, xfrac=slice_frac)


def _run_single(n_total, slice_frac, mode, flnm):
    import FPCAnalysis as fpca

    dfields = fpca.ddhr.field_loader(path=FIXTURE_PATH, num=FIXTURE_NUM)
    x1, x2, y1, y2, z1, z2 = _slice_bounds(dfields, slice_frac)

    t0 = time.perf_counter()
    if mode == 'full':
        dpar = fpca.ddhr.read_particles(flnm)
        # mimic what the non-presliced sweep path does next: filter to the
        # slice in memory, same as fpc.compute_hist_and_cor's gptsparticle
        gpts = (x1 <= dpar['x1']) & (dpar['x1'] <= x2) & (y1 <= dpar['x2']) & (dpar['x2'] <= y2) \
            & (z1 <= dpar['x3']) & (dpar['x3'] <= z2)
        n_in_slice = int(gpts.sum())
    elif mode == 'filtered':
        dpar = fpca.ddhr.read_box_of_particles(flnm, 0, x1, x2, y1, y2, z1, z2)
        n_in_slice = len(dpar['x1'])
    else:
        raise ValueError(mode)
    elapsed_s = time.perf_counter() - t0

    return {
        'n_total': n_total,
        'slice_frac': slice_frac,
        'mode': mode,
        'n_in_slice': n_in_slice,
        'elapsed_s': elapsed_s,
        'peak_rss_mb': _peak_rss_mb(),
    }


def _run_in_subprocess(n_total, slice_frac, mode, flnm):
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--single-run',
         f'{n_total},{slice_frac},{mode},{flnm}'],
        capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
    )
    last_line = [l for l in result.stdout.strip().splitlines() if l.strip()][-1]
    return json.loads(last_line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-totals', type=str, default='1e6,1e7,1e8')
    parser.add_argument('--slice-frac', type=float, default=0.01,
                         help="fraction of the domain's xx width requested per read (default 1%%)")
    parser.add_argument('--single-run', type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run is not None:
        n_total_s, slice_frac_s, mode, flnm = args.single_run.split(',', 3)
        result = _run_single(int(float(n_total_s)), float(slice_frac_s), mode, flnm)
        print(json.dumps(result))
        return

    n_totals = [int(float(s)) for s in args.n_totals.split(',')]

    import FPCAnalysis as fpca
    dfields = fpca.ddhr.field_loader(path=FIXTURE_PATH, num=FIXTURE_NUM)

    print(f"{'n_total':>12} | {'mode':>8} | {'n_in_slice':>10} | {'elapsed_s':>9} | {'peak_rss_mb':>12}")
    print('-' * 65)

    with tempfile.TemporaryDirectory() as tmpdir:
        for n_total in n_totals:
            flnm = os.path.join(tmpdir, f'frame_{n_total}.h5')
            print(f"  writing synthetic frame n_total={n_total}...", file=sys.stderr)
            _write_frame_file(dfields, n_total, flnm)

            for mode in ('full', 'filtered'):
                print(f"  benchmarking n_total={n_total} mode={mode}...", file=sys.stderr)
                r = _run_in_subprocess(n_total, args.slice_frac, mode, flnm)
                print(f"{r['n_total']:>12} | {r['mode']:>8} | {r['n_in_slice']:>10} | "
                      f"{r['elapsed_s']:>9.3f} | {r['peak_rss_mb']:>12.1f}")

            os.remove(flnm)


if __name__ == '__main__':
    main()
