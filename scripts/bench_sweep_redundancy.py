#!/usr/bin/env python
# bench_sweep_redundancy.py>
#
# Part A of the Phase 0 follow-up discussed in docs/optimization_plan.md:
# does fpc.py's redundant per-call particle filtering + double histogramdd
# (see the TODO at fpc.py:210 - "lot of redundancy in this library") actually
# cost real wall time in a realistic multi-slice sweep, and does it scale
# with the number of slices/field-keys the way the theory predicts?
#
# Theory being tested: fpc.compute_correlation_over_x calls
# compute_hist_and_cor once per field key (ex,ey,ez) per slice. Each call
# independently rebuilds the `gptsparticle` boolean mask over the FULL
# in-memory particle array (cost ~ O(n_total), not O(particles in the box))
# and redundantly runs np.histogramdd twice. So total sweep time should have
# a component that scales with (n_slices * n_fieldkeys * n_total) on top of
# the component that scales with actual particles-in-box (roughly constant
# across different slice counts, since the particles are just redistributed
# into more/fewer/smaller boxes covering the same total swept region).
#
# Method: generate one large synthetic particle "frame" in memory (no disk
# I/O here - that's Part B's job, see bench_io_full_vs_filtered.py), run the
# real fpc.compute_correlation_over_x driver across it at a few different
# slice counts (by varying dx over a fixed swept width), and profile the
# WHOLE sweep with cProfile so repeated calls are aggregated automatically.
# Each (n_total, n_slices_requested) combination runs in its own subprocess
# for a clean numba/profiler state.
#
# Usage:
#   FPCAnalysisenv/bin/python scripts/bench_sweep_redundancy.py
#   FPCAnalysisenv/bin/python scripts/bench_sweep_redundancy.py --n-totals 1e6,1e7 --slice-counts 10,50,191,500

import argparse
import cProfile
import contextlib
import io
import json
import os
import pstats
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tests'))

FIXTURE_PATH = 'tests/testdata/dHybridR/M06_th45/'
FIXTURE_NUM = '2000'
VMAX = 5.0
DV = 1.0
SWEEP_WIDTH_FRAC = 0.5  # fraction of the domain's full xx extent that we sweep over


def _profile_breakdown(pr):
    stats = pstats.Stats(pr)
    buckets = {'filter_overhead': 0.0, 'jit_loop': 0.0, 'histogram': 0.0}
    total = 0.0
    for (_filename, _lineno, funcname), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        total += tt
        if funcname in ('compute_hist_and_cor', 'filter_dpar_to_box'):
            buckets['filter_overhead'] += tt
        elif funcname == 'compute_cprimew':
            buckets['jit_loop'] += tt
        elif funcname in ('histogramdd', 'searchsorted', 'binned_statistic_dd', '_bin_edges', '_bin_numbers', '_bincount'):
            buckets['histogram'] += tt
    buckets['other'] = total - sum(buckets.values())
    buckets['total'] = total
    return buckets


def _run_single(n_total, n_slices_requested):
    import FPCAnalysis as fpca
    from _synthetic import make_synthetic_particles

    dfields = fpca.ddhr.field_loader(path=FIXTURE_PATH, num=FIXTURE_NUM)
    dpar = make_synthetic_particles(dfields, n_total, seed=7)
    dpar['q'] = 1.0

    xlo, xhi = float(dfields['ex_xx'][0]), float(dfields['ex_xx'][-1])
    y1, y2 = float(dfields['ex_yy'][0]), float(dfields['ex_yy'][-1])
    z1, z2 = float(dfields['ex_zz'][0]), float(dfields['ex_zz'][-1])
    width = (xhi - xlo) * SWEEP_WIDTH_FRAC
    x1 = xlo + (xhi - xlo - width) / 2.0
    dx = width / n_slices_requested
    x2_end = x1 + width - dx * 1e-6  # tiny epsilon so float accumulation doesn't add an extra slice

    # warm up numba JIT on a trivial subset before the timed/profiled run
    small = {k: (v[:10] if hasattr(v, '__len__') else v) for k, v in dpar.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        fpca.fpc.compute_correlation_over_x(dfields, small, VMAX, DV, dx, 0.0,
                                             xlim=[x1, x1 + dx], ylim=[y1, y2], zlim=[z1, z2])

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        pr.enable()
        _, _, _, x_out, _, _, _, _, _ = fpca.fpc.compute_correlation_over_x(
            dfields, dpar, VMAX, DV, dx, 0.0, xlim=[x1, x2_end], ylim=[y1, y2], zlim=[z1, z2],
        )
        pr.disable()
    wall_s = time.perf_counter() - t0

    breakdown = _profile_breakdown(pr)
    return {
        'n_total': n_total,
        'n_slices_requested': n_slices_requested,
        'n_slices_actual': len(x_out),
        'wall_s': wall_s,
        **breakdown,
    }


def _run_in_subprocess(n_total, n_slices):
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--single-run', f'{n_total},{n_slices}'],
        capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
    )
    last_line = [l for l in result.stdout.strip().splitlines() if l.strip()][-1]
    return json.loads(last_line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-totals', type=str, default='1e6,1e7')
    parser.add_argument('--slice-counts', type=str, default='10,50,191,500')
    parser.add_argument('--single-run', type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run is not None:
        n_total_s, n_slices_s = args.single_run.split(',')
        result = _run_single(int(float(n_total_s)), int(float(n_slices_s)))
        print(json.dumps(result))
        return

    n_totals = [int(float(s)) for s in args.n_totals.split(',')]
    slice_counts = [int(float(s)) for s in args.slice_counts.split(',')]

    for n_total in n_totals:
        print(f"\n=== n_total={n_total} (sweeping {SWEEP_WIDTH_FRAC*100:.0f}% of domain width) ===")
        print(f"{'slices(req/actual)':>20} | {'wall_s':>8} | {'filter_s':>9} | {'jit_s':>8} | "
              f"{'hist_s':>8} | {'other_s':>8} | {'filter_%':>9}")
        print('-' * 90)
        for n_slices in slice_counts:
            print(f"  benchmarking n_total={n_total} slices={n_slices}...", file=sys.stderr)
            r = _run_in_subprocess(n_total, n_slices)
            filter_pct = 100.0 * r['filter_overhead'] / r['total'] if r['total'] > 0 else float('nan')
            print(f"{str(r['n_slices_requested'])+'/'+str(r['n_slices_actual']):>20} | "
                  f"{r['wall_s']:>8.3f} | {r['filter_overhead']:>9.3f} | {r['jit_loop']:>8.3f} | "
                  f"{r['histogram']:>8.3f} | {r['other']:>8.3f} | {filter_pct:>8.1f}%")


if __name__ == '__main__':
    main()
