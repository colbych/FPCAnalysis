# Optimization Plan: Scaling to Larger Particle Datasets

Status key: `[ ]` not started · `[~]` in progress · `[x]` done

Goal: handle 10-100x current particle counts without changing numerical
output, without a rewrite, and without abandoning the project's
readability-first philosophy (`docs/devnotes.md`). See the companion
optimization proposal (chat history / `CLAUDE.md` summary) for the full
findings this plan is based on.

**Ground rule for every phase below:** no phase is "done" until its diff has
been checked against the Phase 0 regression baseline with matching output
(within float tolerance) on `tests/testdata/dHybridR/M06_th45/`.

---

## Phase 0 — Build a regression safety net (prerequisite for everything else) — DONE

There was no automated test that checked FPC numerical output — only
loader-shape tests (`tests/testload.py`, `tests/testframetransform.py`) and
manual notebook re-runs. Landed before any hot-path change, as planned.

Environment note: this machine had no conda and no matching Python version.
Set up via Homebrew Miniforge + the project's own `./install.py`, which
created `FPCAnalysisenv/` (Python 3.11.17, all pinned deps). Use
`FPCAnalysisenv/bin/python` for everything below until the env is activated
in a shell.

- [x] Added `tests/_synthetic.py`: deterministic (fixed-seed) synthetic
      particle generator. **Why synthetic, not real data:** the checked-in
      fixture `tests/testdata/dHybridR/M06_th45/` only contains field data
      (`Output/Fields/...`), no particle data (no `Output/Raw/Sp01/...`) — so
      there was no real particle fixture to regress against. Synthetic
      particles exercise the real interpolation/histogram/correlation code
      path end to end; the numbers aren't physically meaningful, but they
      must stay bit-identical across any change that claims not to affect
      output.
- [x] Added `tests/generate_fpc_golden.py` (manual, not pytest-collected on
      purpose) and the golden baseline `tests/golden/fpc_regression_golden.npz`,
      covering 5 cases: `ex`/`ey`/`ez` with `useBoxFAC=True`, plus `epar` with
      both `useBoxFAC=True` and `useBoxFAC=False` (covers both
      `change_velocity_basis` and `change_velocity_basis_local` code paths).
- [x] Added `tests/test_fpc_regression.py`, 5 tests comparing live output
      against the golden baseline via `np.testing.assert_allclose`. All 5
      pass against current `main`. **Pytest discovery gap found along the
      way:** `tests/testload.py` / `tests/testframetransform.py` are NOT
      discovered by plain `pytest tests/` (pytest's default pattern is
      `test_*.py`/`*_test.py`; `testload.py` has no underscore so it doesn't
      match) — they only ever ran via direct `python tests/testload.py`
      invocation per `docs/regression_unit_test.md`. Left as-is (out of
      scope / a pre-existing gap, not introduced by this effort) but run
      both forms when validating a change: `pytest tests/` AND
      `python tests/testload.py && python tests/testframetransform.py`.
- [x] Added `scripts/bench_fpc.py`: benchmarks particle write/load
      (`data_dhybridr.write_particles_to_hdf5` / `read_particles` — the
      function Phase 1 targets) and `fpc.compute_hist_and_cor`, at a few
      synthetic particle-count scales, with a numba-JIT warm-up call so
      timings reflect steady state, not one-time compilation. Each scale
      runs in its own subprocess so peak-RSS readings aren't contaminated by
      `resource.getrusage`'s cumulative, monotonically-increasing watermark
      across scales run in one process.

**Caveat on what this baseline does and doesn't show:** `read_particles` in
this benchmark reads back a whole (synthetic, single-slice) file it just
wrote — it shows load cost scaling with total particle count, which is the
right trend to track, but it does *not* yet demonstrate the Phase 1 win
(loading a small spatial subset cheaply from a large file). Re-run this same
script after Phase 1 lands a bounds-filtered loader, with a new case that
loads a small fraction of a large file, to show the actual improvement.

**Acceptance:** met. `pytest tests/` passes (5/5), golden-output test is
green against current `main`, baseline numbers recorded below (2026-10-07,
Apple Silicon, macOS, Python 3.11.17, numba 0.60.0, numpy 1.26.4).

| Scale (particles) | write (s) | load (s) | peak RSS (MB) | `compute_hist_and_cor` (s, post-JIT-warmup) |
|---|---|---|---|---|
| 1e5 | 0.001 | 0.006 | 286.8 | 0.009 |
| 1e6 | 0.004 | 0.013 | 399.3 | 0.078 |
| 1e7 | 0.049 | 0.048 | 1432.5 | 0.830 |

Reproduce with: `FPCAnalysisenv/bin/python scripts/bench_fpc.py --scales 1e5,1e6,1e7`

---

## Phase 1 — Stop loading full frames before filtering (highest impact)

Targets findings #1-#3 from the proposal. This is the actual ceiling on
dataset size today.

- [ ] `[ ]` **`data_tristan.py:load_particles`**: add an optional spatial
      bounds parameter (`x1,x2,y1,y2,z1,z2=None`) that, when given, filters
      using h5py hyperslab/boolean selection on the position keys *before*
      pulling other keys, mirroring `data_dhybridr.read_box_of_particles`'s
      approach. Keep the no-bounds call signature working identically
      (default `None` = current full-load behavior) so existing callers are
      unaffected.
- [ ] `[ ]` **`data_dhybridr.py:read_box_of_particles`**: avoid reading
      `x1`/`x2`/`x3` twice (once to build the mask, again implicitly via
      fancy indexing) — read position arrays once into local numpy arrays,
      build the combined mask, then index each dataset once per key.
- [ ] `[ ]` **`preslicedataTristan.py` / `preslicedataTristan.py`-equivalent
      for dHybridR**: switch the preslicing scripts to use the new bounds-
      filtered loader per x-slice instead of loading the whole frame once and
      slicing in-memory in a Python `while` loop. This turns peak RAM from
      `O(total particles)` into `O(particles in widest single slice)`.
      (dHybridR's `preslicedatadHybridR.py` already loads via
      `read_box_of_particles` in some code paths — check `use_restart`/`xlim`
      branches and make the behavior consistent across both.)
- [ ] `[ ]` **`data_dhybridr.py:get_dpar_from_bounds`**: (a) replace the
      per-call `os.listdir` + sort + filename parse with a cached index built
      once per process (e.g. memoize on `dpar_folder`, or write/read a small
      index file alongside the preslice output written by the preslicing
      script); (b) replace `pts[key].extend(_pts[key][:]); np.asarray(...)`
      with collecting arrays in a list and doing one `np.concatenate` at the
      end.
- [ ] `[ ]` **`data_dhybridr.py:read_restart`** (single-threaded path, lines
      ~704-711): replace the per-file `np.concatenate([pts, _pts], axis=0)`
      loop (quadratic reallocation) with collecting into a list and doing one
      `np.concatenate` after the loop.

**Acceptance:** Phase 0 golden tests still pass. Benchmark script shows peak
RSS during preslicing no longer scales with total frame particle count when a
spatial subset is requested — scales with subset size instead. Re-run the
Phase 0 benchmark table at the same scales and record deltas.

---

## Phase 2 — Fix the multiprocessing driver

Targets findings #3-#4.

- [ ] `[ ]` **`fpc.py:comp_cor_over_x_multithread`**: replace the hand-rolled
      `while not_finished: ... time.sleep(10)` polling loop with
      `concurrent.futures.as_completed(futures)`.
- [ ] `[ ]` **`data_dhybridr.py:read_restart`** multithreaded path: same fix,
      replace `time.sleep(1)` busy-wait with `as_completed`.
- [ ] `[ ]` Confirm `dfields` isn't being needlessly re-pickled per task if
      it doesn't have to be — check whether `ProcessPoolExecutor`'s
      `initializer`/`initargs` (load `dfields` once per worker process) is a
      better fit than passing it as a per-task argument, given it's the same
      object across all tasks in a sweep.

**Acceptance:** Golden tests pass. Benchmark script shows wall-clock sweep
time for a multi-slice run improves measurably (record in table) and CPU
utilization across workers is more even during the run (spot-check with
`top`/`htop` during a bench run, not a hard automated gate).

---

## Phase 3 — Vectorize the remaining Python-loop antipatterns

Targets findings #6, #7, #9. These are safe, independent, individually
testable against Phase 0 golden output — do them in any order, one PR each.

- [ ] `[ ]` Factor the triple-nested Python loop that builds 3D `vx`/`vy`/`vz`
      grids (duplicated in `fpc.py:compute_cprime_hist`, `fpc.py:compute_hist`,
      and the debug branch of `fpc.py:compute_hist_and_cor`) into one helper
      using `np.meshgrid(vx, vy, vz, indexing='ij')` — confirm axis order
      matches the existing `[k][j][i]` convention exactly before replacing
      (write a quick equivalence check against the old loop for one case,
      then delete the three duplicated loops in favor of the helper).
- [ ] `[ ]` **`array_ops.py:array_3d_to_2d`**: replace
      `np.apply_along_axis(np.sum, axis, arr3d)` with `np.sum(arr3d, axis=axis)`.
      This is called 12x per slice from `fpc.py:project_CEi_hist` in the
      multiprocessing hot path.
- [ ] `[ ]` **`analysis.py:split_by_init_speed`**: replace the per-particle
      Python string-concat unique-ID construction with a vectorized approach
      (e.g. combine `indi`/`proci` via a vectorized numeric encoding, or use
      `np.core.records`/structured arrays + `np.isin`/`np.searchsorted`
      instead of building then linear-scanning IDs). Replace the
      `np.array([dpar[pkey][_i] for _i in newmainindexes])` loops with direct
      fancy indexing `dpar[pkey][newmainindexes]`.

**Acceptance:** Golden tests pass. Benchmark script shows measurable
improvement at 1e6-1e7 scale for any workflow exercising
`project_CEi_hist`/`split_by_init_speed`.

---

## Phase 4 — Parallelize the per-particle JIT kernel (optional, do last)

Targets finding #11. Only worth doing once Phases 1-3 land, since I/O
dominates until then.

- [ ] `[ ]` **`fpc.py:compute_cprimew`**: add `parallel=True` to its
      `@jit(nopython=True)` decorator and convert the outer loop to
      `numba.prange`. Verify the loop body has no cross-iteration
      dependencies (it doesn't appear to — each `cprimew[i]` write is
      independent) before flipping this on.
- [ ] `[ ]` Re-benchmark; numba parallel loops have thread-pool startup
      overhead, so confirm this is a net win at realistic particle-per-slice
      counts and not just at the largest synthetic benchmark size — report
      both.

**Acceptance:** Golden tests pass. Benchmark table shows improvement at the
particle-per-slice counts realistic for this codebase's actual use (not just
the largest synthetic case).

---

## Explicitly out of scope for this effort

- Rewriting the preslice-file-per-x-slice architecture itself (e.g. moving to
  a single indexed HDF5/Parquet store) — a bigger architectural change than
  "optimize for larger datasets," flag separately if I/O is still the
  bottleneck after Phase 1-2.
- `data_gkeyll.py`'s nested-loop grid construction — operates on pre-binned
  distributions, doesn't scale with particle count, not this effort's target.
- Field-grid-sized `deepcopy` calls in `analysis.py`'s FAC conversion
  functions and `array_ops.py`'s subset/truncate/avg helpers — real but
  secondary; revisit only if profiling after Phase 1-3 still shows them as
  hot.
- Adding new dependencies (dask, GPU libraries) — not needed for the
  identified bottlenecks; would be a deliberate separate decision, not a
  default part of this plan.
