# Optimization Plan: Scaling to Larger Particle Datasets

Status key: `[ ]` not started · `[~]` in progress · `[x]` done

Goal: handle 10-100x current particle counts without changing numerical
output, without a rewrite, and without abandoning the project's
readability-first philosophy (`docs/devnotes.md`). See the companion
optimization proposal (chat history / `CLAUDE.md` summary) for the full
findings this plan is based on.

**Start here if picking this up fresh:** read the "Phase 0 addendum" section
below before touching Phase 1. It has real profiling/benchmark evidence
(not just code-inspection reasoning) for why Phase 1's scope is what it is —
in particular, a plausible-looking objection ("your benchmark makes compute
look like the bottleneck, not I/O") turned out to be a real methodology gap,
and chasing it down changed Phase 1's scope. Don't skip straight to the
checklists without that context, or you'll likely re-raise the same question
and re-do the investigation.

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

## Phase 0 addendum — benchmarking evidence that revised Phase 1's scope — DONE

**Why this addendum exists:** after Phase 0 landed, a reasonable objection
was raised against the baseline table above: it makes `compute_hist_and_cor`
look like the dominant cost relative to particle load, which seemed to
contradict the original proposal's claim that full-frame loading (not
per-particle compute) is the real ceiling on dataset size. That objection was
correct to raise — the Phase 0 benchmark's `load` step reads back a file it
had just written containing *exactly* the particles the compute step then
uses, so it never exercises the "load everything, use a slice of it" pattern
that the proposal was actually about. Two follow-up benchmarks were built to
settle this with real decomposed numbers instead of one ambiguous table.

### Part A — `scripts/bench_sweep_redundancy.py`: is the redundant per-call filtering real?

`fpc.py` has a self-documented TODO (`fpc.py:210`, "lot of redundancy in this
library... compute Hist redundantly... dont compute subset each time for
CEx, CEy, CEz"). Concretely: `fpc.compute_correlation_over_x` calls
`compute_hist_and_cor` once per field key (`ex`,`ey`,`ez`) per slice, and each
call independently rebuilds the `gptsparticle` boolean mask over the
**entire** in-memory particle array — cost `O(n_total)` per call, not
`O(particles actually in the box)` — plus redundantly runs `np.histogramdd`
twice (once unweighted for `hist`, once weighted for `cprimebinned`).

Method: profiled (`cProfile`, aggregated automatically across repeated
calls) a real `fpc.compute_correlation_over_x` sweep over one large in-memory
synthetic particle set, at a few different slice counts for a fixed swept
width, isolating `compute_hist_and_cor`'s own (non-subcall) time as
"filter overhead", `compute_cprimew`'s time as "JIT loop", and
`histogramdd`/`searchsorted` time as "histogram". Each run executes in its
own subprocess for a clean profiler/JIT state.

| n_total | slices (requested) | wall (s) | filter overhead (s) | JIT loop (s) | filter % of total |
|---|---|---|---|---|---|
| 1e6 | 10 | 0.59 | 0.115 | 0.310 | 19% |
| 1e6 | 50 | 0.96 | 0.338 | 0.347 | 35% |
| 1e6 | 191 | 1.89 | 0.890 | 0.379 | 47% |
| 1e6 | 500 | 3.78 | 1.993 | 0.433 | 53% |
| 1e7 | 10 | 5.69 | 1.224 | 2.948 | 22% |
| 1e7 | 50 | 9.91 | 4.569 | 3.287 | 46% |
| 1e7 | 191 | 19.41 | 12.523 | 3.441 | 65% |
| 1e7 | 500 | 35.98 | 26.120 | 3.310 | 73% |

(191 slices was chosen because it's what `dx=0.5` over this fixture's domain
works out to — i.e. the repo's own `analysisinput.txt` example parameters,
not an arbitrary round number.)

**Finding, confirmed not just hypothesized:** JIT compute time stays roughly
flat as slice count grows at fixed `n_total` (expected — same total particles,
redistributed into more/smaller boxes). Filter overhead grows ~linearly with
slice count and, at a realistic slice count, is already **47-73% of total
sweep time** — a bigger current wall-clock cost than the per-particle JIT
loop in exactly the sweep pattern this library's own example config produces.

### Part B — `scripts/bench_io_full_vs_filtered.py`: is the full-load RAM ceiling real?

Compares the two loaders that *already exist* for dHybridR — no new
production code needed to get this evidence:
- `data_dhybridr.read_particles` (current default, always full-load)
- `data_dhybridr.read_box_of_particles` (already in the codebase, does a
  bounds-filtered h5py read)

against the same on-disk synthetic "frame" file, both asked for only a 1%
spatial slice of it. Each (scale, mode) pair runs in its own subprocess so
peak-RSS readings (`resource.getrusage`, a cumulative watermark) aren't
contaminated across comparisons.

| n_total | mode | peak RSS (MB) | elapsed (s) |
|---|---|---|---|
| 1e6 | full | 231.5 | 0.006 |
| 1e6 | filtered | 200.1 | 0.009 |
| 1e7 | full | 667.4 | 0.053 |
| 1e7 | filtered | 297.6 | 0.101 |
| 1e8 | full | 4954.4 | 0.823 |
| 1e8 | filtered | 1330.0 | 1.123 |

**Finding, confirmed:** at 1e8 particles, full-load peak RSS is **3.7x**
higher than the filtered load — real, and it's specifically the thing that
gates whether a dataset fits in RAM at all, independent of wall-clock
considerations. **Finding, unexpected — must inform Phase 1's implementation:**
the filtered loader is currently *slower* in wall time at every scale tested,
not faster. It trades time for memory today, it does not win on both axes.
Root cause: `read_box_of_particles` still reads the full `x1`/`x2`/`x3`
arrays into memory to build its boolean mask before doing h5py fancy-indexed
reads per key — this is the same inefficiency already called out in Phase
1's `get_dpar_from_bounds`/`read_box_of_particles` bullets below, now with a
measured cost attached to it.

### Conclusion — how this changes Phase 1's scope (see updated checklist below)

Both halves of the original proposal's reasoning hold up under evidence, but
the *priority and scope* needed revision:
1. The compute-redundancy fix (dedup `gptsparticle` + the double
   `histogramdd`) is a **new finding, surfaced only by this profiling** — the
   original proposal noted the `fpc.py:210` TODO comment existed but did not
   catalogue it as its own finding or give it a phase/task of its own.
   Evidence now says it belongs in Phase 1: it's a bigger current wall-clock
   cost than anything else identified so far, it's cheap and low-risk to
   fix, and it's orthogonal to the I/O work (different file, `fpc.py` vs
   `data_*.py`), so there's no reason to defer it. **Added to Phase 1 below
   as a new item.**
2. The I/O/RAM ceiling is real and still the thing that decides whether a
   10-100x larger dataset can be attempted at all — but "just redirect
   callers to the already-filtered loader" is not sufficient on its own, per
   the time-vs-memory tradeoff found above. Phase 1's `read_box_of_particles`
   fix needs to actually close that gap, not just inherit it.

---

## Phase 1 — Stop loading full frames before filtering, and stop redundant per-slice filtering (highest impact, evidence-backed)

Targets findings #1-#3 from the original proposal, **plus** the Part A
redundancy finding above (a new finding surfaced by benchmarking after Phase
0, not previously catalogued, added here because it dominates realistic
sweep wall-time more than anything else identified so far — see the Phase 0
addendum above). This phase now addresses both halves of "the actual
ceiling on dataset size today": peak RAM during load, and redundant
repeated work during a sweep.

- [ ] `[ ]` **`fpc.py:compute_correlation_over_x` / `comp_cor_over_x_multithread`
      / `compute_hist_and_cor` / `compute_cprime_hist`** (new item — see
      Part A evidence above): restructure so the
      `gptsparticle` boolean-mask filtering happens **once per slice**, not
      once per field-key call (`ex`,`ey`,`ez`, and again for `etot`/FAC
      variants) — e.g. have the sweep driver filter to the slice's particle
      subset once and pass that subset into each per-field-key call, rather
      than each call re-filtering the full array. Separately, eliminate the
      redundant double `np.histogramdd` call in `compute_cprime_hist`
      (compute the unweighted `hist` and weighted `cprimebinned` without
      binning twice — e.g. derive one from the other, or bin once with both
      a counts array and a weighted-sum array in a single pass). Re-run
      `scripts/bench_sweep_redundancy.py` after this change; filter-overhead
      share of total sweep time should drop back toward the ~20% floor seen
      at low slice counts, at any slice count.

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
      **Measured cost of not fixing this** (Part B, above): at 1e8 particles
      requesting a 1% slice, this loader already wins 3.7x on peak RSS
      (1330MB vs 4954MB) but is *slower* in wall time than a full load
      (1.123s vs 0.823s) — the fix above should close that time gap, not
      just preserve the RAM win. Re-run `scripts/bench_io_full_vs_filtered.py`
      after this fix; filtered mode should win on both RSS and elapsed time.
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

**Acceptance:** Phase 0 golden tests still pass (`pytest tests/`). Peak RSS
during preslicing/loading no longer scales with total frame particle count
when a spatial subset is requested — scales with subset size instead:
re-run `scripts/bench_io_full_vs_filtered.py` and confirm the filtered mode
wins on *both* peak RSS and elapsed time (not just RSS, per the gap found
above). Re-run `scripts/bench_sweep_redundancy.py` and confirm filter-overhead
share of total sweep time no longer grows with slice count. Re-run
`scripts/bench_fpc.py` at the same scales as the Phase 0 table and record
deltas.

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
