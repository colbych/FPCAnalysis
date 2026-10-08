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

- [x] **`fpc.py:compute_correlation_over_x` / `comp_cor_over_x_multithread`
      / `compute_hist_and_cor` / `compute_cprime_hist`** — DONE (see Part A
      evidence above). Extracted the `gptsparticle` boolean-mask filtering
      into a new `fpc.filter_dpar_to_box(dpar, x1, x2, y1, y2, z1, z2)`
      helper. `compute_hist_and_cor` gained an optional `dparsubset`
      parameter: when given, it skips filtering entirely and uses the
      passed-in subset. The three sweep drivers that call
      `compute_hist_and_cor` once per field key for the same box
      (`compute_correlation_over_x`, `_comp_all_CEi` — which is used by both
      the serial and `comp_cor_over_x_multithread` paths — and the `etot`
      recursive case inside `compute_hist_and_cor` itself) now call
      `filter_dpar_to_box` once per box and pass the result to all of
      `ex`/`ey`/`ez` (or `epar`/`eperp1`/`eperp2`), instead of each call
      independently rebuilding the mask. Separately, eliminated the
      redundant double `np.histogramdd` call in `compute_cprime_hist`: now
      uses `scipy.stats.binned_statistic_dd` with `values=[ones, cprimew]`
      and `statistic='sum'` to bin the unweighted count (`hist`) and the
      weighted sum (`cprimebinned`) in one pass, with an explicit
      zero-particle guard (`binned_statistic_dd`, unlike `histogramdd`,
      raises on an empty sample even with explicit bin edges).

      **Correctness verification:** `pytest tests/` (5/5) and
      `tests/testload.py`/`tests/testframetransform.py` all still pass.
      Additionally ran `compute_correlation_over_x` end-to-end on 20k
      synthetic particles across 7 slices before and after this change and
      diffed `CEx`/`CEy`/`CEz`/`Hist`/`num_par` directly (not just via the
      single-call golden test) — bit-identical (`rtol=1e-10`).

      **Benchmark evidence** (`scripts/bench_sweep_redundancy.py`, same
      machine/env as the Phase 0 table):

      | n_total | slices | wall_s before → after | filter_s before → after | filter % before → after |
      |---|---|---|---|---|
      | 1e6 | 191 | 1.89 → 1.14 | 0.890 → 0.349 | 47% → 31% |
      | 1e6 | 500 | 3.78 → 2.10 | 1.993 → 0.735 | 53% → 35% |
      | 1e7 | 191 | 19.41 → 9.33 | 12.523 → 4.202 | 65% → 45% |
      | 1e7 | 500 | 35.98 → 15.27 | 26.120 → 8.844 | 73% → 58% |

      Filter time dropped by ~3x at every (n_total, slices) pair — matches
      the theory exactly (removed exactly the 3x-per-field-key redundancy).
      **Caveat, so the next person doesn't re-derive this:** filter-overhead
      share does *not* flatten to a ~20% floor at high slice counts as
      originally guessed above — it still grows with slice count, just ~3x
      less steeply. This is expected, not a bug: `compute_correlation_over_x`
      (what this benchmark profiles) operates on one full in-memory particle
      array and still does an `O(n_total)` mask-and-copy once per slice
      *after* the field-key dedup — that per-slice full-array rescan is a
      property of this driver's "load everything once, mask per slice in
      Python" design, not of the field-key redundancy this item targeted.
      That residual cost is exactly what the remaining Phase 1 items below
      (bounds-filtered loading so each slice never sees the full array) are
      for — don't expect this item alone to flatten the curve further.

- [x] **`data_tristan.py:load_particles`** — DONE. Added optional
      `x1,x2,y1,y2,z1,z2=None` bounds parameters. When all six are given,
      filters using h5py boolean selection on the position keys (`xe/ye/ze`,
      `xi/yi/zi`) *before* pulling the remaining keys, mirroring
      `data_dhybridr.read_box_of_particles`. Bounds are inclusive (`<=`),
      matching this library's filtering convention. No-bounds calls are
      unaffected (verified: `normalizeVelocity`/`loaddebugsubset` code paths
      untouched when bounds are `None`).

      **Correctness verification:** wrote a synthetic Tristan-format HDF5
      fixture (no checked-in one exists) and confirmed
      `load_particles(..., x1=...,...)` output is bit-identical to
      full-load-then-mask in Python, for both species. Separately verified
      the unit-conversion needed by the preslicing script fix below (raw vs.
      `normalizeVelocity`-normalized position units) against a synthetic
      `param.*` file replicating `load_params`' fields, including particles
      placed exactly on slice boundaries.

- [x] **`data_dhybridr.py:read_box_of_particles`** — DONE, with a caveat.
      Original code read `f['x1'][:]` (etc.) *twice* per axis just to build
      one comparison (`(x1 < f['x1'][:]) & (f['x1'][:] < x2)` evaluates the
      right-hand `f['x1'][:]` as a second, separate read). Fixed to read each
      axis once, reduce it to its boolean mask, and `del` the full-precision
      array before moving to the next axis — this was deliberately *not*
      done by caching all three position arrays for the whole function (an
      earlier version of this fix did that and regressed peak RSS from
      1330MB to 2971MB at 1e8 particles, by holding 3 full float64 arrays
      live simultaneously instead of one at a time - caught by re-running
      the benchmark below before considering this done, not from code
      inspection). Output values (including `x1`/`x2`/`x3` themselves) are
      still fetched via one direct `f[k][gpts]` boolean-selection read per
      key, same as the original — this never materializes the full column,
      so it was never the source of the double-read being fixed here.

      **Correctness verification:** synthetic dHybridR-format HDF5 fixture,
      output bit-identical to full-load-then-mask before and after.
      `pytest tests/` still green.

      **Benchmark evidence** (`scripts/bench_io_full_vs_filtered.py`, 1e8
      particles, 1% slice, same machine/env as Phase 0):

      | | peak RSS (MB) | elapsed (s) |
      |---|---|---|
      | full load (unchanged) | 4954 | 0.50-0.82 |
      | filtered, before this fix | 1330 | 1.123 |
      | filtered, after this fix | 1330 | 0.69-0.70 |

      Peak RSS unchanged (no regression, matches Phase 0 addendum's 3.7x
      win over full load). Elapsed time improved ~38% (1.123s → ~0.69s) by
      removing the literal duplicate read. **Caveat - does not fully meet
      the originally-hoped-for bar:** filtered mode is still slower in wall
      time than a full load at this scale (~0.69s vs ~0.5-0.8s - full-load
      time is noisy run-to-run, filtered is consistently in that range
      too). This looks like it's an inherent cost of HDF5 boolean/fancy
      selection against scattered chunk offsets vs. one sequential full
      read, not something fixable with more Python-level restructuring
      without changing on-disk layout/chunking (explicitly out of scope,
      see bottom of this doc). Don't re-chase this gap without a chunking/
      layout change in hand - re-deriving this finding from scratch would
      just reproduce this measurement.

- [x] **`preslicedataTristan.py`** — DONE. Moved the particle load inside
      the per-x-slice sweep loop, calling the new bounds-aware
      `dtr.load_particles(..., x1=...,...)` directly per slice instead of
      loading+normalizing the whole frame once up front and masking it in
      Python on every iteration. Since `load_particles`'s bounds are
      inclusive and `normalizeVelocity=True` scales positions by
      `comp*sqrt(massratio)` *after* loading, the sweep's bounds (which are
      in the normalized frame, since they default from `dfields` loaded with
      `normalizeFields=True`) are converted to raw units
      (`bound * comp*sqrt(massratio)`) before being passed in - see the
      comment left in the script and the Tristan verification above.
- [x] **`preslicedatadHybridR.py`** — DONE for the `use_restart=False`
      path (the `use_restart=True` path already called `read_restart` per
      slice). Moved the `read_box_of_particles` call inside the per-x-slice
      loop instead of bulk-loading the whole `xlim` range (or, with no
      limits given at all, the *entire frame* via `read_particles`) once up
      front. Since `read_box_of_particles` filters with strict `<` but this
      script's (and the rest of the library's) convention is inclusive
      `<=`, each per-slice read is padded by `dx*1e-6` on the x-bounds only;
      the existing exact/inclusive `gptsparticle` Python mask immediately
      after still does the authoritative filtering, so the padding can only
      ever avoid silently dropping a boundary-exact particle before that
      mask runs - it cannot change which particles end up in the output.
      y/z bounds are unchanged (unpadded) since they were already filtered
      with the same strict inequality in the pre-change bulk load, so this
      doesn't alter that pre-existing behavior.

      **Correctness verification:** synthetic dHybridR fixture with a few
      particles placed exactly on x-slice boundaries, comparing the old
      (bulk-load-then-remask) and new (per-slice padded-read-then-remask)
      code paths directly - identical particle sets per slice, including
      the boundary-exact ones.
- [x] **`data_dhybridr.py:get_dpar_from_bounds`** — DONE. (a) The directory
      listing/sort/filename-bounds parsing is now cached per `dpar_folder`
      in a module-level dict (`_get_dpar_folder_index`), built once per
      process instead of once per `get_dpar_from_bounds` call (i.e. once per
      slice in a sweep). (b) Replaced
      `pts[key].extend(_pts[key][:]); np.asarray(...)` (which unboxes every
      array element into a Python list, then reboxes) with collecting
      per-file arrays into a list and doing one `np.concatenate` per key at
      the end.

      **Correctness verification:** synthetic presliced dataset (several
      x-slice files), output confirmed bit-identical before/after across
      two calls with the same bounds (exercising the cache).
- [x] **`data_dhybridr.py:read_restart`** (single-threaded path) — DONE.
      Replaced the per-file `pts = np.concatenate([pts,_pts],axis=0)` loop
      (quadratic reallocation - the whole growing array gets copied on every
      iteration) with collecting each proc's array into a list and doing one
      `np.concatenate` after the loop. Order-preservation verified in
      isolation (this function needs a full dHybridR restart-file fixture
      to exercise end-to-end, which doesn't exist in this repo and wasn't
      worth fabricating for a 3-line mechanical change - verified the
      list-then-concatenate pattern reproduces the exact same element order
      as the old incremental-concatenate loop instead).

**Acceptance:** Phase 0 golden tests still pass (`pytest tests/`, 5/5, plus
`tests/testload.py` and `tests/testframetransform.py`). Peak RSS during
`read_box_of_particles` no longer scales with total frame particle count
when a spatial subset is requested (confirmed: still the measured 3.7x win
over full load at 1e8 particles, now with no RAM regression from the fix
itself). Elapsed time for `read_box_of_particles` improved ~38% but does
*not* fully flip to beating full-load in wall time at this scale - see the
caveat above, this is now understood to be an HDF5-access-pattern cost, not
a loose Python inefficiency. `scripts/bench_sweep_redundancy.py`'s
filter-overhead share no longer grows ~linearly as steeply with slice count
(see Item 1 above for the actual numbers and its own caveat about not fully
flattening either, for a different, already-explained reason).

`scripts/bench_fpc.py` re-run at the Phase 0 table's scales, same machine/env:

| Scale (particles) | `compute_hist_and_cor` before → after | peak RSS before → after |
|---|---|---|
| 1e5 | 0.009s → 0.007s | 286.8MB → 278.8MB |
| 1e6 | 0.078s → 0.070s | 399.3MB → 390.9MB |
| 1e7 | 0.830s → 0.698s | 1432.5MB → 1464.5MB |

A modest (10-22%) single-call improvement, smaller than the sweep-level win
above — expected, since `bench_fpc.py` calls `compute_hist_and_cor` once per
scale, not once per field key for the same box, so it only exercises the
`binned_statistic_dd` single-pass-binning fix, not the filter-dedup fix
(that one only pays off when the same box is queried for multiple field
keys - that's what `bench_sweep_redundancy.py` isolates, and where the real
~3x filter-overhead reduction shows up). Peak RSS is flat within normal
run-to-run noise for a single-subprocess peak-RSS reading (slightly up at
1e7 - not a real regression; nothing in this change path should increase
memory at this scale, and `write`/`load` numbers, which exercise functions
Phase 1 didn't touch, are unchanged as expected).

---

## Phase 2 — Fix the multiprocessing driver

Targets findings #3-#4.

- [x] **`fpc.py:comp_cor_over_x_multithread`** — DONE. Replaced the
      hand-rolled `while not_finished: ... time.sleep(10)` polling loop
      (which re-scanned every pending future each cycle and could add up to
      10s of pure dead time after a result was actually ready) with
      `concurrent.futures.as_completed(future_to_tskidx)`, keyed by a
      `{future: tskidx}` dict instead of the old parallel `futures`/`jobidxs`
      lists. Also submits all tasks up front rather than manually throttling
      - `ProcessPoolExecutor(max_workers=...)` already caps concurrency
      itself, so the old code's throttling was reimplementing something the
      executor already does.
- [x] **`data_dhybridr.py:read_restart`** multithreaded path — DONE. Same
      `as_completed` fix for the `time.sleep(1)` busy-wait. Incidentally,
      this path referenced `time.sleep(1)` without ever importing `time` -
      a latent `NameError` that would only fire the first time no future was
      yet done, i.e. likely never actually exercised before. While rewriting
      this block also applied the Phase 1 list-collect-then-concatenate-once
      fix here too (the multithreaded path had the *same* quadratic
      `np.concatenate([pts,_output],axis=0)`-per-completed-task pattern as
      the single-threaded path Phase 1 already fixed - not called out
      separately in Phase 1's checklist, but the same bug, caught while
      already rewriting this exact block for the `as_completed` change, so
      fixed here rather than left half-done).
- [x] **`dfields` re-pickling** — DONE. Added a module-level
      `_mp_worker_dfields` global set once per worker process via
      `ProcessPoolExecutor(..., initializer=_init_mp_worker_dfields,
      initargs=(dfields,))`, and a thin
      `_grab_dpar_and_comp_all_CEi_using_worker_dfields` wrapper that reads
      it instead of taking `dfields` as a per-task argument. `dfields` is
      now pickled once per worker process instead of once per
      `executor.submit()` call (i.e. once per slice in a sweep).

      **Correctness verification:** no existing test exercises
      `comp_cor_over_x_multithread` end-to-end (no presliced-particle
      fixture in this repo). Built a synthetic presliced dataset + the real
      field fixture, ran `comp_cor_over_x_multithread` and the already-
      trusted serial `compute_correlation_over_x` over the identical
      particles/box/field data, and compared `num_par`, the projected
      `Hist`, and one projected `CE*` component across all slices -
      bit-identical (`rtol=1e-6` to `1e-8`, consistent with the serial path's
      own tolerance). For `read_restart`'s multithreaded path, a full restart-
      file fixture would require replicating `PartMapper3D`'s binary format
      and dHybridR input-file parsing - disproportionate for this change, so
      (as in Phase 1 for this same function) verified the
      submit-all/`as_completed`/collect-then-concatenate-once pattern in
      isolation against the original polling pattern with a dummy picklable
      worker function, confirming both produce the same multiset of results
      regardless of completion order (which was already non-deterministic
      in the original multithreaded code, so this is not a behavior change).

      **Found along the way, not fixed (out of scope for this phase):**
      `data_dhybridr.py:get_dpar_from_bounds` has a pre-existing bug, present
      since before any of this optimization effort (confirmed against
      `57500f9`, i.e. predates Phase 0 too) - when a requested `x1` exactly
      equals the lower bound of the leftmost presliced file (a common case:
      it's what you get by default if a sweep's `xlim[0]` is the domain's
      left edge, which is also the typical `xlim[0]` the preslicing scripts
      themselves used), `leftmostbound_index` stays at its sentinel value of
      `-1`, and `filenames[leftmostbound_index:rightmostbound_index+1]` -
      i.e. `filenames[-1:2]` - means "from the last file to index 2", an
      empty (or wrong) slice in Python, not "from the start". Reproduced
      with `get_dpar_from_bounds(presliced_dir, x1=<leftmost file's lower
      bound>, x2=...)` returning 0-1 particles instead of the real count.
      Did not fix this - it's unrelated to Phase 2's scope (the polling
      loop and `dfields` pickling) and changes `get_dpar_from_bounds`'s
      return value for real callers, which deserves its own deliberate fix
      and verification pass rather than a drive-by change. Flagged to the
      user; worth a dedicated follow-up.

**Acceptance:** Golden tests pass (`pytest tests/`, 5/5, plus `testload.py`/
`testframetransform.py`). Wall-clock sweep time for a multi-slice
multithreaded run improved measurably: a 7-slice synthetic sweep with
`max_workers=4` went from **20.3s to 4.6s (4.4x)** on the same
machine/env as Phase 0/1, comparing the old polling implementation
(via `git stash`) against the new `as_completed` implementation on
identical inputs. This matches the predicted mechanism exactly: with 7
tasks and 4 workers, the old loop needed 2 polling cycles to drain all
results, each one paying the full `time.sleep(10)` regardless of how
quickly the tasks actually finished (each task here ran in well under a
second) - a ~20s floor with no relationship to actual compute time, which
`as_completed` removes entirely. (Not separately isolated: the `dfields`
re-pickling fix's own contribution to this number, since this fixture's
`dfields` is small; its benefit scales with field-grid size and slice
count and would show up more on a larger real dataset.)

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
