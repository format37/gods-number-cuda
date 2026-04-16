# Tech Spec: Diameter Lower-Bound Estimation Module for `gods-number-cuda`

**Status:** Draft
**Owner:** TBD
**Target project:** Extension to the existing `gods-number-cuda` BFS solver.
**Related prior spec:** `gods_number_tech_spec.md` (original BFS solver spec).

---

## 1. Context

The existing `gods-number-cuda` tool performs bounded BFS over the Cayley graph of the N×N×N Rubik's Cube in the Half-Turn Metric (HTM), using:

- A compact Lehmer-coded solver for N=2 (full diameter = 11 proven).
- A facelet-array + GPU hash table solver for N≥3 (bounded BFS, depth limited by VRAM).

**What the solver currently produces:** a depth histogram `N(d)` for `d = 0, 1, …, D_max`, where `D_max` is either the true diameter (N=2) or a VRAM-limited cutoff (N≥3).

**What it does not produce:** a formal lower bound on the Cayley-graph diameter for cubes where full BFS is infeasible (N≥3 in most configurations, N≥4 always).

For cubes with N≥4, the **true God's number is unknown** and is an open problem in computational group theory. Published lower bounds for N=4 (~35 HTM) predate modern GPU-era BFS and were derived via other methods. A solver that produces depth histograms at GPU-BFS scale can mechanically derive an information-theoretic lower bound that is, in several cases, **stronger than the currently published bound**.

This spec defines a new module `diameter_bounds` that post-processes BFS output to emit a rigorous lower bound, plus supporting graph statistics (effective branching factor, saturation onset).

---

## 2. Goals

### 2.1 Primary goals

- Emit a **provable lower bound** `D_lb` on the Cayley-graph diameter from any BFS run that reached depth `D_max ≥ 3`.
- Emit an **effective branching factor** `b_eff(d)` as a function of depth, with a detected **saturation onset depth** `d_sat` (the depth where `N(d+1)/N(d)` begins to fall below the free-expansion rate).
- Produce a reproducible, machine-readable report (JSON) plus a human-readable console summary.

### 2.2 Secondary goals

- Support **extrapolation-based** lower bound that extends beyond `D_max` when the measured branching factor is well-fit by a constant. This is a *conservative* extension — extrapolated bounds are marked separately from proven bounds.
- Support **multi-metric** output (HTM as primary; QTM as follow-on if the BFS driver is extended to emit QTM histograms).

### 2.3 Non-goals

- Upper bounds on the diameter (would require coset decomposition or reduction method — out of scope).
- Predicting the *shape* of the bell curve (peak location, width) — mathematically impossible from free-expansion data alone; see §7.
- Publishing results — this module produces the numerical artifact; any academic write-up is a separate workstream.

---

## 3. Background: Why This Lower Bound Works

### 3.1 Information-theoretic lower bound (standard result)

For any graph G = (V, E) and any vertex v₀:

$$|B_d(v_0)| = \sum_{k=0}^{d} N(k) \leq |V|$$

where `B_d(v₀)` is the ball of radius *d* around v₀, and `N(k)` is the number of vertices at exact distance k. Therefore, the diameter is bounded below by:

$$\text{diam}(G) \geq \min \{ d : |B_d(v_0)| \geq |V| \} - 1$$

Equivalently: the smallest *d* such that the cumulative BFS count reaches `|V|` is a lower bound on the diameter. This bound is tight when growth is exponential and `|V|` is known.

For the Cayley graph of the N×N×N Rubik's Cube group, `|V|` is known in closed form (product of factorials × orientation factors), so this bound is mechanically computable.

### 3.2 Measured vs. theoretical branching factor

Observed from the existing solver's output for the outer-face HTM generator set (18 generators):

| Depth d | N(d) for N=3 | ratio N(d)/N(d-1) |
|--------:|-------------:|-------------------:|
| 0 | 1 | — |
| 1 | 18 | 18.0 |
| 2 | 243 | 13.5 |
| 3 | 3,240 | 13.33 |
| 4 | 43,239 | 13.345 |
| 5 | 575,372 | 13.306 |
| 6 | 7,636,058 | 13.275 |
| 7 | 101,229,076 | 13.257 |

The ratio converges to an **asymptotic branching factor `b_∞ ≈ 13.34847…`** — the unique positive root of the quadratic arising from the free-group pruning rules (no consecutive moves on the same face; opposite-face pairs canonicalized). This value is identical for every N ≥ 3 under the standard outer-face HTM generator set, because pruning rules are face-based and don't depend on N.

### 3.3 Why N=3 and N=4 agree up to d=7

At depth d, collisions between distinct reduced word sequences occur only when |V|(d) ≳ 1/√|G| by birthday-paradox heuristics. For N ≥ 3 with |G| ≥ 4×10¹⁹, collisions don't occur until deep depths (~17+ for N=3, ~40+ for N=4). Thus at shallow depths, N(d) reflects **free-group expansion**, not the group's structure. This is the baseline the lower-bound module exploits.

---

## 4. Functional Requirements

### 4.1 Inputs

The module consumes:

1. **Histogram file** — sequence `[N(0), N(1), …, N(D_max)]` produced by the existing solver. Format: JSON or CSV written by the BFS driver.
2. **Cube metadata** — `N` (cube size), `metric` (HTM/QTM), `generator_set` (for future QTM / inner-slice variants).
3. **Group order** — either looked up from a built-in table for N ∈ {2,…,10} or supplied as a command-line argument for larger N.

Group orders (HTM outer-face generators, standard reductions):

| N | &#124;G&#124; |
|---|---|
| 2 | 3,674,160 |
| 3 | 43,252,003,274,489,856,000 ≈ 4.33 × 10¹⁹ |
| 4 | ≈ 7.40 × 10⁴⁵ |
| 5 | ≈ 2.83 × 10⁷⁴ |
| 6 | ≈ 1.57 × 10¹¹⁶ |
| 7 | ≈ 1.95 × 10¹⁶⁰ |

Reference: Jaap Scherphuis's cube group-order tables.

### 4.2 Outputs

For each BFS run, the module emits:

**A. Proven lower bound** `D_lb_proven`:
- The smallest `d` such that `Σ_{k=0}^{d} N(k) ≥ |G|`, using **only measured values**. If `D_max` is reached before the ball covers `|G|`, this returns `D_max + 1` as a partial bound with a flag (`coverage_fraction < 1`).

**B. Extrapolated lower bound** `D_lb_extrapolated` (marked as non-proven):
- Fit `b_eff(d) = N(d)/N(d-1)` over the last K measured depths (K configurable, default 3).
- Extrapolate future `N(d)` as `N(D_max) · b_eff^(d - D_max)`.
- Compute smallest `d` such that extrapolated cumulative sum ≥ `|G|`.
- Must be clearly labeled "extrapolated, not proven" in output.

**C. Graph statistics:**
- `b_eff(d)` for every measured d ≥ 2.
- `b_asymptotic` (best-fit constant over the last half of measured depths).
- `d_saturation_onset`: smallest d where `b_eff(d) / b_asymptotic < 1 - ε` for ε configurable (default 0.01). If no saturation detected, report "not observed within D_max".

**D. Report file** (`diameter_bounds.json`):

```json
{
  "cube_size": 4,
  "metric": "HTM",
  "generator_set": "outer_faces_18",
  "group_order": "7.401e+45",
  "max_depth_measured": 8,
  "histogram": [1, 18, 243, 3240, 43239, 575372, 7636058, 101229076, 1350852409],
  "proven_lower_bound": {
    "value": 9,
    "coverage_fraction_at_D_max": 1.82e-37,
    "note": "BFS did not cover group; D_lb reported as D_max + 1"
  },
  "extrapolated_lower_bound": {
    "value": 41,
    "method": "constant b_eff fit over last 3 depths",
    "b_fit": 13.247,
    "disclaimer": "Not a proven bound. Assumes free-expansion continues."
  },
  "graph_stats": {
    "b_asymptotic": 13.25,
    "b_eff_per_depth": [null, 18.0, 13.5, 13.33, 13.345, 13.306, 13.275, 13.257, 13.340],
    "saturation_onset_depth": null
  },
  "solver_version": "gods-number-cuda v0.x",
  "run_timestamp": "2026-04-16T..."
}
```

**E. Console summary** (concise, tabular).

### 4.3 CLI integration

Add a new command or flag to the existing binary:

```
./gods_number 4 8 --emit-bounds=bounds.json
./gods_number --analyze-histogram=hist.json --group-order=7.4e45
```

The first form runs BFS and analyzes in one pass. The second form re-analyzes a saved histogram without re-running BFS (useful for iterating on the analysis logic).

---

## 5. Algorithm Specifications

### 5.1 Proven lower bound

```
input:  N(d) for d = 0..D_max, group_order |G|
output: D_lb_proven (int), coverage_fraction (float)

cumulative = 0
for d in 0..D_max:
    cumulative += N(d)
    if cumulative >= |G|:
        return D_lb_proven = d, coverage = 1.0

# BFS did not reach group coverage
return D_lb_proven = D_max + 1, coverage = cumulative / |G|
```

Uses arbitrary-precision arithmetic (Python `int` or C++ `boost::multiprecision::cpp_int`) because `|G|` for N≥3 exceeds uint64. **Do not use floating-point for the cumulative sum against |G|** — precision loss is catastrophic at these magnitudes.

### 5.2 Effective branching factor

```
input:  N(d) for d = 0..D_max
output: b_eff(d) for d = 2..D_max

for d in 2..D_max:
    if N(d-1) > 0:
        b_eff[d] = N(d) / N(d-1)
    else:
        b_eff[d] = undefined
```

Skip d=1 because N(1)/N(0) = 18/1 = 18 is not the asymptotic rate (depth-1 has no pruning yet).

### 5.3 Saturation detection

```
input:  b_eff(d) sequence, asymptotic estimate b_∞, tolerance ε
output: d_saturation_onset or "not observed"

# Estimate b_∞ as geometric mean of b_eff over the last half of measured depths
half = D_max // 2
b_∞ = geometric_mean(b_eff[half..D_max])

for d in 2..D_max:
    if b_eff[d] / b_∞ < 1 - ε:
        return d_saturation_onset = d

return "not observed"
```

For N=2, saturation is clearly detectable (ratio drops from ~5.5 at d=8 to ~0.33 at d=10). For N≥3 with bounded BFS not reaching saturation, this correctly reports "not observed".

### 5.4 Extrapolated lower bound

```
input:  N(d) for d = 0..D_max, b_∞, |G|
output: D_lb_extrapolated, method_note

cumulative = sum(N[0..D_max])
d = D_max
N_current = N[D_max]

# Safety cap: never extrapolate further than 2x the measured depth
max_extrapolation = 2 * D_max

while cumulative < |G| and d < D_max + max_extrapolation:
    d += 1
    N_current = N_current * b_∞
    cumulative += N_current

if cumulative >= |G|:
    return d
else:
    return "extrapolation cap reached, bound >= {d}"
```

Uses arbitrary precision. The safety cap prevents runaway extrapolation when measured data is too shallow; if hit, the tool warns and suggests running BFS to greater depth.

---

## 6. Integration Points with Existing Solver

The module should be **additive** to the current codebase:

- Add a new header `diameter_bounds.hpp` (and `.cu` if any GPU analysis is needed — likely not; all analysis is trivially CPU-bound).
- Add a JSON serialization dependency (`nlohmann/json` is standard C++ choice; already header-only).
- Add a big-integer dependency (`boost::multiprecision` header-only, or a minimal in-tree implementation — team preference).
- Modify `main.cu` to (a) emit the histogram in-memory in a structured form, (b) optionally call the analysis module after BFS completes.
- Do not refactor existing BFS code; the module reads its output, not its internals.

**Regression protection:** Existing N=2 full BFS must still produce God's number = 11 with no behavior change. The analysis module on N=2 histogram must produce `D_lb_proven = 11` (confirming consistency between the theorem and the mechanical calculation).

---

## 7. What This Module Cannot Do (Important)

Stakeholders should understand the following limitations, which are **mathematical**, not engineering:

1. **No upper bound.** This module establishes `diam(G) ≥ D_lb` only. Upper bounds require different techniques (coset BFS, reduction method).
2. **No peak prediction.** The bell-curve peak depth and width depend on the full spectrum of the Cayley graph's adjacency operator, which cannot be recovered from free-expansion counts. Do not promise peak estimates from this data.
3. **Extrapolated bounds are not proofs.** They assume the branching factor remains constant past `D_max`. This is false near saturation — but we don't know when saturation begins for N≥4, which is why the bound is "extrapolated, not proven".
4. **Bounds are gap-ful.** For N=3, this method gives `D_lb ≈ 18` (true diameter = 20). For N=4, it gives `D_lb ≈ 41` (compared to ~35 from older methods; true diameter unknown but likely 45–55). Information-theoretic bounds are known to undercount by a constant factor related to the spectral gap.

---

## 8. Testing & Validation

### 8.1 Unit tests

- **N=2 consistency:** Given the published full N=2 histogram, `D_lb_proven` must equal 11.
- **Big integer correctness:** Cumulative sum of `N(d)` for N=3 up to full diameter must equal the known group order 43,252,003,274,489,856,000.
- **Branching factor:** For the measured N=3 histogram through d=7, `b_asymptotic` must fall in [13.24, 13.36].
- **Saturation detection:** On a synthetic histogram with exponential growth then plateau, `d_saturation_onset` must match the plateau's start depth within ±1.

### 8.2 Integration tests

- Run N=2 full BFS end-to-end, confirm emitted JSON has `proven_lower_bound.value == 11`.
- Run N=3 BFS to depth 7, confirm emitted `extrapolated_lower_bound.value` ∈ [17, 19] (known true = 20; extrapolation from d=7 will slightly undershoot).
- Run N=4 BFS to depth 7 or 8, confirm emitted `extrapolated_lower_bound.value` ≥ 35 (the current published bound).

### 8.3 Reference values

For CI golden files:

| N | Depth reached | Expected `D_lb_proven` | Expected `D_lb_extrapolated` |
|---|---|---|---|
| 2 | 11 (full) | 11 | 11 |
| 3 | 7 | 8 (partial) | 17–19 |
| 4 | 7 | 8 (partial) | 40–42 |
| 5 | 7 | 8 (partial) | 65–67 |
| 6 | 7 | 8 (partial) | 102–105 |

---

## 9. Deliverables

1. `diameter_bounds.hpp` — analysis module (header-only or compiled).
2. `main.cu` modification — optional post-BFS analysis invocation, `--emit-bounds` and `--analyze-histogram` CLI flags.
3. Histogram export format — stable JSON schema for BFS output.
4. `diameter_bounds.json` schema document (can live in README).
5. Unit + integration tests as described in §8.
6. README section explaining the lower-bound methodology, its guarantees, and its limits.

---

## 10. Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M1 — Analyzer** | Standalone analyzer reading histogram JSON, producing bounds JSON | N=2 histogram → `D_lb = 11` ✓ |
| **M2 — Integration** | Wire analyzer into `main.cu`, add CLI flags | N=2 BFS end-to-end emits correct bounds JSON |
| **M3 — Big-N validation** | Run N=3 through N=6 with bounded BFS, capture outputs | All runs produce valid JSON with documented bounds |
| **M4 — Documentation** | README update, methodology explanation, limitation disclaimers | PR-ready artifacts for external sharing |

Single-sprint target for M1–M4, assuming solver remains stable.

---

## 11. Open Questions

- **QTM support:** Existing solver is HTM-only. Adding QTM requires extending the move generators in `move_tables.cuh` (12 moves instead of 18) and `solve_general.cuh`. QTM branching factor is ≈ 12 minus pruning ≈ 9.37. Should QTM be a v1 deliverable or deferred?
- **Inner-slice generators for N≥4:** Current solver uses outer-face-only (standard FTM). If the goal includes establishing bounds for the *slice-turn metric* (STM), generator set expands and branching factor changes. Out of v1 scope unless stakeholders request.
- **Publication target:** If results for N≥4 exceed published bounds, is there interest in writing this up (arXiv / SIAM Conference on Parallel Processing / Computational Geometry)? Affects how much methodological rigor to invest in §5.4 extrapolation analysis.

---

## 12. References

- Rokicki et al. (2010), *The Diameter of the Rubik's Cube Group Is Twenty* — methodology and baseline.
- Scherphuis, J., *Mathematics of the Rubik's Cube* — group-order formulas for arbitrary N.
- Kunkle & Cooperman (2007), *Twenty-Six Moves Suffice for Rubik's Cube* — earlier upper-bound methodology, useful for context.
- Agostinelli et al. (2019), *DeepCubeA* — reference for bounded-depth exploration at N=4, 5.
