# Tech Spec: God's Number Search for N×N×N Rubik's Cube via Cayley Graph BFS on CUDA

**Status:** M1+M2 Complete (N=2 HTM verified)
**Target hardware:** NVIDIA RTX 4090 (24 GB GDDR6X, compute 8.9, 128 SMs)
**Target:** Computing or bounding God's number (Cayley graph diameter) for cubes of arbitrary size N, with CUDA acceleration and optional coset-based group decomposition.

---

## 1. Goals & Scope

### 1.1 Primary Goals

- **Exact computation** of God's number for tractable cubes (N=2 definitive; N=3 only via coset decomposition and massive compute).
- **Upper/lower bound estimation** for N≥4, where exact BFS is provably infeasible.
- A pure CUDA C++ codebase compiled with nvcc, targeting the RTX 4090 (24 GB VRAM).

### 1.2 Non-Goals

- Building a fast solver for individual scrambles (different problem; use DeepCubeA / Kociemba-style solvers).
- Exact God's number for N≥4 (infeasible: state space for N=4 is ~10⁴⁵, exceeds any conceivable storage).
- Supporting non-cubic twisty puzzles (Megaminx, Square-1) in v1.

### 1.3 Success Criteria

| Deliverable | Target | Status |
|---|---|---|
| N=2 exact God's number, HTM metric | **11** (verified against published result) | **DONE** (0.003s) |
| N=2 full depth histogram | Matches published table | **DONE** |
| N=2 exact God's number, QTM metric | **14** (verified) | Pending |
| N=3 coset BFS pipeline | Runs end-to-end on ≥1 coset, produces valid depth data | Pending |
| N=4+ bounded search | Produces lower bound via reachability from solved state up to configurable depth D | Pending |

---

## 2. Background & Problem Statement

### 2.1 Definitions

- **State space**: All reachable configurations of an N×N×N cube from the solved state.
- **Cayley graph** *G = (V, E)*: vertices = states, edges = single-move transitions under a chosen generator set (face turns).
- **God's number**: diameter of *G*, i.e. max over all states *s* of the shortest path length from *s* to solved.
- **HTM (Half-Turn Metric)**: every face rotation (90°, 180°, 270°) counts as 1 move.
- **QTM (Quarter-Turn Metric)**: only 90° rotations count as 1 move; 180° = 2 moves.

### 2.2 State Space Sizes

| N | |V| | Feasibility |
|---|---|---|
| 2 | 3,674,160 | Direct BFS trivially feasible |
| 3 | 4.3 × 10¹⁹ | Requires coset decomposition + distributed compute |
| 4 | 7.4 × 10⁴⁵ | Exact infeasible; bounded search only |
| 5 | 2.8 × 10⁷⁴ | Exact infeasible; bounded search only |
| 6 | 1.6 × 10¹¹⁶ | Exact infeasible; bounded search only |

### 2.3 Generator Sets

For N×N×N, the set of face-layer moves:

- **Outer faces**: 6 faces × 3 rotations (90°, 180°, 270°) = 18 moves (HTM) or 12 moves (QTM).
- **Inner slices** (for N≥4): additional 6 × (N − 2) × 3 moves per axis family if treating inner slice turns as generators.

---

## 3. Architecture Overview

Pure CUDA C++ with single-file compilation via nvcc.

```
┌───────────────────────────────────────────────────────────┐
│  Host BFS Driver (C++, main.cu)                           │
│  ─ frontier management, depth accounting, histogram       │
│  ─ cudaMemcpy for frontier size readback                  │
└─────────────┬─────────────────────────────────────────────┘
              │ frontier buffer pointers
┌─────────────▼─────────────────────────────────────────────┐
│  expand_frontier kernel (CUDA, main.cu)                   │
│  ─ decode state → apply all 18 moves → re-canonicalize    │
│  ─ atomicCAS on depth_array for dedup                     │
│  ─ atomicAdd to append to next_frontier                   │
└─────────────┬─────────────────────────────────────────────┘
              │
┌─────────────▼─────────────────────────────────────────────┐
│  Device Memory                                            │
│  ─ depth_array: uint32[3674160] (14.7 MB) — dense visited │
│  ─ frontier_curr / frontier_next: uint32[3674160] each    │
│  ─ __constant__: move tables + recanon tables (~1 KB)     │
└───────────────────────────────────────────────────────────┘
```

### 3.1 File Structure

```
src/
  main.cu            — Host BFS loop + expand_frontier kernel + main()
  gpu_tables.cuh     — __constant__ memory declarations (shared)
  state_codec.cuh    — __device__ Lehmer encode/decode, orientation codec,
                       apply_move(), re_canonicalize()
  move_tables.cuh    — 18 move perm/orient tables, 24 re-canonicalization
                       rotations, host-side init + validation
CMakeLists.txt       — Build targeting sm_89, CUDA C++17, -O3 --use_fast_math
```

All device code lives in a single translation unit (`main.cu`) to share `__constant__` memory without requiring relocatable device code.

---

## 4. State Representation (N=2)

### 4.1 Corner Convention

8 physical corner positions, standard Singmaster labeling:

| Index | Position | Sticker faces |
|-------|----------|---------------|
| 0 | URF | U, R, F |
| 1 | UFL | U, F, L |
| 2 | ULB | U, L, B |
| 3 | UBR | U, B, R |
| 4 | DFR | D, F, R |
| 5 | DLF | D, L, F |
| 6 | DBL | D, B, L |
| 7 | DRB | D, R, B |

**Corner 7 (DRB) is fixed** as reference frame, eliminating the 24× whole-cube rotation redundancy. State is described by:
- `perm[7]`: which corner piece occupies each of the 7 mobile positions (values 0–6)
- `orient[7]`: twist of each corner (0 = U/D sticker on U/D face, 1 = CW 120°, 2 = CCW 120°)

### 4.2 Encoding

- **Permutation**: Lehmer code over 7 corners → `perm_idx` ∈ [0, 5040)
- **Orientation**: base-3 encoding of first 6 twists → `orient_idx` ∈ [0, 729). 7th twist recovered from parity constraint (sum ≡ 0 mod 3).
- **Combined**: `state_id = perm_idx * 729 + orient_idx` ∈ [0, 3,674,160)
- **Storage**: `uint32` (4 bytes per state)

### 4.3 Device Implementation

Lehmer encode/decode: O(49) register operations per thread (nested loop over 7 elements). All in registers, no shared memory needed.

Orientation encode: Horner's method, 5 multiply-adds. Decode: 6 divmods by 3.

---

## 5. Move Engine

### 5.1 Move Representation

All 18 HTM moves operate on the full 8-corner state using source-convention permutation tables:

- `move_perm[18][8]`: for each move, `new_perm[i] = old_perm[move_perm[move_id][i]]`
- `move_orient[18][8]`: `new_orient[i] = (old_orient[src] + move_orient[move_id][i]) % 3`

Move ordering: U U2 U' R R2 R' F F2 F' D D2 D' L L2 L' B B2 B'

### 5.2 Base Face Turns (6 generators)

Each defined as a 4-cycle with orientation deltas:

| Face | Cycle | Orient delta |
|------|-------|--------------|
| U | (0 3 2 1) | all zero |
| R | (0 4 7 3) | [2,0,0,1,1,0,0,2] |
| F | (0 1 5 4) | [1,2,0,0,2,1,0,0] |
| D | (4 5 6 7) | all zero |
| L | (1 2 6 5) | [0,1,2,0,0,2,1,0] |
| B | (2 3 7 6) | [0,0,1,2,0,0,2,1] |

180° and 270° moves derived by composition. U/D moves (rotating around U-D axis) have no orientation effect. F/B/R/L moves twist the 4 affected corners.

### 5.3 Re-canonicalization

Moves on faces D, R, B displace corner 7 from position 7. After any move:

1. Find which position `p` now holds piece 7, and its orientation `o`
2. Look up inverse whole-cube rotation from `recanon_perm[p*3+o]` and `recanon_orient[p*3+o]`
3. Apply rotation to all 8 corners, restoring piece 7 to position 7 with orientation 0

The 24 re-canonicalization rotations are generated at startup by BFS over the rotation group (composing basic x/y/z whole-cube rotations), then inverted.

### 5.4 Constant Memory Layout

| Table | Size | Description |
|-------|------|-------------|
| `d_move_perm[18][8]` | 144 B | Move source permutations |
| `d_move_orient[18][8]` | 144 B | Move orientation deltas |
| `d_recanon_perm[24][8]` | 192 B | Re-canonicalization permutations |
| `d_recanon_orient[24][8]` | 192 B | Re-canonicalization orient deltas |
| `d_inv_move[18]` | 18 B | Inverse move lookup |
| **Total** | **~690 B** | Well within 64 KB constant memory limit |

---

## 6. BFS Search Driver (N=2)

### 6.1 Algorithm

Dense BFS using `atomicCAS` for implicit deduplication — no sort/unique pass needed.

```
depth_array[3674160] = all UNVISITED (0xFFFFFFFF)
depth_array[solved_id] = 0
frontier = [solved_id]

for depth = 0, 1, 2, ...:
    launch expand_frontier kernel:
        each thread processes 1 frontier state × 18 moves
        for each new_state:
            if depth_array[new_state] == UNVISITED:
                atomicCAS(depth_array[new_state], UNVISITED, depth+1)
                if CAS succeeded: atomicAdd to append to next_frontier
    read back frontier size
    swap frontier buffers
```

### 6.2 Kernel Design

```cuda
__global__ void expand_frontier(
    const uint32_t* frontier, uint32_t frontier_size,
    uint32_t* depth_array,
    uint32_t* next_frontier, uint32_t* next_count,
    uint32_t next_depth);
```

- **Thread mapping**: 1 thread per frontier state (processes all 18 moves in a loop)
- **Block size**: 256
- **Grid size**: `ceil(frontier_size / 256)`
- **Dedup**: `atomicCAS` on `uint32_t depth_array` — each state claimed by exactly one thread
- **Frontier append**: `atomicAdd` on counter, write state to `next_frontier[pos]`

### 6.3 Memory Budget (RTX 4090, 24 GB)

| Allocation | Size |
|-----------|------|
| `depth_array` (uint32 × 3,674,160) | 14.7 MB |
| `frontier_curr` (uint32 × 3,674,160) | 14.7 MB |
| `frontier_next` (uint32 × 3,674,160) | 14.7 MB |
| Constant memory (move tables) | ~0.7 KB |
| **Total** | **~44 MB (0.18% of VRAM)** |

### 6.4 Verified Results

**Runtime: 0.003 seconds** on RTX 4090.

| Depth | States | Cumulative |
|-------|--------|------------|
| 0 | 1 | 1 |
| 1 | 9 | 10 |
| 2 | 54 | 64 |
| 3 | 321 | 385 |
| 4 | 1,847 | 2,232 |
| 5 | 9,992 | 12,224 |
| 6 | 50,136 | 62,360 |
| 7 | 227,536 | 289,896 |
| 8 | 870,072 | 1,159,968 |
| 9 | 1,887,748 | 3,047,716 |
| 10 | 623,800 | 3,671,516 |
| 11 | 2,644 | 3,674,160 |

**God's number (HTM) = 11. Total states = 3,674,160.** All match published values.

---

## 7. Validation

### 7.1 Startup Checks (run before BFS)

1. **Lehmer round-trip**: encode → decode identity for all 5,040 permutations
2. **Move-inverse identity**: apply move then inverse from solved → returns to solved, for all 18 moves
3. **Orientation parity**: sum of all 8 orientations ≡ 0 (mod 3) after every move
4. **Re-canonicalization**: piece 7 is at position 7 with orientation 0 after every move+recanon
5. **Depth-1 count**: exactly 9 unique neighbors from solved (18 moves, but opposite faces are conjugate under the fixed-corner frame)
6. **Multi-move sequence**: 10-move chain + reverse returns to solved

### 7.2 BFS Output Checks

- Total states = 3,674,160
- God's number = 11 (HTM)
- Full depth histogram matches published reference values

---

## 8. Build & Run

### 8.1 Prerequisites

- CUDA Toolkit 12.0+
- CMake 3.18+
- NVIDIA GPU with compute capability ≥ 7.0 (tested on RTX 4090, sm_89)

### 8.2 Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 8.3 Run

```bash
./gods_number
```

---

## 9. Future Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M1 — N=2 HTM** | State codec, move engine, BFS on GPU | **DONE** — God's number = 11, 0.003s |
| **M2 — N=2 QTM** | Modify generator set to 12 QTM moves | N=2 God's number = 14 (QTM) |
| **M3 — N=3 infra** | Kociemba coordinates, coordinate kernels, single-coset BFS | One coset BFS produces correct per-coset upper bound |
| **M4 — Facelet N** | Facelet codec + move engine for arbitrary N | N=4 bounded BFS runs to depth 8+ |
| **M5 — Coset orchestration** | Multi-coset scheduler, checkpointing, distributed runs | K cosets processed in parallel, results aggregated |
| **M6 — Production run** | Execute N=3 God's number verification | Full N=3 HTM diameter verified at 20 |

---

## 10. Risks & Open Questions

- **Orientation delta correctness**: The most subtle part of the implementation. Validated via move-inverse tests and parity invariants.
- **Re-canonicalization**: 24 whole-cube rotations must be generated and inverted correctly. Validated by checking piece 7 returns to position 7 with orientation 0 after every move.
- **Hash table contention on GPU** for N≥4: may need careful tuning of load factor, or switch to sorted-batch deduplication.
- **Coset enumeration correctness** (N=3): requires careful implementation of Kociemba's two-phase coordinate system.
- **Compute budget for full N=3**: needs dedicated GPU cluster time.

---

## 11. References

- Rokicki, Kociemba, Davidson, Dethridge (2010): *The Diameter of the Rubik's Cube Group Is Twenty*. Original proof of N=3 God's number.
- Kociemba's two-phase algorithm: https://kociemba.org/cube.htm
- Agostinelli et al. (2019): *Solving the Rubik's Cube with Deep Reinforcement Learning and Search* (DeepCubeA) — reference for N=4, N=5 bounded approaches.
- Rokicki's `cube20src` codebase: reference C++ implementation of coset BFS.
