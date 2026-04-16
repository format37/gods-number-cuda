# gods-number-cuda

CUDA BFS solver that computes or explores God's number (the diameter of the Cayley graph) for NxNxN Rubik's Cubes in the Half-Turn Metric.

- **N=2**: Full BFS over all 3,674,160 states. Proves God's number = **11 HTM** in 0.003s.
- **N=3+**: Bounded BFS to a configurable depth. Explores frontier growth and state counts.

## Prerequisites

- NVIDIA GPU (compute capability >= 7.0)
- CUDA Toolkit 12.0+
- CMake 3.18+

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```
./gods_number [N] [max_depth] [--bounds[=file.json]]

  N          Cube size (default: 2)
  max_depth  Stop BFS at this depth, 0 = unlimited (default: 0)
  --bounds   Run diameter lower-bound analysis after BFS
  --bounds=F Also write JSON report to file F
```

### Examples

```bash
./gods_number                        # 2x2x2 full BFS -> God's number = 11
./gods_number 3 7                    # 3x3x3 BFS to depth 7
./gods_number 4 7 --bounds           # 4x4x4 depth 7 + lower-bound analysis
./gods_number 4 7 --bounds=out.json  # same, with JSON report
```

### Sample output (N=3, depth 7)

```
=== 3x3x3 Rubik's Cube BFS (HTM, max_depth=7) ===

GPU: NVIDIA GeForce RTX 4090 (compute 8.9, 128 SMs, 23.5 GB)

BFS from solved state:
  Depth  0:            1 states  (cumulative:            1)
  Depth  1:           18 states  (cumulative:           19)
  Depth  2:          243 states  (cumulative:          262)
  Depth  3:         3240 states  (cumulative:         3502)
  Depth  4:        43239 states  (cumulative:        46741)
  Depth  5:       575372 states  (cumulative:       622113)
  Depth  6:      7636058 states  (cumulative:      8258171)
  Depth  7:    101229076 states  (cumulative:    109487247)

Explored to depth:  7
Total states:       109487247
BFS time:           0.121 seconds
```

## Configuration

The GPU architecture target is set in `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4090 (Ada Lovelace)
```

Common values:

| GPU | Architecture | Value |
|-----|-------------|-------|
| RTX 3090 / A100 | Ampere | `86` / `80` |
| RTX 4090 | Ada Lovelace | `89` |
| H100 | Hopper | `90` |

To target multiple architectures:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
```

## How it works

### N=2 (compact solver)

1. **State encoding** -- Each of the 3,674,160 states (with one corner fixed to eliminate rotational symmetry) is encoded as a `uint32` via Lehmer code (permutation) + base-3 (orientation).
2. **Move application** -- 18 HTM moves via precomputed permutation/orientation tables in CUDA constant memory (~690 bytes). Moves displacing the fixed corner trigger whole-cube re-canonicalization.
3. **Dedup** -- Dense `uint32` depth array with `atomicCAS`. Total GPU memory: ~44 MB.

### N>=3 (general solver)

1. **State encoding** -- Facelet array: `uint8[6*N*N]`, each value 0-5 (face color). State sizes: 54 bytes (N=3), 96 bytes (N=4), 150 bytes (N=5).
2. **Move application** -- Precomputed facelet permutations generated from 3D rotation geometry with face-mapping to resolve edge/corner coordinate ambiguity.
3. **Dedup** -- GPU open-addressing hash table (64-bit FNV-1a + murmurhash3 finalizer, linear probing). Memory auto-sized to ~85% of free VRAM.
4. **Depth limit** -- BFS stops at `max_depth` or when the state buffer fills. On a 24 GB GPU, N=3 fits ~210M states (depth ~7-8).

## Diameter lower bounds

The `--bounds` flag runs a post-BFS analysis that computes:

- **Proven lower bound**: the smallest depth where the cumulative state count could cover the group. For bounded BFS this is `D_max + 1` (trivial but rigorous).
- **Extrapolated lower bound**: assumes the measured branching factor continues, estimates when cumulative states would reach |G|. Not a proof, but a useful estimate.
- **Branching factor**: `b_eff(d) = N(d)/N(d-1)`, converging to ~13.3 for outer-face HTM.
- **Saturation detection**: identifies when frontier growth starts slowing (observable for N=2, not for bounded BFS on larger cubes).

### Reference results (RTX 4090, depth 7)

| Cube | Group order | Proven LB | Extrapolated LB | Known diameter |
|------|------------|-----------|-----------------|----------------|
| 2x2x2 | 3.67e6 | **11** (exact) | 11 | 11 |
| 3x3x3 | 4.33e19 | 8 | 18 | 20 |
| 4x4x4 | 7.40e45 | 8 | 41 | unknown |
| 5x5x5 | 2.83e74 | 7 | 67 | unknown |
| 6x6x6 | 1.57e116 | 7 | 104 | unknown |

See `docs/diameter_lower_bound_tech_spec.md` for methodology details.
