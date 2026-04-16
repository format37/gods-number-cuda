# rubic-solver

CUDA BFS solver that computes God's number (the diameter of the Cayley graph) for the 2x2x2 Rubik's Cube in the Half-Turn Metric.

Explores all 3,674,160 reachable states via breadth-first search on GPU, proving that every position can be solved in at most **11 moves (HTM)**.

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

## Run

```bash
./gods_number
```

Output:

```
=== 2x2x2 Rubik's Cube God's Number Search (HTM) ===

GPU: NVIDIA GeForce RTX 4090 (compute 8.9, 128 SMs, 23.5 GB)

Validating move tables...
  [OK] Lehmer round-trip for all 5040 permutations
  [OK] All 18 move-inverse pairs return to solved
  [OK] 9 unique depth-1 neighbors from solved
  [OK] 10-move sequence + reverse returns to solved
All validations passed.

BFS from solved state (id=0):
  Depth  0:          1 states  (cumulative:          1)
  Depth  1:          9 states  (cumulative:         10)
  ...
  Depth 11:       2644 states  (cumulative:    3674160)

=== Results ===
God's number (HTM): 11
Total states:       3674160
BFS time:           0.003 seconds

ALL CHECKS PASSED
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

Kernel block size (default 256) is defined in `src/main.cu`. The BFS metric (HTM, 18 moves) and cube size (2x2x2) are compile-time constants in the source headers.

## How it works

1. **State encoding** -- Each of the 3,674,160 states of the 2x2x2 cube (with one corner fixed to eliminate rotational symmetry) is encoded as a `uint32` via Lehmer code (permutation) + base-3 (orientation).

2. **Move application** -- All 18 HTM moves are applied on the full 8-corner state using precomputed permutation/orientation tables stored in CUDA constant memory (~690 bytes). Moves that displace the fixed corner trigger a whole-cube re-canonicalization.

3. **BFS with atomicCAS dedup** -- A dense `uint32` depth array (one entry per state) serves as both the visited set and the deduplication mechanism. `atomicCAS` ensures each state is claimed by exactly one thread, eliminating the need for sort-based deduplication. Total GPU memory: ~44 MB.
