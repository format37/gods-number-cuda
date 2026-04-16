#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

#include "gpu_tables.cuh"
#include "move_tables.cuh"
#include "state_codec.cuh"

// ---- Kernel (inline in same TU to share __constant__ memory) ----
__global__ void expand_frontier(
    const uint32_t* __restrict__ frontier,
    uint32_t frontier_size,
    uint32_t* depth_array,
    uint32_t* next_frontier,
    uint32_t* next_count,
    uint32_t next_depth)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    uint32_t state_id = frontier[tid];
    uint8_t perm[8], orient[8];
    decode_state(state_id, perm, orient);

    for (int m = 0; m < NUM_MOVES; m++) {
        uint8_t new_perm[8], new_orient[8];
        apply_move(perm, orient, m, new_perm, new_orient);
        re_canonicalize(new_perm, new_orient);

        uint32_t new_state = encode_state(new_perm, new_orient);

        if (depth_array[new_state] != UNVISITED) continue;

        uint32_t old = atomicCAS(&depth_array[new_state], UNVISITED, next_depth);
        if (old == UNVISITED) {
            uint32_t pos = atomicAdd(next_count, 1);
            next_frontier[pos] = new_state;
        }
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Expected HTM depth histogram for verification
static const uint32_t EXPECTED_HISTOGRAM[] = {
    1,          // depth 0
    9,          // depth 1
    54,         // depth 2
    321,        // depth 3
    1847,       // depth 4
    9992,       // depth 5
    50136,      // depth 6
    227536,     // depth 7
    870072,     // depth 8
    1887748,    // depth 9
    623800,     // depth 10
    2644,       // depth 11
};
static const int EXPECTED_GODS_NUMBER = 11;
static const int EXPECTED_HISTOGRAM_SIZE = 12; // depths 0-11

int main() {
    printf("=== 2x2x2 Rubik's Cube God's Number Search (HTM) ===\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute %d.%d, %d SMs, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Initialize and validate move tables
    if (!init_move_tables()) {
        fprintf(stderr, "Move table validation FAILED. Aborting.\n");
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Allocate depth array on device (uint32, one entry per state)
    uint32_t* d_depth;
    CUDA_CHECK(cudaMalloc(&d_depth, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_depth, 0xFF, NUM_STATES * sizeof(uint32_t)));

    // Allocate frontier buffers
    uint32_t* d_frontier_curr;
    uint32_t* d_frontier_next;
    uint32_t* d_next_count;
    CUDA_CHECK(cudaMalloc(&d_frontier_curr, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_next, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count, sizeof(uint32_t)));

    // Seed with solved state (identity permutation, zero orientation = state_id 0)
    uint32_t solved_id = 0;
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_depth + solved_id, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_curr, &solved_id, sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t frontier_size = 1;
    uint32_t histogram[32] = {0};
    histogram[0] = 1;
    uint32_t total = 1;
    int gods_number = 0;

    printf("BFS from solved state (id=%u):\n", solved_id);
    printf("  Depth %2d: %10u states  (cumulative: %10u)\n", 0, 1u, 1u);

    const uint32_t BLOCK_SIZE = 256;

    // BFS loop
    for (uint32_t depth = 0; frontier_size > 0; depth++) {
        // Reset next frontier counter
        CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(uint32_t)));

        // Launch kernel
        uint32_t grid_size = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_frontier<<<grid_size, BLOCK_SIZE>>>(
            d_frontier_curr, frontier_size,
            d_depth, d_frontier_next, d_next_count,
            depth + 1
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back next frontier size
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (frontier_size > 0) {
            histogram[depth + 1] = frontier_size;
            total += frontier_size;
            gods_number = depth + 1;
            printf("  Depth %2d: %10u states  (cumulative: %10u)\n",
                   depth + 1, frontier_size, total);
        }

        // Swap frontier buffers
        std::swap(d_frontier_curr, d_frontier_next);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // Results
    printf("\n=== Results ===\n");
    printf("God's number (HTM): %d\n", gods_number);
    printf("Total states:       %u\n", total);
    printf("BFS time:           %.3f seconds\n", elapsed);

    // Verification
    printf("\n=== Verification ===\n");
    bool ok = true;

    if (total != NUM_STATES) {
        printf("  [FAIL] Total states: %u (expected %u)\n", total, NUM_STATES);
        ok = false;
    } else {
        printf("  [OK]   Total states: %u\n", total);
    }

    if (gods_number != EXPECTED_GODS_NUMBER) {
        printf("  [FAIL] God's number: %d (expected %d)\n", gods_number, EXPECTED_GODS_NUMBER);
        ok = false;
    } else {
        printf("  [OK]   God's number: %d\n", gods_number);
    }

    for (int d = 0; d < EXPECTED_HISTOGRAM_SIZE; d++) {
        if (histogram[d] != EXPECTED_HISTOGRAM[d]) {
            printf("  [FAIL] Depth %d: %u (expected %u)\n", d, histogram[d], EXPECTED_HISTOGRAM[d]);
            ok = false;
        }
    }
    if (ok) {
        printf("  [OK]   Depth histogram matches expected values\n");
    }

    printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_frontier_curr));
    CUDA_CHECK(cudaFree(d_frontier_next));
    CUDA_CHECK(cudaFree(d_next_count));

    return ok ? 0 : 1;
}
