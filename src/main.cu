#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

// --------------------------------------------------------------------------
// Shared histogram structure returned by both solvers
// --------------------------------------------------------------------------
struct BFSHistogram {
    uint64_t counts[64];
    int depth_count;      // number of entries (max_depth + 1)
    int cube_size;
};

// --------------------------------------------------------------------------
// N=2 compact solver (dense depth array, Lehmer encoding)
// --------------------------------------------------------------------------
#include "gpu_tables.cuh"
#include "move_tables.cuh"
#include "state_codec.cuh"

__global__ void expand_frontier_2x2(
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

static const uint32_t EXPECTED_HISTOGRAM[] = {
    1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644,
};

static int solve_2x2(int max_depth, BFSHistogram& hist) {
    printf("=== 2x2x2 Rubik's Cube God's Number Search (HTM");
    if (max_depth > 0) printf(", max_depth=%d", max_depth);
    printf(") ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute %d.%d, %d SMs, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    if (!init_move_tables()) {
        fprintf(stderr, "Move table validation FAILED.\n");
        return 1;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    uint32_t* d_depth;
    uint32_t* d_frontier_curr;
    uint32_t* d_frontier_next;
    uint32_t* d_next_count;
    CUDA_CHECK(cudaMalloc(&d_depth, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_depth, 0xFF, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_curr, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_next, NUM_STATES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_count, sizeof(uint32_t)));

    uint32_t solved_id = 0;
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_depth + solved_id, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_curr, &solved_id, sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t frontier_size = 1;
    memset(hist.counts, 0, sizeof(hist.counts));
    hist.counts[0] = 1;
    hist.cube_size = 2;
    uint32_t total = 1;
    int gods_number = 0;

    printf("BFS from solved state (id=%u):\n", solved_id);
    printf("  Depth %2d: %10u states  (cumulative: %10u)\n", 0, 1u, 1u);

    const uint32_t BLOCK_SIZE = 256;

    for (uint32_t depth = 0; frontier_size > 0; depth++) {
        if (max_depth > 0 && (int)depth >= max_depth) {
            printf("\n  Stopped at max_depth=%d\n", max_depth);
            break;
        }

        CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(uint32_t)));
        uint32_t grid_size = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_frontier_2x2<<<grid_size, BLOCK_SIZE>>>(
            d_frontier_curr, frontier_size,
            d_depth, d_frontier_next, d_next_count,
            depth + 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (frontier_size > 0) {
            hist.counts[depth + 1] = frontier_size;
            total += frontier_size;
            gods_number = depth + 1;
            printf("  Depth %2d: %10u states  (cumulative: %10u)\n",
                   depth + 1, frontier_size, total);
        }

        std::swap(d_frontier_curr, d_frontier_next);
    }

    hist.depth_count = gods_number + 1;

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    printf("\n=== Results ===\n");
    bool complete = (frontier_size == 0) && (max_depth <= 0 || gods_number < max_depth);
    if (complete)
        printf("God's number (HTM): %d\n", gods_number);
    else
        printf("Explored to depth:  %d\n", gods_number);
    printf("Total states:       %u\n", total);
    printf("BFS time:           %.3f seconds\n", elapsed);

    if (complete) {
        printf("\n=== Verification ===\n");
        bool ok = true;
        if (total != NUM_STATES) {
            printf("  [FAIL] Total states: %u (expected %u)\n", total, NUM_STATES);
            ok = false;
        } else {
            printf("  [OK]   Total states: %u\n", total);
        }
        if (gods_number != 11) {
            printf("  [FAIL] God's number: %d (expected 11)\n", gods_number);
            ok = false;
        } else {
            printf("  [OK]   God's number: %d\n", gods_number);
        }
        for (int d = 0; d < 12; d++) {
            if (hist.counts[d] != EXPECTED_HISTOGRAM[d]) {
                printf("  [FAIL] Depth %d: %llu (expected %u)\n",
                       d, (unsigned long long)hist.counts[d], EXPECTED_HISTOGRAM[d]);
                ok = false;
            }
        }
        if (ok) printf("  [OK]   Depth histogram matches expected values\n");
        printf("\n%s\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        CUDA_CHECK(cudaFree(d_depth));
        CUDA_CHECK(cudaFree(d_frontier_curr));
        CUDA_CHECK(cudaFree(d_frontier_next));
        CUDA_CHECK(cudaFree(d_next_count));
        return ok ? 0 : 1;
    }

    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_frontier_curr));
    CUDA_CHECK(cudaFree(d_frontier_next));
    CUDA_CHECK(cudaFree(d_next_count));
    return 0;
}

// --------------------------------------------------------------------------
// General NxNxN solver (facelet representation, hash table)
// --------------------------------------------------------------------------
#include "solve_general.cuh"

// --------------------------------------------------------------------------
// Diameter lower-bound analysis
// --------------------------------------------------------------------------
#include "diameter_bounds.hpp"

// --------------------------------------------------------------------------
// CLI
// --------------------------------------------------------------------------
static void usage(const char* prog) {
    printf("Usage: %s [N] [max_depth] [--bounds[=file.json]]\n", prog);
    printf("\n");
    printf("  N          Cube size (default: 2)\n");
    printf("  max_depth  Stop BFS at this depth, 0 = unlimited (default: 0)\n");
    printf("  --bounds   Run diameter lower-bound analysis after BFS\n");
    printf("  --bounds=F Also write JSON report to file F\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                        # 2x2x2, full BFS\n", prog);
    printf("  %s 3 7 --bounds           # 3x3x3, depth 7, with bounds analysis\n", prog);
    printf("  %s 4 7 --bounds=out.json  # 4x4x4, depth 7, JSON report to out.json\n", prog);
}

int main(int argc, char** argv) {
    int N = 2;
    int max_depth = 0;
    bool do_bounds = false;
    const char* bounds_file = nullptr;

    // Parse args
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--bounds") == 0) {
            do_bounds = true;
        } else if (strncmp(argv[i], "--bounds=", 9) == 0) {
            do_bounds = true;
            bounds_file = argv[i] + 9;
        } else if (argv[i][0] != '-') {
            if (positional == 0) N = atoi(argv[i]);
            else if (positional == 1) max_depth = atoi(argv[i]);
            positional++;
        }
    }

    if (N < 2 || N > 16) {
        fprintf(stderr, "N must be between 2 and 16 (got %d)\n", N);
        return 1;
    }

    // Run BFS
    BFSHistogram hist;
    memset(&hist, 0, sizeof(hist));
    hist.cube_size = N;
    int rc;

    if (N == 2) {
        rc = solve_2x2(max_depth, hist);
    } else {
        rc = solve_general(N, max_depth, hist);
    }

    // Bounds analysis
    if (do_bounds && hist.depth_count >= 3) {
        BoundsResult bounds = analyze_bounds(
            hist.counts, hist.depth_count, N);

        print_bounds_summary(bounds);

        if (bounds_file) {
            FILE* fp = fopen(bounds_file, "w");
            if (fp) {
                write_bounds_json(bounds, fp);
                fclose(fp);
                printf("Bounds JSON written to: %s\n", bounds_file);
            } else {
                fprintf(stderr, "Failed to open %s for writing\n", bounds_file);
            }
        }
    } else if (do_bounds && hist.depth_count < 3) {
        printf("\nBounds analysis requires at least 3 BFS depths.\n");
    }

    return rc;
}
