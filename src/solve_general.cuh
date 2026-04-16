#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

// ============================================================================
// General NxNxN Rubik's Cube BFS solver — facelet representation
//
// State: uint8_t[6*N*N], each facelet is a color 0–5.
// Moves: precomputed source-permutations of facelets, stored in GPU global mem.
// Dedup: GPU open-addressing hash table (64-bit hashes, linear probing).
// States: flat GPU buffer, indexed by uint32 state index.
// ============================================================================

#define MAX_FACELET_STATE 1536 // 6*N*N, supports N up to 16

// --------------------------------------------------------------------------
// 3D coordinate helpers for move generation
//
// Coordinate system: cube occupies [0,N-1]^3.
// Face assignments: U(y=N-1), D(y=0), F(z=N-1), B(z=0), R(x=N-1), L(x=0).
//
// IMPORTANT: Edge/corner facelets share 3D coordinates between faces.
// To resolve ambiguity, we track source face → destination face explicitly
// using FACE_MAP, and use coord_to_facelet_on_face() instead of
// coord_to_facelet().
// --------------------------------------------------------------------------
struct IVec3 { int x, y, z; };

static IVec3 facelet_to_3d(int face, int r, int c, int N) {
    switch (face) {
        case 0: return {c, N-1, N-1-r};       // U
        case 1: return {N-1, N-1-r, N-1-c};   // R
        case 2: return {c, N-1-r, N-1};        // F
        case 3: return {c, 0, r};              // D
        case 4: return {0, N-1-r, c};          // L
        default: return {N-1-c, N-1-r, 0};    // B
    }
}

// Convert 3D point to facelet index KNOWING which face it belongs to.
// This avoids the edge/corner ambiguity of the generic coord_to_facelet.
static int coord_to_facelet_on_face(IVec3 p, int face, int N) {
    int r, c;
    switch (face) {
        case 0: r = N-1-p.z; c = p.x; break;       // U
        case 1: r = N-1-p.y; c = N-1-p.z; break;   // R
        case 2: r = N-1-p.y; c = p.x; break;        // F
        case 3: r = p.z;     c = p.x; break;        // D
        case 4: r = N-1-p.y; c = p.z; break;        // L
        case 5: r = N-1-p.y; c = N-1-p.x; break;   // B
        default: return -1;
    }
    return face * N * N + r * N + c;
}

static bool in_face_layer(int face, IVec3 p, int N) {
    switch (face) {
        case 0: return p.y == N-1;
        case 1: return p.x == N-1;
        case 2: return p.z == N-1;
        case 3: return p.y == 0;
        case 4: return p.x == 0;
        case 5: return p.z == 0;
    }
    return false;
}

// CW rotation around the face's axis (Singmaster CW convention)
static IVec3 rotate_face_cw(int face, IVec3 p, int N) {
    switch (face) {
        case 0: return {p.z,     p.y, N-1-p.x};  // U: (x,z) -> (z, N-1-x)
        case 1: return {p.x,     p.z, N-1-p.y};  // R: (y,z) -> (z, N-1-y)
        case 2: return {p.y, N-1-p.x,     p.z};  // F: (x,y) -> (y, N-1-x)
        case 3: return {N-1-p.z, p.y,     p.x};  // D: (x,z) -> (N-1-z, x)
        case 4: return {p.x, N-1-p.z,     p.y};  // L: (y,z) -> (N-1-z, y)
        case 5: return {p.y, N-1-p.x,     p.z};  // B: (x,y) -> (y, N-1-x)
    }
    return p;
}

// Face mapping: FACE_MAP[rotating_face][src_face] = dst_face
// When face `rotating_face` rotates CW, a sticker on `src_face` moves to `dst_face`.
// -1 means src_face shouldn't be in the rotation layer.
static const int FACE_MAP[6][6] = {
    // U CW: U->U, R->B, F->R, D->X, L->F, B->L
    { 0,  5,  1, -1,  2,  4},
    // R CW: U->B, R->R, F->U, D->F, L->X, B->D
    { 5,  1,  0,  2, -1,  3},
    // F CW: U->R, R->D, F->F, D->L, L->U, B->X
    { 1,  3,  2,  4,  0, -1},
    // D CW: U->X, R->F, F->L, D->D, L->B, B->R
    {-1,  2,  4,  3,  5,  1},
    // L CW: U->F, R->X, F->D, D->B, L->L, B->U
    { 2, -1,  3,  5,  4,  0},
    // B CW: U->L, R->U, F->X, D->R, L->D, B->B
    { 4,  0, -1,  1,  3,  5},
};

// --------------------------------------------------------------------------
// Host-side move table generation
// --------------------------------------------------------------------------
static void generate_cw_perm(int face, int N, uint16_t* perm) {
    int nf = 6 * N * N;
    for (int i = 0; i < nf; i++) perm[i] = (uint16_t)i; // identity

    // 1. Face grid CW rotation: new[r][c] = old[N-1-c][r]
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            perm[face*N*N + r*N + c] = (uint16_t)(face*N*N + (N-1-c)*N + r);
        }
    }

    // 2. Strip facelets: rotate in 3D, use face_map to resolve destination face
    for (int src = 0; src < nf; src++) {
        int src_face = src / (N*N);
        if (src_face == face) continue; // handled by face grid rotation

        int r = (src % (N*N)) / N, c = src % N;
        IVec3 p = facelet_to_3d(src_face, r, c, N);
        if (!in_face_layer(face, p, N)) continue;

        IVec3 p_new = rotate_face_cw(face, p, N);
        int dst_face = FACE_MAP[face][src_face];
        int dst = coord_to_facelet_on_face(p_new, dst_face, N);
        perm[dst] = (uint16_t)src;
    }
}

static void compose_perm(const uint16_t* a, const uint16_t* b, uint16_t* out, int n) {
    // Apply a first, then b: out[i] = a[b[i]]
    for (int i = 0; i < n; i++) out[i] = a[b[i]];
}

// Generate 18 moves: face*3 + {CW=0, 180=1, CCW=2}
// Face order: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B
static void generate_all_moves(int N, std::vector<uint16_t>& flat) {
    int nf = 6 * N * N;
    flat.resize(18 * nf);

    std::vector<uint16_t> cw(nf), cw2(nf), ccw(nf);
    for (int face = 0; face < 6; face++) {
        generate_cw_perm(face, N, cw.data());
        compose_perm(cw.data(), cw.data(), cw2.data(), nf);
        compose_perm(cw2.data(), cw.data(), ccw.data(), nf);
        memcpy(&flat[(face*3+0)*nf], cw.data(),  nf * sizeof(uint16_t));
        memcpy(&flat[(face*3+1)*nf], cw2.data(), nf * sizeof(uint16_t));
        memcpy(&flat[(face*3+2)*nf], ccw.data(), nf * sizeof(uint16_t));
    }
}

// Validate moves: CW^4 = identity, CW*CCW = identity
static bool validate_general_moves(int N, const std::vector<uint16_t>& moves) {
    int nf = 6 * N * N;
    printf("Validating %dx%dx%d move tables (%d facelets, 18 moves)...\n", N, N, N, nf);

    std::vector<uint16_t> composed(nf), temp(nf);
    for (int face = 0; face < 6; face++) {
        const uint16_t* cw = &moves[(face*3+0)*nf];
        const uint16_t* ccw = &moves[(face*3+2)*nf];

        // CW * CCW = identity
        compose_perm(cw, ccw, composed.data(), nf);
        for (int i = 0; i < nf; i++) {
            if (composed[i] != i) {
                fprintf(stderr, "  [FAIL] Face %d: CW*CCW != identity at %d\n", face, i);
                return false;
            }
        }

        // CW^4 = identity
        memcpy(composed.data(), cw, nf * sizeof(uint16_t));
        for (int rep = 1; rep < 4; rep++) {
            compose_perm(composed.data(), cw, temp.data(), nf);
            std::swap(composed, temp);
        }
        for (int i = 0; i < nf; i++) {
            if (composed[i] != i) {
                fprintf(stderr, "  [FAIL] Face %d: CW^4 != identity at %d\n", face, i);
                return false;
            }
        }
    }

    // Check that each move produces a valid permutation (no duplicate targets)
    for (int m = 0; m < 18; m++) {
        const uint16_t* perm = &moves[m * nf];
        std::vector<bool> seen(nf, false);
        for (int i = 0; i < nf; i++) {
            if (perm[i] >= nf || seen[perm[i]]) {
                fprintf(stderr, "  [FAIL] Move %d: invalid permutation at %d\n", m, i);
                return false;
            }
            seen[perm[i]] = true;
        }
    }

    // Check that from solved, all 18 moves produce distinct states
    std::vector<uint8_t> solved(nf), newstate(nf);
    for (int i = 0; i < nf; i++) solved[i] = (uint8_t)(i / (N*N));

    std::vector<std::vector<uint8_t>> neighbors;
    for (int m = 0; m < 18; m++) {
        const uint16_t* perm = &moves[m * nf];
        for (int i = 0; i < nf; i++) newstate[i] = solved[perm[i]];
        bool dup = false;
        for (auto& prev : neighbors) {
            if (prev == newstate) { dup = true; break; }
        }
        if (!dup) neighbors.push_back(newstate);
    }

    int expected_d1 = (N == 2) ? 18 : 18; // all distinct for facelet rep
    // For N=2, opposite face turns are NOT equivalent in facelet rep
    // (no re-canonicalization), so all 18 are distinct.
    // Actually for any N, from the unique solved state, all 18 face turns
    // produce distinct facelet arrays (each move affects different facelets).
    if ((int)neighbors.size() != expected_d1) {
        fprintf(stderr, "  [FAIL] Expected %d unique depth-1 neighbors, got %d\n",
                expected_d1, (int)neighbors.size());
        return false;
    }

    printf("  [OK] All 18 CW*CCW = identity\n");
    printf("  [OK] All 18 CW^4 = identity\n");
    printf("  [OK] All 18 permutations valid\n");
    printf("  [OK] %d unique depth-1 neighbors from solved\n", (int)neighbors.size());
    printf("All validations passed.\n\n");
    return true;
}

// --------------------------------------------------------------------------
// Device: hash function (FNV-1a + murmurhash3 finalizer)
// --------------------------------------------------------------------------
__device__ __forceinline__
uint64_t hash_facelets(const uint8_t* state, int size) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < size; i++) {
        h ^= state[i];
        h *= 1099511628211ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h | 1; // force non-zero (0 = empty sentinel)
}

// --------------------------------------------------------------------------
// Device: BFS expansion kernel for general NxNxN
// --------------------------------------------------------------------------
__global__ void expand_frontier_general(
    const uint8_t*  __restrict__ states_buf,
    const uint32_t* __restrict__ frontier,
    uint32_t  frontier_size,
    const uint16_t* __restrict__ move_perms,
    int       num_moves,
    int       state_size,
    uint64_t* hash_table,
    uint32_t  hash_mask,
    uint8_t*  states_buf_w,       // writable alias of states_buf for new states
    uint32_t* next_frontier,
    uint32_t* next_count,
    uint32_t* state_count,
    uint32_t  max_states)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    uint32_t idx = frontier[tid];
    const uint8_t* src = states_buf + (uint64_t)idx * state_size;

    // Load current state into local memory
    uint8_t local_state[MAX_FACELET_STATE];
    for (int i = 0; i < state_size; i++) local_state[i] = src[i];

    uint8_t new_state[MAX_FACELET_STATE];

    for (int m = 0; m < num_moves; m++) {
        const uint16_t* perm = move_perms + m * state_size;

        // Apply permutation
        for (int i = 0; i < state_size; i++) {
            new_state[i] = local_state[perm[i]];
        }

        // Hash
        uint64_t h = hash_facelets(new_state, state_size);

        // Hash table insert (linear probing)
        uint32_t slot = (uint32_t)(h & hash_mask);
        bool inserted = false;
        for (int probe = 0; probe < 8192; probe++) {
            uint64_t old = atomicCAS(
                (unsigned long long*)&hash_table[slot],
                0ULL,
                (unsigned long long)h);
            if (old == 0ULL) { inserted = true; break; }
            if (old == h) break; // already exists
            slot = (slot + 1) & hash_mask;
        }

        if (inserted) {
            uint32_t new_idx = atomicAdd(state_count, 1);
            if (new_idx < max_states) {
                uint8_t* dst = states_buf_w + (uint64_t)new_idx * state_size;
                for (int i = 0; i < state_size; i++) dst[i] = new_state[i];
                uint32_t pos = atomicAdd(next_count, 1);
                next_frontier[pos] = new_idx;
            }
        }
    }
}

// --------------------------------------------------------------------------
// Helper: next power of 2
// --------------------------------------------------------------------------
static uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// --------------------------------------------------------------------------
// Host BFS driver
// --------------------------------------------------------------------------
// Forward declaration (defined in main.cu)
struct BFSHistogram;

static int solve_general(int N, int max_depth, BFSHistogram& hist) {
    const int state_size = 6 * N * N;
    const int num_moves = 18;

    if (state_size > MAX_FACELET_STATE) {
        fprintf(stderr, "N=%d exceeds max supported (state_size=%d > %d)\n",
                N, state_size, MAX_FACELET_STATE);
        return 1;
    }

    printf("=== %dx%dx%d Rubik's Cube BFS (HTM", N, N, N);
    if (max_depth > 0) printf(", max_depth=%d", max_depth);
    printf(") ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d, %d SMs, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Generate and validate move tables
    std::vector<uint16_t> h_moves;
    generate_all_moves(N, h_moves);
    if (!validate_general_moves(N, h_moves)) {
        fprintf(stderr, "Move table validation FAILED.\n");
        return 1;
    }

    // Determine memory budget
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t budget = free_mem * 85 / 100;
    printf("GPU memory: %.2f GB free, using %.2f GB budget\n",
           free_mem / (1024.0*1024*1024), budget / (1024.0*1024*1024));

    // Compute max states
    // Per state: state_size (buffer) + ~32 (hash table at ~50%% load) + 8 (frontier)
    size_t per_state_cost = state_size + 40;
    uint32_t max_states = (uint32_t)std::min((size_t)500000000ULL, budget / per_state_cost);
    uint32_t ht_size = next_pow2(std::max(2u * max_states, 1024u));
    uint32_t ht_mask = ht_size - 1;

    // Recompute actual memory
    size_t states_bytes   = (size_t)max_states * state_size;
    size_t ht_bytes       = (size_t)ht_size * sizeof(uint64_t);
    size_t frontier_bytes = (size_t)max_states * sizeof(uint32_t);
    size_t moves_bytes    = h_moves.size() * sizeof(uint16_t);
    size_t total_alloc    = states_bytes + ht_bytes + 2 * frontier_bytes + moves_bytes + 256;

    // Shrink if needed
    while (total_alloc > budget && max_states > 1024) {
        max_states = max_states * 3 / 4;
        ht_size = next_pow2(2 * max_states);
        ht_mask = ht_size - 1;
        states_bytes = (size_t)max_states * state_size;
        ht_bytes = (size_t)ht_size * sizeof(uint64_t);
        frontier_bytes = (size_t)max_states * sizeof(uint32_t);
        total_alloc = states_bytes + ht_bytes + 2 * frontier_bytes + moves_bytes + 256;
    }

    printf("Max states: %u (%.2f GB state buf, %.2f GB hash table)\n",
           max_states,
           states_bytes / (1024.0*1024*1024),
           ht_bytes / (1024.0*1024*1024));
    printf("State size: %d bytes, Hash table: %u slots\n\n", state_size, ht_size);

    // Allocate GPU memory
    uint8_t*  d_states;
    uint64_t* d_hash_table;
    uint32_t* d_frontier_a;
    uint32_t* d_frontier_b;
    uint32_t* d_next_count;
    uint32_t* d_state_count;
    uint16_t* d_moves;

    cudaMalloc(&d_states,     states_bytes);
    cudaMalloc(&d_hash_table, ht_bytes);
    cudaMalloc(&d_frontier_a, frontier_bytes);
    cudaMalloc(&d_frontier_b, frontier_bytes);
    cudaMalloc(&d_next_count, sizeof(uint32_t));
    cudaMalloc(&d_state_count, sizeof(uint32_t));
    cudaMalloc(&d_moves,      moves_bytes);

    cudaMemset(d_hash_table, 0, ht_bytes);

    // Copy move tables
    cudaMemcpy(d_moves, h_moves.data(), moves_bytes, cudaMemcpyHostToDevice);

    // Build solved state: face f has all facelets = f
    std::vector<uint8_t> solved(state_size);
    for (int i = 0; i < state_size; i++) solved[i] = (uint8_t)(i / (N * N));

    // Insert solved state at index 0
    cudaMemcpy(d_states, solved.data(), state_size, cudaMemcpyHostToDevice);
    uint32_t one = 1;
    cudaMemcpy(d_state_count, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Insert solved hash into hash table
    uint64_t solved_hash;
    {
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < state_size; i++) {
            h ^= solved[i];
            h *= 1099511628211ULL;
        }
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        solved_hash = h | 1;
    }
    uint32_t slot = (uint32_t)(solved_hash & ht_mask);
    cudaMemcpy(&d_hash_table[slot], &solved_hash, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Seed frontier with index 0
    uint32_t zero = 0;
    cudaMemcpy(d_frontier_a, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t frontier_size = 1;
    uint64_t total = 1;
    int gods_number = 0;

    memset(hist.counts, 0, sizeof(hist.counts));
    hist.counts[0] = 1;
    hist.cube_size = N;

    printf("BFS from solved state:\n");
    printf("  Depth %2d: %12u states  (cumulative: %12llu)\n", 0, 1u, 1ULL);

    auto t_start = std::chrono::high_resolution_clock::now();
    const uint32_t BLOCK = 256;

    uint32_t* d_cur = d_frontier_a;
    uint32_t* d_nxt = d_frontier_b;

    for (int depth = 0; frontier_size > 0; depth++) {
        if (max_depth > 0 && depth >= max_depth) {
            printf("\n  Stopped at max_depth=%d\n", max_depth);
            break;
        }

        cudaMemset(d_next_count, 0, sizeof(uint32_t));

        uint32_t grid = (frontier_size + BLOCK - 1) / BLOCK;
        expand_frontier_general<<<grid, BLOCK>>>(
            d_states, d_cur, frontier_size,
            d_moves, num_moves, state_size,
            d_hash_table, ht_mask,
            d_states,
            d_nxt, d_next_count, d_state_count, max_states);
        cudaDeviceSynchronize();

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "\nCUDA error: %s\n", cudaGetErrorString(err));
            break;
        }

        cudaMemcpy(&frontier_size, d_next_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (frontier_size > 0) {
            total += frontier_size;
            gods_number = depth + 1;
            if (depth + 1 < 64) hist.counts[depth + 1] = frontier_size;
            printf("  Depth %2d: %12u states  (cumulative: %12llu)\n",
                   depth + 1, frontier_size, (unsigned long long)total);
        }

        // Check if we're running out of state buffer
        uint32_t current_states;
        cudaMemcpy(&current_states, d_state_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (current_states >= max_states) {
            printf("\n  Stopped: state buffer full (%u states)\n", current_states);
            // Last depth is truncated -- exclude from histogram for bounds analysis
            if (frontier_size > 0 && depth + 1 < 64) {
                hist.counts[depth + 1] = 0;
                gods_number = depth; // revert to last complete depth
            }
            break;
        }

        std::swap(d_cur, d_nxt);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    printf("\n=== Results ===\n");
    if (frontier_size == 0 && (max_depth <= 0 || gods_number < max_depth))
        printf("God's number (HTM): %d\n", gods_number);
    else
        printf("Explored to depth:  %d\n", gods_number);
    printf("Total states:       %llu\n", (unsigned long long)total);
    printf("BFS time:           %.3f seconds\n", elapsed);

    hist.depth_count = gods_number + 1;

    // Cleanup
    cudaFree(d_states);
    cudaFree(d_hash_table);
    cudaFree(d_frontier_a);
    cudaFree(d_frontier_b);
    cudaFree(d_next_count);
    cudaFree(d_state_count);
    cudaFree(d_moves);

    return 0;
}
