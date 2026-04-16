#pragma once
#include <cstdint>
#include "gpu_tables.cuh"

// ============================================================================
// Device-side state encoding/decoding for the 2x2x2 Rubik's Cube
//
// State ID = perm_idx * 729 + orient_idx
//   perm_idx  ∈ [0, 5040)   — Lehmer code of 7-corner permutation
//   orient_idx ∈ [0, 729)   — base-3 encoding of 6 corner orientations
//   state_id  ∈ [0, 3674160)
// ============================================================================

static constexpr uint32_t NUM_STATES = 3674160;
static constexpr uint32_t NUM_PERMS = 5040;
static constexpr uint32_t NUM_ORIENTS = 729;
static constexpr uint32_t UNVISITED = 0xFFFFFFFF;

// Factorials for Lehmer decode
__device__ __forceinline__
uint32_t factorial(int n) {
    // Only need 0! through 6!
    constexpr uint32_t fact[7] = {1, 1, 2, 6, 24, 120, 720};
    return fact[n];
}

// Encode a 7-element permutation to Lehmer code [0, 5040)
__device__ __forceinline__
uint32_t perm_to_lehmer(const uint8_t perm[7]) {
    uint32_t code = 0;
    for (int i = 0; i < 7; i++) {
        int cnt = 0;
        for (int j = i + 1; j < 7; j++) {
            if (perm[j] < perm[i]) cnt++;
        }
        code = code * (7 - i) + cnt;
    }
    return code;
}

// Decode Lehmer code to 7-element permutation
__device__ __forceinline__
void lehmer_to_perm(uint32_t code, uint8_t perm[7]) {
    uint8_t avail[7] = {0, 1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 7; i++) {
        uint32_t f = factorial(6 - i);
        uint32_t idx = code / f;
        code %= f;
        perm[i] = avail[idx];
        // Shift remaining elements left
        for (int j = idx; j < 6 - i; j++) {
            avail[j] = avail[j + 1];
        }
    }
}

// Encode 6 orientation values (0-2) to index [0, 729)
__device__ __forceinline__
uint32_t orient_to_idx(const uint8_t orient[8]) {
    uint32_t idx = 0;
    for (int i = 0; i < 6; i++) {
        idx = idx * 3 + orient[i];
    }
    return idx;
}

// Decode orientation index to 8 orientation values (7th from parity, 8th = 0)
__device__ __forceinline__
void idx_to_orient(uint32_t idx, uint8_t orient[8]) {
    int sum = 0;
    for (int i = 5; i >= 0; i--) {
        orient[i] = idx % 3;
        idx /= 3;
        sum += orient[i];
    }
    orient[6] = (3 - (sum % 3)) % 3;
    orient[7] = 0; // fixed corner always orientation 0
}

// Encode full 8-corner state to state_id [0, 3674160)
// Assumes piece 7 is at position 7 with orientation 0 (canonical form)
__device__ __forceinline__
uint32_t encode_state(const uint8_t perm[8], const uint8_t orient[8]) {
    return perm_to_lehmer(perm) * NUM_ORIENTS + orient_to_idx(orient);
}

// Decode state_id to full 8-corner state
__device__ __forceinline__
void decode_state(uint32_t state_id, uint8_t perm[8], uint8_t orient[8]) {
    uint32_t perm_idx = state_id / NUM_ORIENTS;
    uint32_t orient_idx = state_id % NUM_ORIENTS;
    lehmer_to_perm(perm_idx, perm);
    perm[7] = 7; // fixed corner
    idx_to_orient(orient_idx, orient);
}

// Apply a move to an 8-corner state using the move tables in constant memory
__device__ __forceinline__
void apply_move(const uint8_t perm[8], const uint8_t orient[8],
                int move_id,
                uint8_t new_perm[8], uint8_t new_orient[8]) {
    for (int i = 0; i < 8; i++) {
        uint8_t src = d_move_perm[move_id][i];
        new_perm[i] = perm[src];
        new_orient[i] = (orient[src] + d_move_orient[move_id][i]) % 3;
    }
}

// Re-canonicalize: move piece 7 back to position 7 with orientation 0
// by applying the appropriate whole-cube rotation inverse
__device__ __forceinline__
void re_canonicalize(uint8_t perm[8], uint8_t orient[8]) {
    // Find where piece 7 is
    int pos7 = 7; // optimistic default
    for (int i = 0; i < 8; i++) {
        if (perm[i] == 7) { pos7 = i; break; }
    }

    // If already canonical, skip
    if (pos7 == 7 && orient[7] == 0) return;

    int rot_idx = pos7 * 3 + orient[pos7];

    uint8_t new_perm[8], new_orient[8];
    for (int i = 0; i < 8; i++) {
        uint8_t src = d_recanon_perm[rot_idx][i];
        new_perm[i] = perm[src];
        new_orient[i] = (orient[src] + d_recanon_orient[rot_idx][i]) % 3;
    }

    for (int i = 0; i < 8; i++) {
        perm[i] = new_perm[i];
        orient[i] = new_orient[i];
    }
}
