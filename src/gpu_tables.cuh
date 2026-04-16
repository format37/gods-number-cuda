#pragma once
#include <cstdint>

// Shared constants
static constexpr int NUM_MOVES = 18;
static constexpr int NUM_CORNERS = 8;
static constexpr int NUM_ROTATIONS = 24;

// Device-side constant memory (single translation unit)
__constant__ uint8_t d_move_perm[NUM_MOVES][NUM_CORNERS];
__constant__ uint8_t d_move_orient[NUM_MOVES][NUM_CORNERS];
__constant__ uint8_t d_recanon_perm[NUM_ROTATIONS][NUM_CORNERS];
__constant__ uint8_t d_recanon_orient[NUM_ROTATIONS][NUM_CORNERS];
__constant__ uint8_t d_inv_move[NUM_MOVES];
