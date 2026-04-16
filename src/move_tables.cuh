#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "gpu_tables.cuh"

// ============================================================================
// 2x2x2 Rubik's Cube Move Tables (host-side initialization)
//
// Corner positions:
//   0=URF  1=UFL  2=ULB  3=UBR  4=DFR  5=DLF  6=DBL  7=DRB
//
// Corner 7 (DRB) is fixed as reference frame. After applying any move that
// displaces corner 7, we re-canonicalize via a whole-cube rotation.
//
// Orientation: 0 = U/D sticker on U/D face, 1 = CW twist, 2 = CCW twist.
// Invariant: sum of all 8 orientations ≡ 0 (mod 3).
//
// 18 HTM moves: U U2 U' R R2 R' F F2 F' D D2 D' L L2 L' B B2 B'
// ============================================================================

// Host-side tables (copied to __constant__ memory at init)
static uint8_t h_move_perm[NUM_MOVES][NUM_CORNERS];
static uint8_t h_move_orient[NUM_MOVES][NUM_CORNERS];
static uint8_t h_recanon_perm[NUM_ROTATIONS][NUM_CORNERS];
static uint8_t h_recanon_orient[NUM_ROTATIONS][NUM_CORNERS];

// Inverse move lookup
static uint8_t h_inv_move[NUM_MOVES];

// ----------------------------------------------------------------------------
// Helper: compose two moves (source-convention permutations + orientations)
// result[i] = apply B to the output of A at position i
// new_perm[i] = a_perm[b_perm[i]]  (B tells where to get from A's result)
// Wait -- source convention: perm[i] = "position i gets piece from position perm[i]"
// Compose A then B: first apply A, then apply B.
//   After A: pos i has piece from A.perm[i] with orient A.orient[i]
//   After B on top of A: pos i gets piece from pos B.perm[i] in A's result
//     = piece originally from A.perm[B.perm[i]] with orient A.orient[B.perm[i]] + B.orient[i]
// So: result.perm[i] = A.perm[B.perm[i]]
//     result.orient[i] = (A.orient[B.perm[i]] + B.orient[i]) % 3
// ----------------------------------------------------------------------------
static void compose_move(
    const uint8_t a_perm[8], const uint8_t a_orient[8],
    const uint8_t b_perm[8], const uint8_t b_orient[8],
    uint8_t out_perm[8], uint8_t out_orient[8])
{
    for (int i = 0; i < 8; i++) {
        out_perm[i] = a_perm[b_perm[i]];
        out_orient[i] = (a_orient[b_perm[i]] + b_orient[i]) % 3;
    }
}

// ----------------------------------------------------------------------------
// Define the 6 base face turns (90° CW looking at face)
// Each is a 4-cycle in source-convention + orientation deltas
// ----------------------------------------------------------------------------
struct BaseTurn {
    uint8_t perm[8];
    uint8_t orient[8];
};

// Cycle notation -> source permutation:
// Cycle (a b c d) means a->b->c->d->a (piece at a goes to b).
// Source: new pos b gets old pos a. So perm[b]=a, perm[c]=b, perm[d]=c, perm[a]=d.
//
// U CW: cycle (0 3 2 1) means 0->3, 3->2, 2->1, 1->0
//   Source: perm[3]=0, perm[2]=3, perm[1]=2, perm[0]=1
static constexpr BaseTurn BASE_TURNS[6] = {
    // U: cycle (0 3 2 1), no orientation change
    {{1, 2, 3, 0,  4, 5, 6, 7}, {0, 0, 0, 0,  0, 0, 0, 0}},
    // R: cycle (0 4 7 3) means 0->4, 4->7, 7->3, 3->0
    //   Source: perm[4]=0, perm[7]=4, perm[3]=7, perm[0]=3
    {{3, 1, 2, 7,  0, 5, 6, 4}, {2, 0, 0, 1,  1, 0, 0, 2}},
    // F: cycle (0 1 5 4) means 0->1, 1->5, 5->4, 4->0
    //   Source: perm[1]=0, perm[5]=1, perm[4]=5, perm[0]=4
    {{4, 0, 2, 3,  5, 1, 6, 7}, {1, 2, 0, 0,  2, 1, 0, 0}},
    // D: cycle (4 5 6 7) means 4->5, 5->6, 6->7, 7->4
    //   Source: perm[5]=4, perm[6]=5, perm[7]=6, perm[4]=7
    {{0, 1, 2, 3,  7, 4, 5, 6}, {0, 0, 0, 0,  0, 0, 0, 0}},
    // L: cycle (1 2 6 5) means 1->2, 2->6, 6->5, 5->1
    //   Source: perm[2]=1, perm[6]=2, perm[5]=6, perm[1]=5
    {{0, 5, 1, 3,  4, 6, 2, 7}, {0, 1, 2, 0,  0, 2, 1, 0}},
    // B: cycle (2 3 7 6) means 2->3, 3->7, 7->6, 6->2
    //   Source: perm[3]=2, perm[7]=3, perm[6]=7, perm[2]=6
    {{0, 1, 6, 2,  4, 5, 7, 3}, {0, 0, 1, 2,  0, 0, 2, 1}},
};

// Move ordering: U U2 U' R R2 R' F F2 F' D D2 D' L L2 L' B B2 B'
// Index: base_face * 3 + {0=CW, 1=180, 2=CCW}
// Base faces: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B

static void build_move_tables() {
    for (int face = 0; face < 6; face++) {
        const BaseTurn& base = BASE_TURNS[face];

        // CW (90°) = base turn applied once
        memcpy(h_move_perm[face*3 + 0], base.perm, 8);
        memcpy(h_move_orient[face*3 + 0], base.orient, 8);

        // 180° = base composed with itself
        compose_move(base.perm, base.orient, base.perm, base.orient,
                     h_move_perm[face*3 + 1], h_move_orient[face*3 + 1]);

        // CCW (270°) = 180° composed with base
        compose_move(h_move_perm[face*3 + 1], h_move_orient[face*3 + 1],
                     base.perm, base.orient,
                     h_move_perm[face*3 + 2], h_move_orient[face*3 + 2]);
    }

    // Build inverse move lookup
    // CW inverse is CCW, CCW inverse is CW, 180 inverse is 180
    for (int face = 0; face < 6; face++) {
        h_inv_move[face*3 + 0] = face*3 + 2; // CW -> CCW
        h_inv_move[face*3 + 1] = face*3 + 1; // 180 -> 180
        h_inv_move[face*3 + 2] = face*3 + 0; // CCW -> CW
    }
}

// ----------------------------------------------------------------------------
// Build the 24 whole-cube rotation tables for re-canonicalization.
//
// We generate all 24 rotations by composing basic x, y, z rotations.
// Then for each rotation R, we check where piece 7 ends up (position p,
// orientation o) and store R as recanon[p*3+o].
//
// When piece 7 is found at (p, o) after a move, we apply recanon[p*3+o]
// to bring piece 7 back to position 7 with orientation 0.
// ----------------------------------------------------------------------------

// Basic whole-cube rotations (x = rotate around R-L axis so F->U,
// y = rotate around U-D axis so F->L, z = rotate around F-B axis so U->R)
// These are permutations of corner positions + orientation adjustments.

// x rotation (R-face stays, F->U->B->D->F when looking from R):
//   Positions: URF->UBR->ULB->UFL cycle in one layer... actually let me
//   think about this as "the whole cube rotates so that the F face goes to U".
//   x CW (looking from R side): U->B, F->U, D->F, B->D
//   Corner mapping:
//     URF(0)->UBR(3), UFL(1)->URF(0), ULB(2)->UFL(1), UBR(3)->ULB(2)  -- but wait
//     Actually: x rotates so that F->U. Corner URF: its U sticker goes to B, its R stays, its F goes to U.
//     Position-wise: URF becomes... the corner that was at URF goes to the DFR position? No.
//
// This gets complicated. Let me use a different approach: generate all 24
// rotations by composition of face-based whole-cube rotations.
//
// A whole-cube x rotation is equivalent to applying R, then L', then adjusting
// for the fact that it's a cube rotation, not a slice move. For a 2x2x2,
// whole-cube x = R * L' (since there are no middle slices).
// Similarly: y = U * D', z = F * B'
//
// But these are face moves that include orientation changes. For whole-cube
// rotations, the ORIENTATION CONVENTION changes because what counts as
// "the U/D sticker" depends on the frame. So this approach is tricky.
//
// Simpler approach: enumerate all 24 rotations explicitly.
// A rotation is defined by which original face goes to U (6 choices) and
// which goes to F (4 choices per U-face choice) = 24 total.
//
// I'll generate them by tracking where each corner position maps to and
// what happens to its orientation.

struct Rotation {
    uint8_t perm[8];   // source permutation: new_pos[i] gets piece from perm[i]
    uint8_t orient[8]; // orientation adjustment at each position
};

static void generate_rotations(Rotation rots[24]) {
    // Start with identity
    Rotation id;
    for (int i = 0; i < 8; i++) { id.perm[i] = i; id.orient[i] = 0; }

    // Basic rotations using face-move compositions:
    // For a 2x2x2, whole-cube rotation x = R * L_inv as permutations.
    // But we need to be careful about orientation convention.
    //
    // Actually, let's just define the 3 basic rotations directly.

    // y rotation (around U-D axis, U stays, F->R when looking from top):
    //   Position mapping: URF(0)->UBR(3), UFL(1)->URF(0), ULB(2)->UFL(1), UBR(3)->ULB(2)
    //                     DFR(4)->DRB(7), DLF(5)->DFR(4), DBL(6)->DLF(5), DRB(7)->DBL(6)
    //   This is a CW rotation when viewed from above: F->R->B->L->F
    //   Source perm: pos 3 gets from 0, pos 0 gets from 1, pos 1 gets from 2, pos 2 gets from 3
    //               pos 7 gets from 4, pos 4 gets from 5, pos 5 gets from 6, pos 6 gets from 7
    //   No orientation change (U/D stickers stay on U/D faces)
    Rotation rot_y;
    rot_y.perm[0] = 1; rot_y.perm[1] = 2; rot_y.perm[2] = 3; rot_y.perm[3] = 0;
    rot_y.perm[4] = 5; rot_y.perm[5] = 6; rot_y.perm[6] = 7; rot_y.perm[7] = 4;
    for (int i = 0; i < 8; i++) rot_y.orient[i] = 0;

    // x rotation (around R-L axis, R stays, U->F when looking from right):
    //   F goes up, U goes back, B goes down, D goes front
    //   URF(0)->DFR(4), UFL(1)->UFL... no.
    //   Let me think position by position.
    //   After x rotation (F->U): the position that was URF is now...
    //   the U face becomes B, F becomes U, D becomes F, B becomes D.
    //   So the corner at physical position "URF" now corresponds to what was at "FDR"=DFR.
    //   URF <- DFR(4), UBR <- URF(0), DFR <- DRB(7), DRB <- UBR(3)
    //   UFL <- DLF(5), ULB <- UFL(1), DLF <- DBL(6), DBL <- ULB(2)
    //   Source: pos 0 gets from 4, pos 3 gets from 0, pos 4 gets from 7, pos 7 gets from 3
    //           pos 1 gets from 5, pos 2 gets from 1, pos 5 gets from 6, pos 6 gets from 2
    //   Orientation: corners that move between U and D layers get twisted.
    //   The U/D sticker reference: after x rotation, what was the F sticker is now on U.
    //   Corners moving from D-layer to U-layer or vice versa: their U/D reference sticker
    //   moves off the U/D face. This is equivalent to a twist.
    //   For x rotation: all corners get orientation change.
    //   Actually for whole-cube rotation, if we redefine the reference frame,
    //   orientations DON'T change (the whole cube rotates including our reference).
    //   But our orientation is defined relative to the U/D face of the FIXED frame
    //   (the frame where DRB is always at position 7). So whole-cube rotation
    //   DOES change orientations in our encoding.
    //
    //   For x rotation (F->U): the U/D axis rotates. A corner's "U/D sticker"
    //   in the old frame becomes an F/B sticker in the new frame. So orientation changes.
    //   Specifically, for the x rotation:
    //   Corners at positions that cycle between U and D layers: +1 or +2 twist.
    //
    //   This is equivalent to R * L' as face moves:
    Rotation rot_x;
    // R move: perm = {3,1,2,7,0,5,6,4}, orient = {2,0,0,1,1,0,0,2}
    // L' move: L is {0,5,1,3,4,6,2,7} with orient {0,1,2,0,0,2,1,0}
    // L' = L composed 3 times. Let me compute L' directly.
    // L CCW = L^3. L perm = {0,5,1,3,4,6,2,7}, orient={0,1,2,0,0,2,1,0}
    // L^2: compose L with L
    uint8_t l_p[8] = {0,5,1,3,4,6,2,7};
    uint8_t l_o[8] = {0,1,2,0,0,2,1,0};
    uint8_t l2_p[8], l2_o[8], l3_p[8], l3_o[8];
    compose_move(l_p, l_o, l_p, l_o, l2_p, l2_o);
    compose_move(l2_p, l2_o, l_p, l_o, l3_p, l3_o); // L' = L^3

    // x = R * L': apply L' first, then R
    uint8_t r_p[8] = {3,1,2,7,0,5,6,4};
    uint8_t r_o[8] = {2,0,0,1,1,0,0,2};
    // compose: first L', then R. compose_move(A, B) = A then B means
    // result.perm[i] = A.perm[B.perm[i]]. We want L' then R.
    // So A=L', B=R: result[i] = L'_perm[R_perm[i]]
    // Wait, the compose_move function: compose(a, b) means "apply a first, then b"?
    // Let me re-check: compose_move(a_perm, a_orient, b_perm, b_orient, ...)
    // result.perm[i] = a.perm[b.perm[i]]
    // This means: first apply b (to determine source), then look up in a.
    // Actually no: in source convention, state after applying perm p to state s:
    //   new_state[i] = old_state[p[i]]
    // Applying a then b: after a, state[i] = original[a.perm[i]]
    //                     after b on that, state[i] = (after a)[b.perm[i]] = original[a.perm[b.perm[i]]]
    // So compose(a, b) means "a first, then b". result.perm[i] = a.perm[b.perm[i]]
    // For x = L' then R: compose(L', R)
    // NO wait, I defined it wrong above in compose_move.
    // Let me re-read: compose_move says result.perm[i] = A.perm[B.perm[i]]
    // "apply B to the output of A" -- this is confusing. Let me reclarify.
    //
    // In our compose_move: result = "apply A, then apply B"
    //   After A: pos i has piece from A.perm[i]
    //   After B on A's result: pos i gets from pos B.perm[i] in A's result
    //     = piece from A.perm[B.perm[i]]
    // So compose_move(A, B) = A then B. result.perm[i] = A.perm[B.perm[i]].
    // Correct.
    //
    // For x = L' then R: compose_move(L', R) = L' then R. No.
    // Wait. x rotation = doing L' AND R simultaneously (both layers, whole cube).
    // On a 2x2x2 there are only 2 layers. x = R applied to R layer + L' applied to L layer.
    // Since the layers don't overlap, this is just: for positions affected by R, apply R;
    // for positions affected by L', apply L'.
    //
    // R affects positions {0, 3, 4, 7}. L affects positions {1, 2, 5, 6}.
    // These are disjoint and cover all 8 positions. So:
    for (int i = 0; i < 8; i++) {
        // R affects 0,3,4,7; L affects 1,2,5,6
        if (i == 0 || i == 3 || i == 4 || i == 7) {
            rot_x.perm[i] = r_p[i];
            rot_x.orient[i] = r_o[i];
        } else {
            rot_x.perm[i] = l3_p[i]; // L' = L^3
            rot_x.orient[i] = l3_o[i];
        }
    }

    // z rotation (around F-B axis, F stays, U->R when looking from front):
    //   z = F * B': F affects {0,1,4,5}, B affects {2,3,6,7}
    uint8_t f_p[8] = {4,0,2,3,5,1,6,7};
    uint8_t f_o[8] = {1,2,0,0,2,1,0,0};
    uint8_t b_p[8] = {0,1,6,2,4,5,7,3};
    uint8_t b_o[8] = {0,0,1,2,0,0,2,1};
    // B' = B^3
    uint8_t b2_p[8], b2_o[8], b3_p[8], b3_o[8];
    compose_move(b_p, b_o, b_p, b_o, b2_p, b2_o);
    compose_move(b2_p, b2_o, b_p, b_o, b3_p, b3_o);

    Rotation rot_z;
    for (int i = 0; i < 8; i++) {
        if (i == 0 || i == 1 || i == 4 || i == 5) {
            rot_z.perm[i] = f_p[i];
            rot_z.orient[i] = f_o[i];
        } else {
            rot_z.perm[i] = b3_p[i];
            rot_z.orient[i] = b3_o[i];
        }
    }

    // Generate all 24 rotations by composing x, y, z:
    // We'll BFS through compositions of x, y, z to find all 24 unique rotations.
    int count = 0;
    bool used[8][3]; // track which (position_of_7, orient_of_7) we've seen
    memset(used, 0, sizeof(used));

    // Queue for BFS over rotation group
    Rotation queue[24];
    queue[0] = id;
    count = 1;
    used[7][0] = true; // identity: piece 7 at position 7 with orient 0

    // Store in rots indexed by (pos*3 + orient) of piece 7
    rots[7*3 + 0] = id;

    Rotation generators[6]; // x, x^2, x^3, y, y^2, y^3... actually just x, y, z and inverses
    // Use x, x', y, y', z, z'
    // x' = x^3
    Rotation rot_x2, rot_x3, rot_y2, rot_y3, rot_z2, rot_z3;
    compose_move(rot_x.perm, rot_x.orient, rot_x.perm, rot_x.orient, rot_x2.perm, rot_x2.orient);
    compose_move(rot_x2.perm, rot_x2.orient, rot_x.perm, rot_x.orient, rot_x3.perm, rot_x3.orient);
    compose_move(rot_y.perm, rot_y.orient, rot_y.perm, rot_y.orient, rot_y2.perm, rot_y2.orient);
    compose_move(rot_y2.perm, rot_y2.orient, rot_y.perm, rot_y.orient, rot_y3.perm, rot_y3.orient);
    compose_move(rot_z.perm, rot_z.orient, rot_z.perm, rot_z.orient, rot_z2.perm, rot_z2.orient);
    compose_move(rot_z2.perm, rot_z2.orient, rot_z.perm, rot_z.orient, rot_z3.perm, rot_z3.orient);

    generators[0] = rot_x;
    generators[1] = rot_x3; // x'
    generators[2] = rot_y;
    generators[3] = rot_y3; // y'
    generators[4] = rot_z;
    generators[5] = rot_z3; // z'

    int head = 0;
    while (head < count && count < 24) {
        Rotation cur = queue[head++];
        for (int g = 0; g < 6; g++) {
            Rotation next;
            compose_move(cur.perm, cur.orient, generators[g].perm, generators[g].orient,
                         next.perm, next.orient);
            // Find where piece 7 ends up
            int pos7 = -1, ori7 = -1;
            for (int i = 0; i < 8; i++) {
                if (next.perm[i] == 7) {
                    // In source convention: position i gets piece from position perm[i].
                    // Wait -- perm[i] is the SOURCE position. The piece that was originally
                    // at position perm[i] goes to position i. So if we start with identity
                    // (piece j at position j), after applying rotation, position i has
                    // piece next.perm[i]. So if next.perm[i] == 7, piece 7 is NOT at
                    // position i. Rather, position i has the piece that was originally at
                    // position 7... no.
                    //
                    // Source convention: new[i] = old[perm[i]].
                    // In identity state, old[j] = piece j at position j.
                    // After rotation: new[i] = piece perm[i]. So position i has piece perm[i].
                    // We want: which position has piece 7? It's the position i where perm[i] == 7.
                    pos7 = i;
                    ori7 = next.orient[i];
                    break;
                }
            }
            if (pos7 >= 0 && !used[pos7][ori7]) {
                used[pos7][ori7] = true;
                queue[count++] = next;
                rots[pos7 * 3 + ori7] = next;
            }
        }
    }

    if (count != 24) {
        fprintf(stderr, "ERROR: Generated %d rotations, expected 24\n", count);
    }

    // Now rots[pos7*3 + ori7] = the rotation that PRODUCES piece 7 at (pos7, ori7).
    // For re-canonicalization, we need the INVERSE: when piece 7 is at (pos7, ori7),
    // apply the inverse rotation to move it back to (7, 0).
    // Inverse of a source-perm: if perm[i] = j, then inv_perm[j] = i.
    // Inverse orient: inv_orient[j] = (3 - orient[inv_perm[j]]) % 3... actually:
    // For rotation R: new[i] = old[R.perm[i]] with orient R.orient[i]
    // Inverse R^-1: old[j] = new[R^-1.perm[j]] with orient R^-1.orient[j]
    // R^-1.perm: if R.perm[i] = j, then R^-1.perm[j] = i
    // R^-1.orient[j] = (3 - R.orient[i]) % 3 where i = R^-1.perm[j]...
    // Let me just compute: for each rotation, find its inverse by composing until identity.
    // Or: inverse perm[j] = i where perm[i] = j.
    //     inverse orient[j] = (3 - orient[i]) % 3 where perm[i] = j.

    Rotation inv_rots[24];
    for (int r = 0; r < 24; r++) {
        // Compute inverse
        for (int j = 0; j < 8; j++) {
            for (int i = 0; i < 8; i++) {
                if (rots[r].perm[i] == j) {
                    inv_rots[r].perm[j] = i;
                    inv_rots[r].orient[j] = (3 - rots[r].orient[i]) % 3;
                    break;
                }
            }
        }
    }

    // Verify: compose(rot, inv_rot) should give identity
    for (int r = 0; r < 24; r++) {
        uint8_t check_p[8], check_o[8];
        compose_move(rots[r].perm, rots[r].orient,
                     inv_rots[r].perm, inv_rots[r].orient,
                     check_p, check_o);
        for (int i = 0; i < 8; i++) {
            if (check_p[i] != i || check_o[i] != 0) {
                fprintf(stderr, "ERROR: Rotation %d inverse verification failed\n", r);
                break;
            }
        }
    }

    // Store inverse rotations as the re-canonicalization tables
    for (int r = 0; r < 24; r++) {
        memcpy(h_recanon_perm[r], inv_rots[r].perm, 8);
        memcpy(h_recanon_orient[r], inv_rots[r].orient, 8);
    }
}

// ----------------------------------------------------------------------------
// Copy tables to GPU constant memory
// ----------------------------------------------------------------------------
static void copy_tables_to_device() {
    cudaMemcpyToSymbol(d_move_perm, h_move_perm, sizeof(h_move_perm));
    cudaMemcpyToSymbol(d_move_orient, h_move_orient, sizeof(h_move_orient));
    cudaMemcpyToSymbol(d_recanon_perm, h_recanon_perm, sizeof(h_recanon_perm));
    cudaMemcpyToSymbol(d_recanon_orient, h_recanon_orient, sizeof(h_recanon_orient));
    cudaMemcpyToSymbol(d_inv_move, h_inv_move, sizeof(h_inv_move));
}

// ----------------------------------------------------------------------------
// Host-side validation
// ----------------------------------------------------------------------------
static void host_apply_move(const uint8_t perm[8], const uint8_t orient[8],
                            int move_id,
                            uint8_t out_perm[8], uint8_t out_orient[8])
{
    for (int i = 0; i < 8; i++) {
        out_perm[i] = perm[h_move_perm[move_id][i]];
        out_orient[i] = (orient[h_move_perm[move_id][i]] + h_move_orient[move_id][i]) % 3;
    }
}

static void host_recanon(uint8_t perm[8], uint8_t orient[8]) {
    // Find piece 7
    int pos7 = -1;
    for (int i = 0; i < 8; i++) {
        if (perm[i] == 7) { pos7 = i; break; }
    }
    int ori7 = orient[pos7];
    int rot_idx = pos7 * 3 + ori7;

    uint8_t new_perm[8], new_orient[8];
    // Apply re-canonicalization rotation
    for (int i = 0; i < 8; i++) {
        new_perm[i] = perm[h_recanon_perm[rot_idx][i]];
        new_orient[i] = (orient[h_recanon_perm[rot_idx][i]] + h_recanon_orient[rot_idx][i]) % 3;
    }
    memcpy(perm, new_perm, 8);
    memcpy(orient, new_orient, 8);
}

// Encode state on host
static uint32_t host_perm_to_lehmer(const uint8_t perm[7]) {
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

static void host_lehmer_to_perm(uint32_t code, uint8_t perm[7]) {
    uint8_t avail[7] = {0,1,2,3,4,5,6};
    for (int i = 0; i < 7; i++) {
        uint32_t fact = 1;
        for (int k = 1; k <= 6 - i; k++) fact *= k;
        uint32_t idx = code / fact;
        code %= fact;
        perm[i] = avail[idx];
        for (int j = idx; j < 6 - i; j++) avail[j] = avail[j+1];
    }
}

static uint32_t host_encode(const uint8_t perm[8], const uint8_t orient[8]) {
    uint32_t perm_idx = host_perm_to_lehmer(perm); // only first 7
    uint32_t orient_idx = 0;
    for (int i = 0; i < 6; i++) {
        orient_idx = orient_idx * 3 + orient[i];
    }
    return perm_idx * 729 + orient_idx;
}

static bool validate_move_tables() {
    printf("Validating move tables...\n");

    // Test 1: Lehmer round-trip for all 5040 permutations
    for (uint32_t code = 0; code < 5040; code++) {
        uint8_t perm[7];
        host_lehmer_to_perm(code, perm);
        uint32_t code2 = host_perm_to_lehmer(perm);
        if (code != code2) {
            fprintf(stderr, "Lehmer round-trip failed: %u -> %u\n", code, code2);
            return false;
        }
    }
    printf("  [OK] Lehmer round-trip for all 5040 permutations\n");

    // Test 2: Move-inverse identity from solved state
    uint8_t solved_p[8] = {0,1,2,3,4,5,6,7};
    uint8_t solved_o[8] = {0,0,0,0,0,0,0,0};

    for (int m = 0; m < 18; m++) {
        uint8_t p1[8], o1[8], p2[8], o2[8];
        host_apply_move(solved_p, solved_o, m, p1, o1);
        host_recanon(p1, o1);

        int inv_m = h_inv_move[m];
        host_apply_move(p1, o1, inv_m, p2, o2);
        host_recanon(p2, o2);

        // Check orientation parity after move
        int sum = 0;
        for (int i = 0; i < 8; i++) sum += o1[i];
        if (sum % 3 != 0) {
            fprintf(stderr, "Orientation parity violated after move %d\n", m);
            return false;
        }

        // Check piece 7 is at position 7 after recanon
        if (p1[7] != 7 || o1[7] != 0) {
            fprintf(stderr, "Recanon failed for move %d: piece7 at pos %d orient %d\n",
                    m, -1, -1);
            // Find piece 7
            for (int i = 0; i < 8; i++) {
                if (p1[i] == 7) {
                    fprintf(stderr, "  piece 7 at position %d orient %d\n", i, o1[i]);
                    break;
                }
            }
            return false;
        }

        // Check inverse gives back solved
        uint32_t enc = host_encode(p2, o2);
        if (enc != 0) {
            fprintf(stderr, "Move %d then inverse %d did not return to solved (got %u)\n",
                    m, inv_m, enc);
            return false;
        }
    }
    printf("  [OK] All 18 move-inverse pairs return to solved\n");

    // Test 3: Count unique depth-1 neighbors from solved
    uint32_t neighbors[18];
    int unique = 0;
    for (int m = 0; m < 18; m++) {
        uint8_t p1[8], o1[8];
        host_apply_move(solved_p, solved_o, m, p1, o1);
        host_recanon(p1, o1);
        neighbors[m] = host_encode(p1, o1);
        bool dup = false;
        for (int j = 0; j < m; j++) {
            if (neighbors[j] == neighbors[m]) { dup = true; break; }
        }
        if (!dup) unique++;
    }
    if (unique != 9) {
        fprintf(stderr, "Expected 9 unique depth-1 neighbors, got %d\n", unique);
        return false;
    }
    printf("  [OK] 9 unique depth-1 neighbors from solved\n");

    // Test 4: Move-inverse for a chain of random-ish moves
    uint8_t p[8], o[8];
    memcpy(p, solved_p, 8);
    memcpy(o, solved_o, 8);
    int move_seq[] = {0, 3, 6, 9, 12, 15, 1, 4, 7, 10};
    int n_moves = 10;
    for (int i = 0; i < n_moves; i++) {
        uint8_t np[8], no_[8];
        host_apply_move(p, o, move_seq[i], np, no_);
        host_recanon(np, no_);
        memcpy(p, np, 8);
        memcpy(o, no_, 8);
        // Check orientation parity
        int sum = 0;
        for (int j = 0; j < 8; j++) sum += o[j];
        if (sum % 3 != 0) {
            fprintf(stderr, "Orientation parity violated after %d moves\n", i+1);
            return false;
        }
    }
    // Apply inverses in reverse
    for (int i = n_moves - 1; i >= 0; i--) {
        uint8_t np[8], no_[8];
        host_apply_move(p, o, h_inv_move[move_seq[i]], np, no_);
        host_recanon(np, no_);
        memcpy(p, np, 8);
        memcpy(o, no_, 8);
    }
    if (host_encode(p, o) != 0) {
        fprintf(stderr, "Move sequence + inverse did not return to solved\n");
        return false;
    }
    printf("  [OK] 10-move sequence + reverse returns to solved\n");

    printf("All validations passed.\n\n");
    return true;
}

// Master initialization function
static bool init_move_tables() {
    build_move_tables();
    Rotation rots[24];
    generate_rotations(rots);
    copy_tables_to_device();
    return validate_move_tables();
}
