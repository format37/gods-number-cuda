#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <ctime>

// ============================================================================
// Diameter Lower-Bound Estimation Module
//
// Post-processes BFS depth histogram to compute:
//   1. Proven lower bound on Cayley-graph diameter
//   2. Effective branching factor per depth
//   3. Saturation onset detection
//   4. Extrapolated lower bound (non-proven, constant b_eff assumption)
//
// See docs/diameter_lower_bound_tech_spec.md for methodology.
// ============================================================================

static constexpr int MAX_HIST_DEPTH = 64;

// --------------------------------------------------------------------------
// Group order table (log10 values for N=2..10, outer-face HTM)
//
// Sources:
//   N=2: 3,674,160  (exact)
//   N=3: 43,252,003,274,489,856,000  (exact)
//   N=4..10: Jaap Scherphuis group-order formulas
// --------------------------------------------------------------------------
struct GroupOrderEntry {
    int n;
    double log10_order;     // log10(|G|)
    const char* exact_str;  // human-readable
};

static const GroupOrderEntry GROUP_ORDERS[] = {
    { 2,   6.5651,  "3,674,160"},
    { 3,  19.6361,  "4.33e19"},
    { 4,  45.8694,  "7.40e45"},
    { 5,  74.4515,  "2.83e74"},
    { 6, 116.1962,  "1.57e116"},
    { 7, 160.2900,  "1.95e160"},
    { 8, 217.5432,  "3.49e217"},
    { 9, 277.1457,  "1.40e277"},
    {10, 349.9076,  "8.07e349"},
};
static const int NUM_GROUP_ORDERS = sizeof(GROUP_ORDERS) / sizeof(GROUP_ORDERS[0]);

static double lookup_group_order_log10(int n) {
    for (int i = 0; i < NUM_GROUP_ORDERS; i++) {
        if (GROUP_ORDERS[i].n == n) return GROUP_ORDERS[i].log10_order;
    }
    return -1.0; // unknown
}

static const char* lookup_group_order_str(int n) {
    for (int i = 0; i < NUM_GROUP_ORDERS; i++) {
        if (GROUP_ORDERS[i].n == n) return GROUP_ORDERS[i].exact_str;
    }
    return "unknown";
}

// --------------------------------------------------------------------------
// Analysis results
// --------------------------------------------------------------------------
struct BoundsResult {
    int cube_size;
    int max_depth_measured;

    // Histogram
    uint64_t histogram[MAX_HIST_DEPTH];
    int histogram_len;

    // Group order
    double group_order_log10;
    const char* group_order_str;

    // Proven lower bound
    int proven_lower_bound;
    bool full_coverage;            // true if BFS covered entire group
    double coverage_fraction_log10; // log10(cumsum / |G|)

    // Branching factors
    double b_eff[MAX_HIST_DEPTH];  // b_eff[d] = N(d)/N(d-1), valid for d>=2
    double b_asymptotic;           // geometric mean over last half of depths
    int b_eff_count;               // number of valid b_eff entries

    // Saturation
    int saturation_onset;          // -1 if not observed
    double saturation_epsilon;

    // Extrapolated bound
    int extrapolated_lower_bound;
    double b_fit;                  // branching factor used for extrapolation
    bool extrapolation_capped;     // hit safety cap?
};

// --------------------------------------------------------------------------
// Core analysis
// --------------------------------------------------------------------------
static BoundsResult analyze_bounds(
    const uint64_t* histogram, int hist_len,
    int cube_size,
    double group_order_log10_override = -1.0)
{
    BoundsResult r;
    memset(&r, 0, sizeof(r));
    r.cube_size = cube_size;
    r.max_depth_measured = hist_len - 1;
    r.histogram_len = hist_len;
    r.saturation_onset = -1;
    r.saturation_epsilon = 0.01;

    for (int i = 0; i < hist_len && i < MAX_HIST_DEPTH; i++) {
        r.histogram[i] = histogram[i];
    }

    // Group order
    if (group_order_log10_override > 0) {
        r.group_order_log10 = group_order_log10_override;
        r.group_order_str = "user-supplied";
    } else {
        r.group_order_log10 = lookup_group_order_log10(cube_size);
        r.group_order_str = lookup_group_order_str(cube_size);
    }

    if (r.group_order_log10 < 0) {
        fprintf(stderr, "Warning: No group order for N=%d. "
                "Use --group-order-log10=<val> to supply one.\n", cube_size);
        r.proven_lower_bound = r.max_depth_measured;
        r.extrapolated_lower_bound = -1;
        return r;
    }

    // --- §5.1: Proven lower bound ---
    // Cumulative sum in log10 space. For N>=3, cumsum << |G|, so we just
    // compare log10(cumsum) vs log10(|G|).
    uint64_t cumsum = 0;
    r.full_coverage = false;
    r.proven_lower_bound = r.max_depth_measured + 1;

    for (int d = 0; d < hist_len; d++) {
        cumsum += histogram[d];
    }

    // For N=2, cumsum fits in uint64 and group order is small
    if (cube_size == 2) {
        uint64_t group_order_exact = 3674160ULL;
        if (cumsum >= group_order_exact) {
            r.full_coverage = true;
            // Find exact depth where coverage was reached
            uint64_t running = 0;
            for (int d = 0; d < hist_len; d++) {
                running += histogram[d];
                if (running >= group_order_exact) {
                    r.proven_lower_bound = d;
                    break;
                }
            }
        }
    }

    // Coverage fraction in log10
    if (cumsum > 0) {
        r.coverage_fraction_log10 = log10((double)cumsum) - r.group_order_log10;
    } else {
        r.coverage_fraction_log10 = -999.0;
    }

    // --- §5.2: Effective branching factor ---
    r.b_eff_count = 0;
    for (int d = 0; d < MAX_HIST_DEPTH; d++) r.b_eff[d] = 0.0;

    for (int d = 2; d < hist_len; d++) {
        if (histogram[d-1] > 0) {
            r.b_eff[d] = (double)histogram[d] / (double)histogram[d-1];
            r.b_eff_count = d;
        }
    }

    // --- §5.3: Asymptotic branching factor (geometric mean of last half) ---
    int half = (hist_len > 4) ? hist_len / 2 : 2;
    double log_sum = 0.0;
    int count = 0;
    for (int d = half; d < hist_len; d++) {
        if (r.b_eff[d] > 0) {
            log_sum += log(r.b_eff[d]);
            count++;
        }
    }
    r.b_asymptotic = (count > 0) ? exp(log_sum / count) : 13.0;

    // --- §5.3: Saturation detection ---
    for (int d = 2; d < hist_len; d++) {
        if (r.b_eff[d] > 0 && r.b_eff[d] / r.b_asymptotic < 1.0 - r.saturation_epsilon) {
            r.saturation_onset = d;
            break;
        }
    }

    // --- §5.4: Extrapolated lower bound ---
    // Use log-space arithmetic to avoid big-integer dependency.
    // cumsum_extrap(d) = measured_cumsum + N(D_max) * b * (b^(d-D_max) - 1) / (b-1)
    // For large d: ≈ N(D_max) * b^(d-D_max+1) / (b-1)
    // Solve: log10(cumsum_extrap) = group_order_log10
    //
    // Simplified: d_extra = D_max + ceil(
    //   (group_order_log10 + log10(b-1) - log10(N(D_max)) - log10(b)) / log10(b)
    // )

    r.b_fit = r.b_asymptotic;
    double log10_b = log10(r.b_fit);
    double log10_bm1 = log10(r.b_fit - 1.0);
    double log10_N_last = (histogram[hist_len-1] > 0)
        ? log10((double)histogram[hist_len-1])
        : 0.0;

    if (r.full_coverage) {
        // Already have exact answer
        r.extrapolated_lower_bound = r.proven_lower_bound;
    } else if (log10_b > 0 && histogram[hist_len-1] > 0) {
        double numerator = r.group_order_log10 + log10_bm1 - log10_N_last - log10_b;
        int extra_steps = (int)ceil(numerator / log10_b);

        // Safety cap: generous limit (log-space computation has no runaway risk)
        int max_extra = std::max(1000, 20 * r.max_depth_measured);
        if (extra_steps > max_extra) {
            extra_steps = max_extra;
            r.extrapolation_capped = true;
        }
        if (extra_steps < 1) extra_steps = 1;

        r.extrapolated_lower_bound = r.max_depth_measured + extra_steps;
    } else {
        r.extrapolated_lower_bound = -1;
    }

    return r;
}

// --------------------------------------------------------------------------
// Console output
// --------------------------------------------------------------------------
static void print_bounds_summary(const BoundsResult& r) {
    printf("\n=== Diameter Lower-Bound Analysis ===\n");
    printf("Cube size:      %dx%dx%d\n", r.cube_size, r.cube_size, r.cube_size);
    printf("Metric:         HTM (18 outer-face generators)\n");
    printf("Group order:    %s (log10 = %.4f)\n", r.group_order_str, r.group_order_log10);
    printf("BFS depth:      %d\n", r.max_depth_measured);

    // Branching factor table
    printf("\nBranching factor b_eff(d) = N(d)/N(d-1):\n");
    for (int d = 2; d < r.histogram_len && d < MAX_HIST_DEPTH; d++) {
        if (r.b_eff[d] > 0) {
            printf("  d=%2d: b_eff = %.5f\n", d, r.b_eff[d]);
        }
    }
    printf("  b_asymptotic = %.5f\n", r.b_asymptotic);

    // Saturation
    if (r.saturation_onset >= 0) {
        printf("  Saturation onset at d=%d\n", r.saturation_onset);
    } else {
        printf("  Saturation: not observed within measured depths\n");
    }

    // Proven bound
    printf("\nProven lower bound: %d", r.proven_lower_bound);
    if (r.full_coverage) {
        printf("  (full group coverage — this IS God's number)\n");
    } else {
        printf("  (BFS coverage = 10^%.1f of group)\n", r.coverage_fraction_log10);
    }

    // Extrapolated bound
    if (r.extrapolated_lower_bound > 0 && !r.full_coverage) {
        printf("Extrapolated lower bound: %d", r.extrapolated_lower_bound);
        printf("  (assuming constant b_eff = %.3f; NOT a proof)\n", r.b_fit);
        if (r.extrapolation_capped) {
            printf("  WARNING: extrapolation safety cap reached, actual bound may be higher\n");
        }
    }
    printf("\n");
}

// --------------------------------------------------------------------------
// JSON output
// --------------------------------------------------------------------------
static void write_bounds_json(const BoundsResult& r, FILE* fp) {
    fprintf(fp, "{\n");
    fprintf(fp, "  \"cube_size\": %d,\n", r.cube_size);
    fprintf(fp, "  \"metric\": \"HTM\",\n");
    fprintf(fp, "  \"generator_set\": \"outer_faces_18\",\n");
    fprintf(fp, "  \"group_order\": \"%s\",\n", r.group_order_str);
    fprintf(fp, "  \"group_order_log10\": %.4f,\n", r.group_order_log10);
    fprintf(fp, "  \"max_depth_measured\": %d,\n", r.max_depth_measured);

    // Histogram
    fprintf(fp, "  \"histogram\": [");
    for (int d = 0; d < r.histogram_len; d++) {
        if (d > 0) fprintf(fp, ", ");
        fprintf(fp, "%llu", (unsigned long long)r.histogram[d]);
    }
    fprintf(fp, "],\n");

    // Proven bound
    fprintf(fp, "  \"proven_lower_bound\": {\n");
    fprintf(fp, "    \"value\": %d,\n", r.proven_lower_bound);
    fprintf(fp, "    \"full_coverage\": %s,\n", r.full_coverage ? "true" : "false");
    fprintf(fp, "    \"coverage_log10\": %.2f\n", r.coverage_fraction_log10);
    fprintf(fp, "  },\n");

    // Extrapolated bound
    fprintf(fp, "  \"extrapolated_lower_bound\": {\n");
    if (r.extrapolated_lower_bound > 0 && !r.full_coverage) {
        fprintf(fp, "    \"value\": %d,\n", r.extrapolated_lower_bound);
        fprintf(fp, "    \"method\": \"constant b_eff extrapolation\",\n");
        fprintf(fp, "    \"b_fit\": %.5f,\n", r.b_fit);
        fprintf(fp, "    \"capped\": %s,\n", r.extrapolation_capped ? "true" : "false");
        fprintf(fp, "    \"disclaimer\": \"Not a proven bound. Assumes free-expansion continues.\"\n");
    } else if (r.full_coverage) {
        fprintf(fp, "    \"value\": %d,\n", r.proven_lower_bound);
        fprintf(fp, "    \"method\": \"exact (full BFS)\",\n");
        fprintf(fp, "    \"disclaimer\": \"Exact result from complete BFS.\"\n");
    } else {
        fprintf(fp, "    \"value\": null,\n");
        fprintf(fp, "    \"method\": \"unavailable\"\n");
    }
    fprintf(fp, "  },\n");

    // Graph stats
    fprintf(fp, "  \"graph_stats\": {\n");
    fprintf(fp, "    \"b_asymptotic\": %.5f,\n", r.b_asymptotic);
    fprintf(fp, "    \"b_eff_per_depth\": [");
    for (int d = 0; d < r.histogram_len; d++) {
        if (d > 0) fprintf(fp, ", ");
        if (d < 2 || r.b_eff[d] == 0.0)
            fprintf(fp, "null");
        else
            fprintf(fp, "%.5f", r.b_eff[d]);
    }
    fprintf(fp, "],\n");
    fprintf(fp, "    \"saturation_onset_depth\": ");
    if (r.saturation_onset >= 0)
        fprintf(fp, "%d\n", r.saturation_onset);
    else
        fprintf(fp, "null\n");
    fprintf(fp, "  },\n");

    // Timestamp
    time_t now = time(NULL);
    char timebuf[64];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S", localtime(&now));
    fprintf(fp, "  \"run_timestamp\": \"%s\"\n", timebuf);

    fprintf(fp, "}\n");
}
