// StereoForge v13 — 16x16 GPU Tile Classifier (`tile_classifier.slang`)
// Classifies each 16x16 target tile (4x4 coarse 1/4-scale pixels) across three independent dimensions:
//   1. Motion State (`bits 2..3`):
//      - 0 = Static
//      - 1 = DynamicForeground
//      - 2 = DisocclusionWake
//      - 3 = Unknown (never treated as Static)
//   2. Range State (`bits 4..6`):
//      - 0 = TooNear_Unknown (< Z_min or outside binocular overlap; never reported as free space)
//      - 1 = Near (0.25..1.0 m, d >= 63 px; prioritized for L2/L3 + extended 256-disparity search)
//      - 2 = Middle_Useful (1.0..5.0 m, d in [12.6, 63) px; best opportunity for L0/L1 work reduction)
//      - 3 = Far_Useful (5.0..Z_max(sigma_req); small-disparity full-res search)
//      - 4 = Far_BeyondAccuracy (> Z_max(sigma_req); skip optional fine/subpixel refinement)
//   3. Quality State (`bits 7..8`):
//      - 0 = Easy (smooth, well-supported coarse geometry)
//      - 1 = Detailed (ordinary textured surface)
//      - 2 = Uncertain (depth discontinuity, thin structure, or low coarse confidence)
// And selects one of the 4 Compute-Adaptive Processing Levels (`bits 0..1`):
//   - L0_Minimal   = 0: Validated coarse + 1-tap ±1 check (skips full-res 5-tap & continuous subpixel)
//   - L1_Standard  = 1: Guided full-resolution stereo (r=2, 5-tap, epipolar + range pruned)
//   - L2_Precision = 2: Expanded matching (r=3..4), strict L-R, and 5-point continuous subpixel
//   - L3_Recovery  = 3: Wider disparity search (r=6), multi-hypothesis & P3 disocclusion re-search

struct TileClassifierParameters {
    uint64_t coarse_disp;
    uint64_t coarse_conf;
    uint64_t coarse_val;
    uint64_t luma_left;
    uint64_t motion_mask;
    uint64_t tile_out;
    uint64_t tile_priority_out;
    uint width;
    uint height;
    uint coarse_width;
    uint coarse_height;
    float max_planar_disp_span;
    float min_planar_conf;
    float high_grad_thresh;
    float near_disp_thresh;
    float min_range_disp;
    float max_range_disp;
};

struct Push { uint64_t parameters; };
[[vk::push_constant]] Push pc;
#define tp (*(TileClassifierParameters*)pc.parameters)

groupshared uint gs_grad_high;
groupshared uint gs_grad_med;
groupshared uint gs_motion_flags;
groupshared uint gs_valid_coarse;
groupshared uint gs_min_disp_mm;
groupshared uint gs_max_disp_mm;
groupshared uint gs_min_conf_mm;

[shader("compute")][numthreads(16, 16, 1)]
void classify_tiles(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex) {
    if (gidx == 0u) {
        gs_grad_high = 0u;
        gs_grad_med = 0u;
        gs_motion_flags = 0u;
        gs_valid_coarse = 0u;
        gs_min_disp_mm = 0xffffffffu;
        gs_max_disp_mm = 0u;
        gs_min_conf_mm = 1000u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint x = tid.x, y = tid.y;
    if (x < tp.width && y < tp.height) {
        uint idx = y * tp.width + x;
        uint* luma = (uint*)tp.luma_left;
        uint xm = (x > 0u) ? (x - 1u) : x;
        uint xp = (x + 1u < tp.width) ? (x + 1u) : x;
        uint ym = (y > 0u) ? (y - 1u) : y;
        uint yp = (y + 1u < tp.height) ? (y + 1u) : y;
        int gx = abs(int(luma[y * tp.width + xp]) - int(luma[y * tp.width + xm]));
        int gy = abs(int(luma[yp * tp.width + x]) - int(luma[ym * tp.width + x]));
        float g = float(gx + gy);
        if (g >= tp.high_grad_thresh) InterlockedAdd(gs_grad_high, 1u);
        else if (g >= tp.high_grad_thresh * 0.5f) InterlockedAdd(gs_grad_med, 1u);

        if (tp.motion_mask != 0ull) {
            uint m = ((uint*)tp.motion_mask)[idx];
            if (m != 0u) InterlockedOr(gs_motion_flags, 1u << (m & 3u));
        }
    }

    // First 16 threads inspect the 4x4 coarse block corresponding to this 16x16 target tile
    if (gidx < 16u) {
        uint cx = gid.x * 4u + (gidx & 3u);
        uint cy = gid.y * 4u + (gidx >> 2u);
        if (cx < tp.coarse_width && cy < tp.coarse_height) {
            uint cidx = cy * tp.coarse_width + cx;
            uint cv = ((uint*)tp.coarse_val)[cidx];
            float cd = ((float*)tp.coarse_disp)[cidx];
            float cc = ((float*)tp.coarse_conf)[cidx];
            if (cv != 0u && isfinite(cd) && cd >= 0.f) {
                InterlockedAdd(gs_valid_coarse, 1u);
                uint d_q = uint(clamp(cd * 100.f, 0.f, 100000.f));
                uint c_q = uint(clamp(cc * 1000.f, 0.f, 1000.f));
                InterlockedMin(gs_min_disp_mm, d_q);
                InterlockedMax(gs_max_disp_mm, d_q);
                InterlockedMin(gs_min_conf_mm, c_q);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (gidx == 0u) {
        uint tiles_x = (tp.width + 15u) / 16u;
        uint tile_idx = gid.y * tiles_x + gid.x;
        float d_span = (gs_valid_coarse > 0u && gs_max_disp_mm >= gs_min_disp_mm)
            ? float(gs_max_disp_mm - gs_min_disp_mm) * 0.01f
            : 999.f;
        float min_conf = float(gs_min_conf_mm) * 0.001f;
        float mult = float(tp.width) / float(max(tp.coarse_width, 1u));
        float max_target_disp = (gs_valid_coarse > 0u) ? (float(gs_max_disp_mm) * 0.01f * mult) : 0.f;

        // 1. Motion state: 0=Static, 1=Dynamic, 2=DisocclusionWake, 3=Unknown
        uint motion_state = 0u;
        if ((gs_motion_flags & (1u << 2u)) != 0u) motion_state = 2u;
        else if ((gs_motion_flags & (1u << 1u)) != 0u) motion_state = 1u;
        else if ((gs_motion_flags & (1u << 3u)) != 0u) motion_state = 3u;

        // 2. Range state: 0=TooNear_Unknown, 1=Near, 2=Middle_Useful, 3=Far_Useful, 4=Far_BeyondAccuracy
        uint range_state = 2u;
        if (tp.max_range_disp > 0.f && max_target_disp > tp.max_range_disp) range_state = 0u;
        else if (max_target_disp >= tp.near_disp_thresh) range_state = 1u;
        else if (tp.min_range_disp > 0.f && max_target_disp < tp.min_range_disp && gs_valid_coarse == 16u && d_span <= tp.max_planar_disp_span) range_state = 4u;
        else if (max_target_disp < 12.6f) range_state = 3u;

        // 3. Quality state: 0=Easy, 1=Detailed, 2=Uncertain
        uint quality_state = 1u;
        if (gs_valid_coarse < 16u || d_span > tp.max_planar_disp_span * 1.5f || min_conf < tp.min_planar_conf || gs_grad_high >= 32u) {
            quality_state = 2u;
        } else if (gs_valid_coarse == 16u && min_conf >= tp.min_planar_conf && d_span <= tp.max_planar_disp_span && gs_grad_high < 8u) {
            quality_state = 0u;
        }

        // 4. Select processing level L0..L3 and compute-budget priority E[dQ]/(E[dT] + eps)
        uint level = 1u; // L1_Standard default
        float delta_q = 0.25f, delta_t = 1.0f;
        if (motion_state == 2u || gs_valid_coarse < 12u || (quality_state == 2u && d_span > 1.5f)) {
            level = 3u; // L3_Recovery
            delta_q = 0.95f;
            delta_t = 1.6f;
        } else if (quality_state == 2u || range_state == 1u || motion_state == 1u || motion_state == 3u) {
            level = 2u; // L2_Precision (never downgrade near obstacles, dynamic objects, thin structures, or unknown motion)
            delta_q = 0.80f;
            delta_t = 1.3f;
        } else if ((quality_state == 0u && motion_state == 0u && range_state == 2u) || range_state == 4u) {
            level = 0u; // L0_Minimal
            delta_q = 0.02f;
            delta_t = 0.25f;
        }

        uint packed = (level & 3u) | ((motion_state & 3u) << 2u) | ((range_state & 7u) << 4u) | ((quality_state & 3u) << 7u);
        ((uint*)tp.tile_out)[tile_idx] = packed;
        if (tp.tile_priority_out != 0ull) {
            ((float*)tp.tile_priority_out)[tile_idx] = delta_q / (delta_t + 1e-3f);
        }
    }
}
