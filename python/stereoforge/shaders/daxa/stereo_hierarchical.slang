#include "stereo_parameters.slang"
#include "../ops/sgm.slang"

struct LayeredPush { uint64_t left_parameters, right_parameters; uint pass, groups_x; };
[[vk::push_constant]] ConstantBuffer<LayeredPush> layered_pc;

uint sample_luma8_with_contrast(uint* img, uint w4, uint h4, int sx8, int sy8, out uint contrast4x4) {
    int bx = clamp(sx8 * 2, 0, int(w4) - 2);
    int by = clamp(sy8 * 2, 0, int(h4) - 2);
    uint r0 = uint(by) * w4 + uint(bx);
    uint r1 = r0 + w4;
    uint p00 = img[r0], p01 = img[r0 + 1u], p10 = img[r1], p11 = img[r1 + 1u];
    uint avg = (p00 + p01 + p10 + p11 + 2u) >> 2u;
    uint mn = min(min(p00, p01), min(p10, p11));
    uint mx = max(max(p00, p01), max(p10, p11));
    for (int dy = -1; dy <= 2; ++dy) {
        int yy = clamp(by + dy, 0, int(h4) - 1);
        uint row = uint(yy) * w4;
        for (int dx = -1; dx <= 2; ++dx) {
            int xx = clamp(bx + dx, 0, int(w4) - 1);
            uint v = img[row + uint(xx)];
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }
    contrast4x4 = mx - mn;
    return avg;
}

uint sample_luma8(uint* img, uint w4, uint h4, int sx8, int sy8) {
    int bx = clamp(sx8 * 2, 0, int(w4) - 2);
    int by = clamp(sy8 * 2, 0, int(h4) - 2);
    uint r0 = uint(by) * w4 + uint(bx);
    uint r1 = r0 + w4;
    return (img[r0] + img[r0 + 1u] + img[r1] + img[r1 + 1u] + 2u) >> 2u;
}

uint compute_census8(uint* img4, uint w4, uint h4, uint w8, uint h8, int x8, int y8, bool hybrid) {
    if (hybrid) {
        int gxl = 0, gxr = 0;
        uint lt = 0, lb = 0;
        [unroll] for (int dy = -2; dy <= 2; ++dy) {
            uint p_m2 = sample_luma8(img4, w4, h4, x8 - 2, y8 + dy);
            uint p_m1 = sample_luma8(img4, w4, h4, x8 - 1, y8 + dy);
            uint p_0  = sample_luma8(img4, w4, h4, x8,     y8 + dy);
            uint p_p1 = sample_luma8(img4, w4, h4, x8 + 1, y8 + dy);
            uint p_p2 = sample_luma8(img4, w4, h4, x8 + 2, y8 + dy);
            gxl += int(p_0) - int(p_m2);
            gxr += int(p_p2) - int(p_0);
            if (dy <= 0) lt += p_m2 + p_m1 + p_0 + p_p1 + p_p2;
            if (dy >= 0) lb += p_m2 + p_m1 + p_0 + p_p1 + p_p2;
        }
        uint q_gxl = uint(clamp((gxl + 60) / 15, 0, 8));
        uint q_gxr = uint(clamp((gxr + 60) / 15, 0, 8));
        uint q_lt = min(lt / (15u * 28u), 8u);
        uint q_lb = min(lb / (15u * 28u), 8u);
        return sgm_thermo8(q_gxl) | (sgm_thermo8(q_gxr) << 8) | (sgm_thermo8(q_lt) << 16) | (sgm_thermo8(q_lb) << 24);
    }
    uint center = sample_luma8(img4, w4, h4, x8, y8);
    uint value = 0, bit = 0;
    for (int oy = -2; oy <= 2; ++oy) {
        for (int ox = -2; ox <= 2; ++ox) {
            if (ox == 0 && oy == 0) continue;
            if (sample_luma8(img4, w4, h4, x8 + ox, y8 + oy) < center) value |= 1u << bit;
            ++bit;
        }
    }
    const int2 sym[8] = { int2(0,1), int2(0,-1), int2(1,0), int2(-1,0), int2(0,2), int2(0,-2), int2(2,0), int2(-2,0) };
    for (uint k = 0; k < 8; ++k) {
        int2 d = sym[k];
        if (sample_luma8(img4, w4, h4, x8 + d.y, y8 + d.x) < sample_luma8(img4, w4, h4, x8 - d.y, y8 - d.x))
            value |= 1u << (24u + k);
    }
    return value;
}

// Stage 0: Compute 1/4-scale (320x180) and 1/8-scale (160x90) Census descriptors & contrast in one pass.
[shader("compute")][numthreads(64, 1, 1)]
void coarse_prep_1to8_1to4(uint3 tid : SV_DispatchThreadID) {
    Parameters lp = *(Parameters*)layered_pc.left_parameters;
    uint w4 = lp.width, h4 = lp.height, n4 = w4 * h4;
    uint i4 = tid.x;
    if (i4 >= n4) return;
    uint x4 = i4 % w4, y4 = i4 / w4;
    bool hybrid = (lp.subpixel & 2u) != 0;
    uint* luma_l = (uint*)lp.left_address;
    uint* luma_r = (uint*)lp.right_address;
    uint* desc4_l = (uint*)lp.left_desc_address;
    uint* desc4_r = (uint*)lp.right_desc_address;
    uint* valid4_l = (uint*)lp.left_valid_address;
    uint* valid4_r = (uint*)lp.right_valid_address;

    bool sup4 = sgm_census_supported(w4, h4, i4);
    desc4_l[i4] = sup4 ? sgm_census(luma_l, w4, h4, i4) : 0u;
    desc4_r[i4] = sup4 ? sgm_census(luma_r, w4, h4, i4) : 0u;
    valid4_l[i4] = uint(sup4);
    valid4_r[i4] = uint(sup4);

    if ((x4 & 1u) == 0u && (y4 & 1u) == 0u) {
        uint w8 = w4 >> 1u, h8 = h4 >> 1u, n8 = w8 * h8;
        uint x8 = x4 >> 1u, y8 = y4 >> 1u, i8 = y8 * w8 + x8;
        uint* pen = (uint*)lp.penalties_address;
        uint* luma8_l = pen + 0u * n8;
        uint* luma8_r = pen + 1u * n8;
        uint* desc8_l = pen + 2u * n8;
        uint* desc8_r = pen + 3u * n8;
        uint* valid8_l = pen + 4u * n8;
        uint* valid8_r = pen + 5u * n8;

        uint c_l = 0, c_r = 0;
        uint avg_l = sample_luma8_with_contrast(luma_l, w4, h4, int(x8), int(y8), c_l);
        uint avg_r = sample_luma8_with_contrast(luma_r, w4, h4, int(x8), int(y8), c_r);
        luma8_l[i8] = (avg_l & 0xffu) | ((c_l & 0xffu) << 16u);
        luma8_r[i8] = (avg_r & 0xffu) | ((c_r & 0xffu) << 16u);

        bool sup8 = (x8 >= 2u && y8 >= 2u && x8 + 2u < w8 && y8 + 2u < h8);
        desc8_l[i8] = sup8 ? compute_census8(luma_l, w4, h4, w8, h8, int(x8), int(y8), false) : 0u;
        desc8_r[i8] = sup8 ? compute_census8(luma_r, w4, h4, w8, h8, int(x8), int(y8), false) : 0u;
        valid8_l[i8] = uint(sup8);
        valid8_r[i8] = uint(sup8);
    }
}

#if !defined(SF_CPU_TARGET)
groupshared uint gs_desc8_own[8][36];
groupshared uint gs_valid8_own[8][36];
groupshared uint gs_desc8_other[8][68];
groupshared uint gs_valid8_other[8][68];
#endif

// Stage 1: 1/8-scale (160x90) workgroup-cooperative 13-tap local cost-filtered Census matching.
[shader("compute")][numthreads(32, 4, 1)]
void coarse_match_1to8(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gidx : SV_GroupIndex) {
    if (gid.z >= 2u) return;
    uint eye = gid.z;
    Parameters lp = *(Parameters*)layered_pc.left_parameters;
    uint w8 = lp.width >> 1u, h8 = lp.height >> 1u, n8 = w8 * h8;
    uint D8 = clamp(lp.disparities >> 1u, 1u, 32u);
    uint* pen = (uint*)lp.penalties_address;
    uint* desc8_own = pen + (2u + eye) * n8;
    uint* desc8_other = pen + (2u + (1u - eye)) * n8;
    uint* valid8_own = pen + (4u + eye) * n8;
    uint* valid8_other = pen + (4u + (1u - eye)) * n8;
    float* disp8_out = (float*)(pen + (6u + eye) * n8);
    float* conf8_out = (float*)(pen + (8u + eye) * n8);
    uint* flags8_out = pen + (10u + eye) * n8;

#if !defined(SF_CPU_TARGET)
    int base_x = int(gid.x * 32u) - 2;
    int base_y = int(gid.y * 4u) - 2;
    int other_base_x = (eye == 0u) ? (base_x - int(D8)) : base_x;
    uint other_cols = 36u + D8;

    for (uint k = gidx; k < 8u * 36u; k += 128u) {
        uint hy = k / 36u, hx = k % 36u;
        int gx = base_x + int(hx), gy = base_y + int(hy);
        bool ok = (gx >= 0 && gx < int(w8) && gy >= 0 && gy < int(h8));
        uint idx = ok ? (uint(gy) * w8 + uint(gx)) : 0u;
        gs_desc8_own[hy][hx] = ok ? desc8_own[idx] : 0u;
        gs_valid8_own[hy][hx] = ok ? valid8_own[idx] : 0u;
    }
    for (uint k = gidx; k < 8u * other_cols; k += 128u) {
        uint hy = k / other_cols, hx = k % other_cols;
        int gx = other_base_x + int(hx), gy = base_y + int(hy);
        bool ok = (gx >= 0 && gx < int(w8) && gy >= 0 && gy < int(h8));
        uint idx = ok ? (uint(gy) * w8 + uint(gx)) : 0u;
        gs_desc8_other[hy][hx] = ok ? desc8_other[idx] : 0u;
        gs_valid8_other[hy][hx] = ok ? valid8_other[idx] : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
#endif

    uint x8 = gid.x * 32u + gtid.x, y8 = gid.y * 4u + gtid.y;
    if (x8 >= w8 || y8 >= h8) return;
    uint i8 = y8 * w8 + x8;
    uint lx = gtid.x + 2u, ly = gtid.y + 2u;
#if !defined(SF_CPU_TARGET)
    uint center_v8 = gs_valid8_own[ly][lx];
#else
    uint center_v8 = valid8_own[i8];
#endif
    if (center_v8 == 0u) {
        disp8_out[i8] = asfloat(0x7fc00000u);
        conf8_out[i8] = 0.f;
        flags8_out[i8] = 2u; // risk=1, valid=0
        return;
    }

    static const int2 kTaps[13] = {
        int2(0, 0),
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
        int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1),
        int2(-2, 0), int2(2, 0), int2(0, -2), int2(0, 2)
    };
    static const uint kWeights[13] = { 4u, 2u, 2u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u };
    uint own_d[13];
    uint own_v[13];
    [unroll] for (uint t = 0; t < 13u; ++t) {
        int2 o = kTaps[t];
#if !defined(SF_CPU_TARGET)
        own_d[t] = gs_desc8_own[int(ly) + o.y][int(lx) + o.x];
        own_v[t] = gs_valid8_own[int(ly) + o.y][int(lx) + o.x];
#else
        int gx = int(x8) + o.x, gy = int(y8) + o.y;
        bool ok = (gx >= 0 && gx < int(w8) && gy >= 0 && gy < int(h8));
        uint idx = ok ? (uint(gy) * w8 + uint(gx)) : 0u;
        own_d[t] = ok ? desc8_own[idx] : 0u;
        own_v[t] = ok ? valid8_own[idx] : 0u;
#endif
    }

    uint best = 0xffffffffu, second = 0xffffffffu, third = 0xffffffffu, winner = 0xffffffffu;
    uint minus = 0xffffffffu, plus = 0xffffffffu, prev_cost = 0xffffffffu, count = 0u;
    uint base_ox = (eye == 0u) ? (lx + D8) : lx;

    for (uint d = 0; d < D8; ++d) {
        int gx_match = int(x8) + (eye != 0u ? int(d) : -int(d));
        uint cost = 0xffffffffu;
        if (gx_match >= 0 && gx_match < int(w8)) {
#if !defined(SF_CPU_TARGET)
            uint ox = (eye == 0u) ? (base_ox - d) : (base_ox + d);
            uint oth_v0 = gs_valid8_other[ly][ox];
            uint oth_d0 = gs_desc8_other[ly][ox];
#else
            uint j0 = y8 * w8 + uint(gx_match);
            uint oth_v0 = valid8_other[j0];
            uint oth_d0 = desc8_other[j0];
#endif
            if (oth_v0 != 0u) {
                uint c0 = countbits(own_d[0] ^ oth_d0);
                uint sum_c = 4u * c0;
                [unroll] for (uint t = 1u; t < 13u; ++t) {
                    int2 o = kTaps[t];
#if !defined(SF_CPU_TARGET)
                    uint oy = uint(int(ly) + o.y);
                    uint oxx = uint(int(ox) + o.x);
                    bool v = (own_v[t] != 0u) && (gs_valid8_other[oy][oxx] != 0u);
                    uint ct = v ? countbits(own_d[t] ^ gs_desc8_other[oy][oxx]) : c0;
#else
                    int gx_o = gx_match + o.x, gy_o = int(y8) + o.y;
                    bool ok_o = (gx_o >= 0 && gx_o < int(w8) && gy_o >= 0 && gy_o < int(h8));
                    uint idx_o = ok_o ? (uint(gy_o) * w8 + uint(gx_o)) : 0u;
                    bool v = (own_v[t] != 0u) && ok_o && (valid8_other[idx_o] != 0u);
                    uint ct = v ? countbits(own_d[t] ^ desc8_other[idx_o]) : c0;
#endif
                    sum_c += kWeights[t] * ct;
                }
                cost = sum_c;
                ++count;
                if (cost < best) {
                    third = second; second = best; best = cost; winner = d;
                    minus = prev_cost; plus = 0xffffffffu;
                } else if (cost < second) {
                    third = second; second = cost;
                } else if (cost < third) {
                    third = cost;
                }
            }
        }
        if (winner != 0xffffffffu && d == winner + 1u) plus = cost;
        prev_cost = cost;
    }

    bool plateau = (count >= 3u && second == best && best <= 180u && (plus == best || minus == best) && third >= best + 1u);
    uint eff_second = plateau ? third : second;
    bool valid = (count >= 2u && winner > 0u && best < 360u && eff_second > best);
    float d_out = valid ? float(winner) : asfloat(0x7fc00000u);
    float conf_out = 0.f;
    if (valid) {
        conf_out = clamp(float(eff_second - best) / float(max(eff_second, 1u)), 0.f, 1.f);
        if (winner > 0u && winner + 1u < D8 && minus != 0xffffffffu && plus != 0xffffffffu) {
            if (minus == best && plus > best) d_out -= 0.25f;
            else if (plus == best && minus > best) d_out += 0.25f;
            else {
                float denom = float(minus) + float(plus) - 2.f * float(best);
                if (denom > 0.f) d_out += clamp(0.5f * (float(minus) - float(plus)) / denom, -1.f, 1.f);
            }
        }
    }
    disp8_out[i8] = d_out;
    conf8_out[i8] = conf_out;
    flags8_out[i8] = valid ? 1u : 2u;
}

// Stage 2: 1/8-scale Left-Right consistency verification, 3x3 outlier-trimmed smoothing & risk tagging.
[shader("compute")][numthreads(64, 1, 1)]
void coarse_lr_risk_1to8(uint3 tid : SV_DispatchThreadID) {
    Parameters lp = *(Parameters*)layered_pc.left_parameters;
    uint w8 = lp.width >> 1u, h8 = lp.height >> 1u, n8 = w8 * h8;
    uint i8 = tid.x;
    if (i8 >= n8) return;
    uint x8 = i8 % w8, y8 = i8 / w8;
    uint* pen = (uint*)lp.penalties_address;

    [unroll] for (uint eye = 0; eye < 2u; ++eye) {
        uint* luma8 = pen + eye * n8;
        float* disp_own = (float*)(pen + (6u + eye) * n8);
        float* disp_other = (float*)(pen + (6u + (1u - eye)) * n8);
        float* conf_own = (float*)(pen + (8u + eye) * n8);
        uint* flags_own = pen + (10u + eye) * n8;
        uint* flags_other = pen + (10u + (1u - eye)) * n8;
        float* disp_clean = (float*)(pen + (12u + eye) * n8);
        uint* flags_clean = pen + (14u + eye) * n8;

        float d0 = disp_own[i8];
        float c0 = conf_own[i8];
        bool v0 = (flags_own[i8] & 1u) != 0u && isfinite(d0);
        bool lr0 = false;
        if (v0) {
            int mx = int(floor(float(x8) + (eye != 0u ? d0 : -d0) + 0.5f));
            if (mx >= 0 && mx < int(w8)) {
                uint j8 = y8 * w8 + uint(mx);
                float d_oth = disp_other[j8];
                if ((flags_other[j8] & 1u) != 0u && isfinite(d_oth) && abs(d0 - d_oth) <= 1.25f) {
                    lr0 = true;
                }
            }
        }

        uint cnt_lr = 0u;
        float mn = 3.402823466e38f, mx_d = -3.402823466e38f, sum_d = 0.f, sum_w = 0.f;
        for (int dy = -1; dy <= 1; ++dy) {
            int yy = int(y8) + dy;
            if (yy < 0 || yy >= int(h8)) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = int(x8) + dx;
                if (xx < 0 || xx >= int(w8)) continue;
                uint n_idx = uint(yy) * w8 + uint(xx);
                float dn = disp_own[n_idx];
                if ((flags_own[n_idx] & 1u) == 0u || !isfinite(dn)) continue;
                int mxx = int(floor(float(xx) + (eye != 0u ? dn : -dn) + 0.5f));
                if (mxx < 0 || mxx >= int(w8)) continue;
                uint mj = uint(yy) * w8 + uint(mxx);
                float doth = disp_other[mj];
                if ((flags_other[mj] & 1u) != 0u && isfinite(doth) && abs(dn - doth) <= 1.35f) {
                    ++cnt_lr;
                    mn = min(mn, dn);
                    mx_d = max(mx_d, dn);
                    float w = (dx == 0 && dy == 0) ? 2.f : 1.f;
                    sum_d += w * dn;
                    sum_w += w;
                }
            }
        }

        float span = (cnt_lr > 0u) ? (mx_d - mn) : 99.f;
        bool repaired = (!lr0) && (cnt_lr >= 5u) && (span <= 0.85f);
        bool valid_final = lr0 || repaired;
        float d_final = valid_final ? ((lr0 && cnt_lr >= 4u && span <= 0.65f) ? (0.6f * d0 + 0.4f * (sum_d / sum_w)) : (repaired ? (sum_d / sum_w) : d0)) : asfloat(0x7fc00000u);
        bool risk = (!lr0) || (cnt_lr < 5u) || (span > 1.00f) || (c0 < 0.06f);
        disp_clean[i8] = d_final;
        flags_clean[i8] = uint(valid_final) | (uint(risk) << 1u);
    }
}

#if !defined(SF_CPU_TARGET)
groupshared uint gs_desc4_own[8][36];
groupshared uint gs_valid4_own[8][36];
groupshared uint gs_desc4_other[8][100];
groupshared uint gs_valid4_other[8][100];
#endif

// Stage 3: 1/4-scale (320x180) guided primary path (r=2) + selective 1/4-scale full fallback on risk/edge regions.
[shader("compute")][numthreads(32, 4, 1)]
void coarse_selective_fallback_1to4(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gidx : SV_GroupIndex) {
    if (gid.z >= 2u) return;
    uint eye = gid.z;
    Parameters lp = *(Parameters*)layered_pc.left_parameters;
    Parameters ep = (eye == 0u) ? lp : *(Parameters*)layered_pc.right_parameters;
    uint w4 = lp.width, h4 = lp.height, n4 = w4 * h4;
    uint w8 = w4 >> 1u, h8 = h4 >> 1u, n8 = w8 * h8;
    uint D4 = clamp(lp.disparities, 1u, 64u);

    uint* desc4_own = (uint*)(eye == 0u ? lp.left_desc_address : lp.right_desc_address);
    uint* desc4_other = (uint*)(eye == 0u ? lp.right_desc_address : lp.left_desc_address);
    uint* valid4_own = (uint*)(eye == 0u ? lp.left_valid_address : lp.right_valid_address);
    uint* valid4_other = (uint*)(eye == 0u ? lp.right_valid_address : lp.left_valid_address);
    uint* pen = (uint*)lp.penalties_address;
    uint* luma8_own = pen + eye * n8;
    float* disp8_clean = (float*)(pen + (12u + eye) * n8);
    uint* flags8_clean = pen + (14u + eye) * n8;

    float* disp4_out = (float*)ep.disparity_address;
    float* conf4_out = (float*)ep.confidence_address;
    uint* valid4_out = (uint*)ep.validity_address;

#if !defined(SF_CPU_TARGET)
    int base_x = int(gid.x * 32u) - 2;
    int base_y = int(gid.y * 4u) - 2;
    int other_base_x = (eye == 0u) ? (base_x - int(D4)) : base_x;
    uint other_cols = 36u + D4;

    for (uint k = gidx; k < 8u * 36u; k += 128u) {
        uint hy = k / 36u, hx = k % 36u;
        int gx = base_x + int(hx), gy = base_y + int(hy);
        bool ok = (gx >= 0 && gx < int(w4) && gy >= 0 && gy < int(h4));
        uint idx = ok ? (uint(gy) * w4 + uint(gx)) : 0u;
        gs_desc4_own[hy][hx] = ok ? desc4_own[idx] : 0u;
        gs_valid4_own[hy][hx] = ok ? valid4_own[idx] : 0u;
    }
    for (uint k = gidx; k < 8u * other_cols; k += 128u) {
        uint hy = k / other_cols, hx = k % other_cols;
        int gx = other_base_x + int(hx), gy = base_y + int(hy);
        bool ok = (gx >= 0 && gx < int(w4) && gy >= 0 && gy < int(h4));
        uint idx = ok ? (uint(gy) * w4 + uint(gx)) : 0u;
        gs_desc4_other[hy][hx] = ok ? desc4_other[idx] : 0u;
        gs_valid4_other[hy][hx] = ok ? valid4_other[idx] : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
#endif

    uint x4 = gid.x * 32u + gtid.x, y4 = gid.y * 4u + gtid.y;
    if (x4 >= w4 || y4 >= h4) return;
    uint i4 = y4 * w4 + x4;
    uint lx = gtid.x + 2u, ly = gtid.y + 2u;
#if !defined(SF_CPU_TARGET)
    uint center_v4 = gs_valid4_own[ly][lx];
#else
    uint center_v4 = valid4_own[i4];
#endif
    if (center_v4 == 0u) {
        disp4_out[i4] = asfloat(0x7fc00000u);
        conf4_out[i4] = 0.f;
        valid4_out[i4] = 0u;
        return;
    }

    // Bilinearly inspect 2x2 parent 1/8-scale neighborhood
    float cx8 = (float(x4) + 0.5f) * 0.5f - 0.5f;
    float cy8 = (float(y4) + 0.5f) * 0.5f - 0.5f;
    int ix8 = int(floor(cx8)), iy8 = int(floor(cy8));
    float wx = cx8 - float(ix8), wy = cy8 - float(iy8);
    float weights[4] = { (1.f - wx) * (1.f - wy), wx * (1.f - wy), (1.f - wx) * wy, wx * wy };
    float pred_d4 = 0.f, w_sum = 0.f, min_d4 = 3.402823466e38f, max_d4 = -3.402823466e38f;
    uint valid_taps = 0u;
    bool any_risk = false;
    [unroll] for (uint tap = 0; tap < 4u; ++tap) {
        int xx = clamp(ix8 + int(tap & 1u), 0, int(w8) - 1);
        int yy = clamp(iy8 + int(tap >> 1u), 0, int(h8) - 1);
        uint j8 = uint(yy) * w8 + uint(xx);
        uint fl = flags8_clean[j8];
        if ((fl & 2u) != 0u) any_risk = true;
        float d8 = disp8_clean[j8];
        if ((fl & 1u) != 0u && isfinite(d8) && d8 > 0.f) {
            float d4_cand = 2.f * d8;
            float w = weights[tap] + 0.01f;
            pred_d4 += w * d4_cand;
            w_sum += w;
            min_d4 = min(min_d4, d4_cand);
            max_d4 = max(max_d4, d4_cand);
            ++valid_taps;
        }
    }
    if (w_sum > 0.f) pred_d4 /= w_sum;

    uint p_idx8 = uint(clamp(int(x4 >> 1u), 0, int(w8) - 1)) + uint(clamp(int(y4 >> 1u), 0, int(h8) - 1)) * w8;
    uint contrast8 = (luma8_own[p_idx8] >> 16u) & 0xffu;
    bool low_risk = (valid_taps == 4u) && (!any_risk) && ((max_d4 - min_d4) <= 1.6f);

    uint lo = 0u, hi = D4 - 1u;
    float prior_weight = 0.f;
    if (low_risk) {
        int c4 = int(floor(pred_d4 + 0.5f));
        lo = uint(clamp(c4 - 2, 0, int(D4) - 1));
        hi = uint(clamp(c4 + 2, int(lo), int(D4) - 1));
        prior_weight = 6.0f;
    } else if (valid_taps >= 3u && (max_d4 - min_d4) <= 3.5f) {
        lo = uint(clamp(int(floor(min_d4)) - 3, 0, int(D4) - 1));
        hi = uint(clamp(int(ceil(max_d4)) + 3, int(lo), int(D4) - 1));
        prior_weight = (contrast8 <= 16u) ? 2.5f : 1.0f;
    } else if (valid_taps >= 2u && (max_d4 - min_d4) <= 2.4f && contrast8 <= 16u) {
        prior_weight = 2.5f;
    }

    static const int2 kTaps[13] = {
        int2(0, 0),
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
        int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1),
        int2(-2, 0), int2(2, 0), int2(0, -2), int2(0, 2)
    };
    static const uint kWeights[13] = { 4u, 2u, 2u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u };
    uint own_d[13];
    uint own_v[13];
    [unroll] for (uint t = 0; t < 13u; ++t) {
        int2 o = kTaps[t];
#if !defined(SF_CPU_TARGET)
        own_d[t] = gs_desc4_own[int(ly) + o.y][int(lx) + o.x];
        own_v[t] = gs_valid4_own[int(ly) + o.y][int(lx) + o.x];
#else
        int gx = int(x4) + o.x, gy = int(y4) + o.y;
        bool ok = (gx >= 0 && gx < int(w4) && gy >= 0 && gy < int(h4));
        uint idx = ok ? (uint(gy) * w4 + uint(gx)) : 0u;
        own_d[t] = ok ? desc4_own[idx] : 0u;
        own_v[t] = ok ? valid4_own[idx] : 0u;
#endif
    }

    uint best = 0xffffffffu, second = 0xffffffffu, third = 0xffffffffu, winner = 0xffffffffu;
    uint minus = 0xffffffffu, plus = 0xffffffffu, prev_cost = 0xffffffffu, count = 0u;
    uint base_ox = (eye == 0u) ? (lx + D4) : lx;

    for (uint d = lo; d <= hi; ++d) {
        int gx_match = int(x4) + (eye != 0u ? int(d) : -int(d));
        uint cost = 0xffffffffu;
        if (gx_match >= 0 && gx_match < int(w4)) {
#if !defined(SF_CPU_TARGET)
            uint ox = (eye == 0u) ? (base_ox - d) : (base_ox + d);
            uint oth_v0 = gs_valid4_other[ly][ox];
            uint oth_d0 = gs_desc4_other[ly][ox];
#else
            uint j0 = y4 * w4 + uint(gx_match);
            uint oth_v0 = valid4_other[j0];
            uint oth_d0 = desc4_other[j0];
#endif
            if (oth_v0 != 0u) {
                uint c0 = countbits(own_d[0] ^ oth_d0);
                uint sum_c = 4u * c0;
                [unroll] for (uint t = 1u; t < 13u; ++t) {
                    int2 o = kTaps[t];
#if !defined(SF_CPU_TARGET)
                    uint oy = uint(int(ly) + o.y);
                    uint oxx = uint(int(ox) + o.x);
                    bool v = (own_v[t] != 0u) && (gs_valid4_other[oy][oxx] != 0u);
                    uint ct = v ? countbits(own_d[t] ^ gs_desc4_other[oy][oxx]) : c0;
#else
                    int gx_o = gx_match + o.x, gy_o = int(y4) + o.y;
                    bool ok_o = (gx_o >= 0 && gx_o < int(w4) && gy_o >= 0 && gy_o < int(h4));
                    uint idx_o = ok_o ? (uint(gy_o) * w4 + uint(gx_o)) : 0u;
                    bool v = (own_v[t] != 0u) && ok_o && (valid4_other[idx_o] != 0u);
                    uint ct = v ? countbits(own_d[t] ^ desc4_other[idx_o]) : c0;
#endif
                    sum_c += kWeights[t] * ct;
                }
                if (prior_weight > 0.f) {
                    sum_c += uint(min(24.f, floor(prior_weight * abs(float(d) - pred_d4) + 0.5f)));
                }
                cost = sum_c;
                ++count;
                if (cost < best) {
                    third = second; second = best; best = cost; winner = d;
                    minus = prev_cost; plus = 0xffffffffu;
                } else if (cost < second) {
                    third = second; second = cost;
                } else if (cost < third) {
                    third = cost;
                }
            }
        }
        if (winner != 0xffffffffu && d == winner + 1u) plus = cost;
        prev_cost = cost;
    }

    bool plateau = (count >= 3u && second == best && best <= 180u && (plus == best || minus == best) && third >= best + 1u);
    uint eff_second = plateau ? third : second;
    bool valid = (count >= 2u && winner > 0u && best < 380u && eff_second > best);
    float d_out = valid ? float(winner) : asfloat(0x7fc00000u);
    float conf_out = 0.f;
    if (valid) {
        conf_out = clamp(float(eff_second - best) / float(max(eff_second, 1u)), 0.f, 1.f);
        if ((lp.subpixel & 1u) != 0u && winner > lo && winner < hi && minus != 0xffffffffu && plus != 0xffffffffu) {
            if (minus == best && plus > best) d_out -= 0.25f;
            else if (plus == best && minus > best) d_out += 0.25f;
            else {
                float denom = float(minus) + float(plus) - 2.f * float(best);
                if (denom > 0.f) d_out += clamp(0.5f * (float(minus) - float(plus)) / denom, -1.f, 1.f);
            }
        } else if (low_risk) {
            d_out = clamp(0.5f * d_out + 0.5f * pred_d4, float(lo), float(hi));
        }
    }
    disp4_out[i4] = d_out;
    conf4_out[i4] = conf_out;
    valid4_out[i4] = uint(valid);
}
