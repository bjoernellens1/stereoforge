"""Portable Slang-to-CPU execution backend (`slangc -target cpp -DSF_CPU_TARGET=1`).

Compiles the production Slang shader modules (`preprocess.slang` and `refinement.slang`)
directly into native C++ via `slangc -target cpp -DSF_CPU_TARGET=1` and builds a shared
library (`libstereoforge_slang_cpu.so`) that executes the exact same `.slang` shader
entrypoints on CPU host memory without Vulkan, CUDA, ROCm, or a GPU.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build" / "ros-daxa-native"
SLANGC = BUILD_DIR / "_deps" / "sf_refinement_slang_sdk-src" / "bin" / "slangc"
SLANG_INCLUDE = BUILD_DIR / "_deps" / "sf_refinement_slang_sdk-src" / "include"
DAXA_INCLUDE = BUILD_DIR / "_deps" / "sf_daxa-src" / "include"
REFINEMENT_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "daxa" / "refinement.slang"
PREPROCESS_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "daxa" / "preprocess.slang"
STEREO_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "daxa" / "stereo.slang"
STEREO_HIER_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "daxa" / "stereo_hierarchical.slang"
DEPTH_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "daxa" / "depth.slang"
REFINEMENT_OPS_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "ops" / "refinement.slang"
SGM_OPS_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "ops" / "sgm.slang"


class _CpuPreprocessParameters(ctypes.Structure):
    _fields_ = [
        ("raw", ctypes.c_uint64),
        ("original", ctypes.c_uint64 * 2),
        ("rectified", ctypes.c_uint64 * 2),
        ("luma", ctypes.c_uint64 * 2),
        ("support", ctypes.c_uint64 * 2),
        ("maps", ctypes.c_uint64 * 4),
        ("eye_width", ctypes.c_uint32),
        ("source_height", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]


class _CpuStereoParameters(ctypes.Structure):
    _fields_ = [
        ("left_address", ctypes.c_uint64),
        ("right_address", ctypes.c_uint64),
        ("left_desc_address", ctypes.c_uint64),
        ("right_desc_address", ctypes.c_uint64),
        ("left_valid_address", ctypes.c_uint64),
        ("right_valid_address", ctypes.c_uint64),
        ("costs_address", ctypes.c_uint64),
        ("sums_address", ctypes.c_uint64),
        ("disparity_address", ctypes.c_uint64),
        ("confidence_address", ctypes.c_uint64),
        ("validity_address", ctypes.c_uint64),
        ("depth_address", ctypes.c_uint64),
        ("depth_valid_address", ctypes.c_uint64),
        ("counters_address", ctypes.c_uint64),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("disparities", ctypes.c_uint32),
        ("paths", ctypes.c_uint32),
        ("p1", ctypes.c_uint32),
        ("p2", ctypes.c_uint32),
        ("subpixel", ctypes.c_uint32),
        ("metric_enabled", ctypes.c_uint32),
        ("fx", ctypes.c_float),
        ("baseline_m", ctypes.c_float),
        ("cx_offset", ctypes.c_float),
        ("packing_bits", ctypes.c_uint32),
        ("penalties_address", ctypes.c_uint64),
    ]


class _CpuDepthParameters(ctypes.Structure):
    _fields_ = [
        ("inputs", ctypes.c_uint64 * 8),
        ("outputs", ctypes.c_uint64 * 9),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("tolerance", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("baseline", ctypes.c_float),
        ("principal_offset", ctypes.c_float),
        ("source_low", ctypes.c_uint32),
        ("source_high", ctypes.c_uint32),
    ]


class _CpuRefinementParameters(ctypes.Structure):
    _fields_ = [
        ("original", ctypes.c_uint64 * 2),
        ("coarse", ctypes.c_uint64 * 3),
        ("maps", ctypes.c_uint64 * 4),
        ("output", ctypes.c_uint64 * 30),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("source_width", ctypes.c_uint32),
        ("source_height", ctypes.c_uint32),
        ("coarse_width", ctypes.c_uint32),
        ("coarse_height", ctypes.c_uint32),
        ("radius", ctypes.c_uint32),
        ("disparities", ctypes.c_uint32),
        ("min_margin", ctypes.c_uint32),
        ("subpixel", ctypes.c_uint32),
        ("prior_floor", ctypes.c_float),
        ("max_span", ctypes.c_float),
        ("lr_tolerance", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("baseline", ctypes.c_float),
        ("principal_offset", ctypes.c_float),
        ("source_low", ctypes.c_uint32),
        ("source_high", ctypes.c_uint32),
        ("borrowed", ctypes.c_uint64 * 4),
        ("completed", ctypes.c_uint64 * 7),
        ("planar_cfg", ctypes.c_uint32),
        ("planar_coarse_lr_tol", ctypes.c_float),
        ("planar_patch_luma_tol", ctypes.c_float),
        ("planar_max_rmse", ctypes.c_float),
    ]


assert ctypes.sizeof(_CpuRefinementParameters) == 488


def _ensure_cpu_library() -> Path:
    cache_dir = Path("/tmp/stereoforge_slang_cpu_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    h.update(b"slang_cpu_v11_avx512_popcnt_fused_v3")
    for p in (
        REFINEMENT_SLANG,
        PREPROCESS_SLANG,
        STEREO_HIER_SLANG,
        DEPTH_SLANG,
        REFINEMENT_OPS_SLANG,
        SGM_OPS_SLANG,
    ):
        h.update(p.read_bytes())
    digest = h.hexdigest()[:16]
    so_path = cache_dir / f"libstereoforge_slang_cpu_{digest}.so"
    if so_path.exists():
        return so_path

    compile_specs = [
        (REFINEMENT_SLANG, "rectify"),
        (REFINEMENT_SLANG, "census"),
        (REFINEMENT_SLANG, "match"),
        (REFINEMENT_SLANG, "lr"),
        (REFINEMENT_SLANG, "metric"),
        (REFINEMENT_SLANG, "planar"),
        (PREPROCESS_SLANG, "decode_yuyv"),
        (PREPROCESS_SLANG, "rectify_gray"),
        (STEREO_HIER_SLANG, "coarse_prep_1to8_1to4"),
        (STEREO_HIER_SLANG, "coarse_match_1to8"),
        (STEREO_HIER_SLANG, "coarse_lr_risk_1to8"),
        (STEREO_HIER_SLANG, "coarse_selective_fallback_1to4"),
        (DEPTH_SLANG, "lr_consistency"),
        (DEPTH_SLANG, "metric_observation"),
    ]
    generated_sections = []
    for slang_file, entry in compile_specs:
        cpp_out = cache_dir / f"gen_{entry}.cpp"
        subprocess.run(
            [
                str(SLANGC),
                str(slang_file),
                "-I",
                str(DAXA_INCLUDE),
                "-target",
                "cpp",
                "-DSF_CPU_TARGET=1",
                "-entry",
                entry,
                "-o",
                str(cpp_out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        src = cpp_out.read_text(encoding="utf-8")
        src = src.replace(
            "SLANG_PRELUDE_EXPORT",
            "static inline __attribute__((always_inline, flatten))",
        )
        src = src.replace(
            f"void _{entry}(",
            f"static inline __attribute__((always_inline, flatten)) void _{entry}(",
        )
        lines = []
        for line in src.splitlines():
            if line.strip().startswith("#include"):
                continue
            if "for (uint32_t x = 0; x < " in line:
                lines.append("        #pragma GCC ivdep")
            lines.append(line)
        body = "\n".join(lines)
        generated_sections.append(f"namespace ns_{entry} {{\n{body}\n}}")

    driver_cpp = cache_dir / f"driver_{digest}.cpp"
    joined_sections = "\n\n".join(generated_sections)
    driver_cpp.write_text(
        f"""
#include "{SLANG_INCLUDE}/slang-cpp-prelude.h"
#include <cstdint>
#include <omp.h>

static void __attribute__((constructor)) sf_init_omp() {{
    omp_set_max_active_levels(1);
    omp_set_dynamic(0);
    int n = omp_get_num_procs();
    int phys = (n >= 16) ? 16 : n;
    omp_set_num_threads(phys);
}}

{joined_sections}

extern "C" {{

void sf_cpu_dispatch_1d(int entry_id, uint64_t params_addr, uint32_t total_threads) {{
    uint32_t num_groups = (total_threads + 63u) / 64u;
    #define SF_CALL_1D(idx, ns, fn) case idx: {{ ns::Push_0 push{{params_addr}}; ns::GlobalParams_0 gp{{&push}}; ns::fn(&vin, nullptr, &gp); break; }}
    #pragma omp parallel for schedule(static)
    for (uint32_t g = 0; g < num_groups; ++g) {{
        ComputeVaryingInput vin{{}};
        vin.startGroupID = {{g, 0u, 0u}};
        vin.endGroupID = {{g + 1u, 1u, 1u}};
        switch (entry_id) {{
            SF_CALL_1D(0, ns_rectify, rectify_Group)
            SF_CALL_1D(1, ns_census, census_Group)
            SF_CALL_1D(2, ns_match, match_Group)
            SF_CALL_1D(3, ns_lr, lr_Group)
            SF_CALL_1D(4, ns_metric, metric_Group)
            SF_CALL_1D(5, ns_lr_consistency, lr_consistency_Group)
            SF_CALL_1D(6, ns_metric_observation, metric_observation_Group)
        }}
    }}
    #undef SF_CALL_1D
}}

void sf_cpu_dispatch_preprocess(int entry_id, uint64_t params_addr, uint32_t width, uint32_t height) {{
    uint32_t gx_cnt = (width + 7u) / 8u;
    uint32_t gy_cnt = (height + 7u) / 8u;
    #pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t gz = 0; gz < 2u; ++gz) {{
        for (uint32_t gy = 0; gy < gy_cnt; ++gy) {{
            for (uint32_t gx = 0; gx < gx_cnt; ++gx) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{gx, gy, gz}};
                vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                if (entry_id == 0) {{
                    ns_decode_yuyv::Push_0 push{{params_addr}};
                    ns_decode_yuyv::GlobalParams_0 gp{{&push}};
                    ns_decode_yuyv::decode_yuyv_Group(&vin, nullptr, &gp);
                }} else {{
                    ns_rectify_gray::Push_0 push{{params_addr}};
                    ns_rectify_gray::GlobalParams_0 gp{{&push}};
                    ns_rectify_gray::rectify_gray_Group(&vin, nullptr, &gp);
                }}
            }}
        }}
    }}
}}

void sf_cpu_dispatch_hier_1d(int entry_id, uint64_t left_params_addr, uint64_t right_params_addr, uint32_t total_threads) {{
    uint32_t num_groups = (total_threads + 63u) / 64u;
    #pragma omp parallel for schedule(static)
    for (uint32_t g = 0; g < num_groups; ++g) {{
        ComputeVaryingInput vin{{}};
        vin.startGroupID = {{g, 0u, 0u}};
        vin.endGroupID = {{g + 1u, 1u, 1u}};
        if (entry_id == 0) {{
            ns_coarse_prep_1to8_1to4::LayeredPush_0 push{{left_params_addr, right_params_addr, 0u, num_groups}};
            ns_coarse_prep_1to8_1to4::GlobalParams_0 gp{{&push}};
            ns_coarse_prep_1to8_1to4::coarse_prep_1to8_1to4_Group(&vin, nullptr, &gp);
        }} else {{
            ns_coarse_lr_risk_1to8::LayeredPush_0 push{{left_params_addr, right_params_addr, 0u, num_groups}};
            ns_coarse_lr_risk_1to8::GlobalParams_0 gp{{&push}};
            ns_coarse_lr_risk_1to8::coarse_lr_risk_1to8_Group(&vin, nullptr, &gp);
        }}
    }}
}}

void sf_cpu_dispatch_hier_3d(int entry_id, uint64_t left_params_addr, uint64_t right_params_addr, uint32_t width, uint32_t height) {{
    uint32_t gx_cnt = (width + 31u) / 32u;
    uint32_t gy_cnt = (height + 3u) / 4u;
    #pragma omp parallel for collapse(3) schedule(static)
    for (uint32_t gz = 0; gz < 2u; ++gz) {{
        for (uint32_t gy = 0; gy < gy_cnt; ++gy) {{
            for (uint32_t gx = 0; gx < gx_cnt; ++gx) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{gx, gy, gz}};
                vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                if (entry_id == 0) {{
                    ns_coarse_match_1to8::LayeredPush_0 push{{left_params_addr, right_params_addr, 0u, gx_cnt}};
                    ns_coarse_match_1to8::GlobalParams_0 gp{{&push}};
                    ns_coarse_match_1to8::coarse_match_1to8_Group(&vin, nullptr, &gp);
                }} else {{
                    ns_coarse_selective_fallback_1to4::LayeredPush_0 push{{left_params_addr, right_params_addr, 0u, gx_cnt}};
                    ns_coarse_selective_fallback_1to4::GlobalParams_0 gp{{&push}};
                    ns_coarse_selective_fallback_1to4::coarse_selective_fallback_1to4_Group(&vin, nullptr, &gp);
                }}
            }}
        }}
    }}
}}

void sf_cpu_dispatch_planar(uint64_t params_addr, uint32_t width, uint32_t height) {{
    uint32_t groups_x = (width + 15u) / 16u;
    uint32_t groups_y = (height + 15u) / 16u;
    #pragma omp parallel for collapse(2) schedule(dynamic, 2)
    for (uint32_t gy = 0; gy < groups_y; ++gy) {{
        for (uint32_t gx = 0; gx < groups_x; ++gx) {{
            ComputeVaryingInput vin{{}};
            vin.startGroupID = {{gx, gy, 0u}};
            vin.endGroupID = {{gx + 1u, gy + 1u, 1u}};
            ns_planar::Push_0 push{{params_addr}};
            ns_planar::GlobalParams_0 gp{{&push}};
            ns_planar::planar_Group(&vin, nullptr, &gp);
        }}
    }}
}}

void sf_cpu_run_refinement_pipeline(
    uint64_t ref_params_addr,
    uint32_t width,
    uint32_t height,
    int ultra_fast,
    int planar_recovery
) {{
    uint32_t n = width * height;
    uint32_t n2 = 2u * n;
    uint32_t groups_both = (n2 + 63u) / 64u;
    uint32_t groups_metric = (ultra_fast != 0) ? ((n + 63u) / 64u) : groups_both;
    uint32_t planar_gx = (width + 15u) / 16u;
    uint32_t planar_gy = (height + 15u) / 16u;

    #pragma omp parallel
    {{
        {{
            ns_census::Push_0 push{{ref_params_addr}};
            ns_census::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_both; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_census::census_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_match::Push_0 push{{ref_params_addr}};
            ns_match::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_both; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_match::match_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_lr::Push_0 push{{ref_params_addr}};
            ns_lr::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_both; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_lr::lr_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_metric::Push_0 push{{ref_params_addr}};
            ns_metric::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_metric; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_metric::metric_Group(&vin, nullptr, &gp);
            }}
        }}
        if (planar_recovery != 0) {{
            ns_planar::Push_0 push{{ref_params_addr}};
            ns_planar::GlobalParams_0 gp{{&push}};
            #pragma omp for collapse(2) schedule(dynamic, 2)
            for (uint32_t gy = 0; gy < planar_gy; ++gy) {{
                for (uint32_t gx = 0; gx < planar_gx; ++gx) {{
                    ComputeVaryingInput vin{{}};
                    vin.startGroupID = {{gx, gy, 0u}};
                    vin.endGroupID = {{gx + 1u, gy + 1u, 1u}};
                    ns_planar::planar_Group(&vin, nullptr, &gp);
                }}
            }}
        }}
    }}
}}

void sf_cpu_run_full_pipeline(
    uint64_t prep_t_addr,
    uint64_t prep_c_addr,
    uint64_t sp_l_addr,
    uint64_t sp_r_addr,
    uint64_t dp_addr,
    uint64_t ref_params_addr,
    uint32_t eye_width,
    uint32_t source_height,
    uint32_t width,
    uint32_t height,
    uint32_t coarse_width,
    uint32_t coarse_height,
    int ultra_fast,
    int planar_recovery
) {{
    uint32_t src_gx = (eye_width + 7u) / 8u, src_gy = (source_height + 7u) / 8u;
    uint32_t tgt_gx = (width + 7u) / 8u, tgt_gy = (height + 7u) / 8u;
    uint32_t crs_gx = (coarse_width + 7u) / 8u, crs_gy = (coarse_height + 7u) / 8u;
    uint32_t n_c = coarse_width * coarse_height;
    uint32_t w8 = (coarse_width >> 1u) > 0u ? (coarse_width >> 1u) : 1u;
    uint32_t h8 = (coarse_height >> 1u) > 0u ? (coarse_height >> 1u) : 1u;
    uint32_t n8 = w8 * h8;
    uint32_t groups_c = (n_c + 63u) / 64u;
    uint32_t groups_8 = (n8 + 63u) / 64u;
    uint32_t groups_c2 = (2u * n_c + 63u) / 64u;
    uint32_t m8_gx = (w8 + 31u) / 32u, m8_gy = (h8 + 3u) / 4u;
    uint32_t m4_gx = (coarse_width + 31u) / 32u, m4_gy = (coarse_height + 3u) / 4u;
    uint32_t n = width * height;
    uint32_t n2 = 2u * n;
    uint32_t groups_both = (n2 + 63u) / 64u;
    uint32_t groups_metric = (ultra_fast != 0) ? ((n + 63u) / 64u) : groups_both;
    uint32_t planar_gx = (width + 15u) / 16u, planar_gy = (height + 15u) / 16u;

    #pragma omp parallel
    {{
        {{
            ns_decode_yuyv::Push_0 push{{prep_t_addr}};
            ns_decode_yuyv::GlobalParams_0 gp{{&push}};
            #pragma omp for collapse(3) schedule(static)
            for (uint32_t gz = 0; gz < 2u; ++gz) {{
                for (uint32_t gy = 0; gy < src_gy; ++gy) {{
                    for (uint32_t gx = 0; gx < src_gx; ++gx) {{
                        ComputeVaryingInput vin{{}};
                        vin.startGroupID = {{gx, gy, gz}};
                        vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                        ns_decode_yuyv::decode_yuyv_Group(&vin, nullptr, &gp);
                    }}
                }}
            }}
        }}
        {{
            ns_rectify_gray::Push_0 push_t{{prep_t_addr}};
            ns_rectify_gray::GlobalParams_0 gp_t{{&push_t}};
            #pragma omp for collapse(3) schedule(static) nowait
            for (uint32_t gz = 0; gz < 2u; ++gz) {{
                for (uint32_t gy = 0; gy < tgt_gy; ++gy) {{
                    for (uint32_t gx = 0; gx < tgt_gx; ++gx) {{
                        ComputeVaryingInput vin{{}};
                        vin.startGroupID = {{gx, gy, gz}};
                        vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                        ns_rectify_gray::rectify_gray_Group(&vin, nullptr, &gp_t);
                    }}
                }}
            }}
            ns_rectify_gray::Push_0 push_c{{prep_c_addr}};
            ns_rectify_gray::GlobalParams_0 gp_c{{&push_c}};
            #pragma omp for collapse(3) schedule(static)
            for (uint32_t gz = 0; gz < 2u; ++gz) {{
                for (uint32_t gy = 0; gy < crs_gy; ++gy) {{
                    for (uint32_t gx = 0; gx < crs_gx; ++gx) {{
                        ComputeVaryingInput vin{{}};
                        vin.startGroupID = {{gx, gy, gz}};
                        vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                        ns_rectify_gray::rectify_gray_Group(&vin, nullptr, &gp_c);
                    }}
                }}
            }}
        }}
        {{
            ns_coarse_prep_1to8_1to4::LayeredPush_0 push_cp{{sp_l_addr, sp_r_addr, 0u, groups_c}};
            ns_coarse_prep_1to8_1to4::GlobalParams_0 gp_cp{{&push_cp}};
            #pragma omp for schedule(static) nowait
            for (uint32_t g = 0; g < groups_c; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_coarse_prep_1to8_1to4::coarse_prep_1to8_1to4_Group(&vin, nullptr, &gp_cp);
            }}
            ns_census::Push_0 push_cen{{ref_params_addr}};
            ns_census::GlobalParams_0 gp_cen{{&push_cen}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_both; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_census::census_Group(&vin, nullptr, &gp_cen);
            }}
        }}
        {{
            ns_coarse_match_1to8::LayeredPush_0 push{{sp_l_addr, sp_r_addr, 0u, m8_gx}};
            ns_coarse_match_1to8::GlobalParams_0 gp{{&push}};
            #pragma omp for collapse(3) schedule(static)
            for (uint32_t gz = 0; gz < 2u; ++gz) {{
                for (uint32_t gy = 0; gy < m8_gy; ++gy) {{
                    for (uint32_t gx = 0; gx < m8_gx; ++gx) {{
                        ComputeVaryingInput vin{{}};
                        vin.startGroupID = {{gx, gy, gz}};
                        vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                        ns_coarse_match_1to8::coarse_match_1to8_Group(&vin, nullptr, &gp);
                    }}
                }}
            }}
        }}
        {{
            ns_coarse_lr_risk_1to8::LayeredPush_0 push{{sp_l_addr, sp_r_addr, 0u, groups_8}};
            ns_coarse_lr_risk_1to8::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_8; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_coarse_lr_risk_1to8::coarse_lr_risk_1to8_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_coarse_selective_fallback_1to4::LayeredPush_0 push{{sp_l_addr, sp_r_addr, 0u, m4_gx}};
            ns_coarse_selective_fallback_1to4::GlobalParams_0 gp{{&push}};
            #pragma omp for collapse(3) schedule(static)
            for (uint32_t gz = 0; gz < 2u; ++gz) {{
                for (uint32_t gy = 0; gy < m4_gy; ++gy) {{
                    for (uint32_t gx = 0; gx < m4_gx; ++gx) {{
                        ComputeVaryingInput vin{{}};
                        vin.startGroupID = {{gx, gy, gz}};
                        vin.endGroupID = {{gx + 1u, gy + 1u, gz + 1u}};
                        ns_coarse_selective_fallback_1to4::coarse_selective_fallback_1to4_Group(&vin, nullptr, &gp);
                    }}
                }}
            }}
        }}
        {{
            ns_lr_consistency::Push_0 push{{dp_addr}};
            ns_lr_consistency::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_c2; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_lr_consistency::lr_consistency_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_match::Push_0 push{{ref_params_addr}};
            ns_match::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_both; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_match::match_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_lr::Push_0 push{{ref_params_addr}};
            ns_lr::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_both; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_lr::lr_Group(&vin, nullptr, &gp);
            }}
        }}
        {{
            ns_metric::Push_0 push{{ref_params_addr}};
            ns_metric::GlobalParams_0 gp{{&push}};
            #pragma omp for schedule(static)
            for (uint32_t g = 0; g < groups_metric; ++g) {{
                ComputeVaryingInput vin{{}};
                vin.startGroupID = {{g, 0u, 0u}};
                vin.endGroupID = {{g + 1u, 1u, 1u}};
                ns_metric::metric_Group(&vin, nullptr, &gp);
            }}
        }}
        if (planar_recovery != 0) {{
            ns_planar::Push_0 push{{ref_params_addr}};
            ns_planar::GlobalParams_0 gp{{&push}};
            #pragma omp for collapse(2) schedule(dynamic, 2)
            for (uint32_t gy = 0; gy < planar_gy; ++gy) {{
                for (uint32_t gx = 0; gx < planar_gx; ++gx) {{
                    ComputeVaryingInput vin{{}};
                    vin.startGroupID = {{gx, gy, 0u}};
                    vin.endGroupID = {{gx + 1u, gy + 1u, 1u}};
                    ns_planar::planar_Group(&vin, nullptr, &gp);
                }}
            }}
        }}
    }}
}}

}}
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "g++",
            "-O3",
            "-march=native",
            "-mtune=native",
            "-funroll-loops",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-fno-semantic-interposition",
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-fopenmp",
            "-Wno-unused-variable",
            "-Wno-unused-but-set-variable",
            str(driver_cpp),
            "-o",
            str(so_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return so_path


class SlangCpuRefinementBackend:
    """Executes the `.slang` stereo & refinement pipelines on CPU via `slangc -target cpp`."""

    def __init__(self) -> None:
        self.so_path = _ensure_cpu_library()
        self._lib = ctypes.CDLL(str(self.so_path))
        self._lib.sf_cpu_dispatch_1d.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self._lib.sf_cpu_dispatch_1d.restype = None
        self._lib.sf_cpu_dispatch_planar.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self._lib.sf_cpu_dispatch_planar.restype = None
        self._lib.sf_cpu_dispatch_preprocess.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self._lib.sf_cpu_dispatch_preprocess.restype = None
        self._lib.sf_cpu_dispatch_hier_1d.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self._lib.sf_cpu_dispatch_hier_1d.restype = None
        self._lib.sf_cpu_dispatch_hier_3d.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        self._lib.sf_cpu_dispatch_hier_3d.restype = None
        self._lib.sf_cpu_run_refinement_pipeline.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.sf_cpu_run_refinement_pipeline.restype = None
        self._lib.sf_cpu_run_full_pipeline.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.sf_cpu_run_full_pipeline.restype = None
        self._ref_ws_cache: dict[tuple[int, int], tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]] = {}
        self._full_ws_cache: dict[tuple[int, int, int, int, int, int], dict[str, Any]] = {}

    def _get_refinement_ws(self, width: int, height: int) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        key = (width, height)
        if key not in self._ref_ws_cache:
            n = width * height
            n2 = 2 * n
            float_planes = {0, 1, 3, 4, 9, 21, 22}
            u64_planes = {7, 8}
            single_eye_planes = {26, 29}
            outputs: Dict[int, np.ndarray] = {}
            for p_idx in range(30):
                if p_idx in (27, 28):
                    continue
                cnt = n if p_idx in single_eye_planes else n2
                if p_idx in float_planes:
                    outputs[p_idx] = np.zeros(cnt, dtype=np.float32)
                elif p_idx in u64_planes:
                    outputs[p_idx] = np.zeros(cnt, dtype=np.int64)
                else:
                    outputs[p_idx] = np.zeros(cnt, dtype=np.uint32)
            completed: Dict[int, np.ndarray] = {}
            for c_idx in range(7):
                if c_idx in (0, 1, 6):
                    completed[c_idx] = np.zeros(n2, dtype=np.float32)
                elif c_idx in (4, 5):
                    completed[c_idx] = np.zeros(n2, dtype=np.int64)
                else:
                    completed[c_idx] = np.zeros(n2, dtype=np.uint32)
            self._ref_ws_cache[key] = (outputs, completed)
        return self._ref_ws_cache[key]

    def run_pipeline(
        self,
        *,
        luma: np.ndarray,
        support: np.ndarray,
        coarse_disparity: np.ndarray,
        coarse_confidence: np.ndarray,
        coarse_validity: np.ndarray,
        width: int,
        height: int,
        coarse_width: int,
        coarse_height: int,
        fx: float,
        baseline: float,
        principal_offset: float,
        radius: int = 4,
        disparities: int = 128,
        min_margin: int = 1,
        subpixel: bool = True,
        ortho_hybrid: bool = True,
        fast_mode: bool = False,
        ultra_fast: bool = False,
        selective_repair: bool = True,
        planar_recovery: bool = True,
        planar_min_anchors: int = 10,
        planar_coarse_affine: bool = True,
        planar_ray_crossing_guard: bool = True,
        planar_coarse_lr_guard: bool = True,
        planar_coarse_lr_tol: float = 2.0,
        planar_patch_luma_tol: float = 12.0,
        planar_max_rmse: float = 0.65,
        prior_floor: float = 0.05,
        max_span: float = 2.0,
        lr_tolerance: float = 1.0,
        source_id: int = 1,
        initial_outputs: Dict[int, np.ndarray] | None = None,
        stages: tuple[int, ...] = (1, 2, 3, 4),
    ) -> Dict[str, np.ndarray]:
        n2 = 2 * width * height
        luma_buf = np.ascontiguousarray(luma, dtype=np.uint32)
        support_buf = np.ascontiguousarray(support, dtype=np.uint32)
        c_disp_buf = np.ascontiguousarray(coarse_disparity, dtype=np.float32)
        c_conf_buf = np.ascontiguousarray(coarse_confidence, dtype=np.float32)
        c_val_buf = np.ascontiguousarray(coarse_validity, dtype=np.uint32)

        ws_outputs, completed = self._get_refinement_ws(width, height)
        outputs: Dict[int, np.ndarray] = dict(ws_outputs)
        outputs[27] = luma_buf
        outputs[28] = support_buf
        if initial_outputs:
            for p_idx, arr in initial_outputs.items():
                outputs[p_idx] = np.ascontiguousarray(arr.copy())

        params = _CpuRefinementParameters()
        params.coarse[0] = c_disp_buf.ctypes.data
        params.coarse[1] = c_conf_buf.ctypes.data
        params.coarse[2] = c_val_buf.ctypes.data
        for p_idx in range(30):
            params.output[p_idx] = outputs[p_idx].ctypes.data
        if planar_recovery:
            for c_idx in range(7):
                params.completed[c_idx] = completed[c_idx].ctypes.data

        params.width = width
        params.height = height
        params.source_width = width
        params.source_height = height
        params.coarse_width = coarse_width
        params.coarse_height = coarse_height
        params.radius = radius
        params.disparities = disparities
        params.min_margin = min_margin
        params.subpixel = (
            (1 if subpixel else 0)
            | (2 if ortho_hybrid else 0)
            | (4 if fast_mode else 0)
            | (8 if selective_repair else 0)
            | (16 if ultra_fast else 0)
        )
        params.prior_floor = prior_floor
        params.max_span = max_span
        params.lr_tolerance = lr_tolerance
        params.fx = fx
        params.baseline = baseline
        params.principal_offset = principal_offset
        params.source_low = source_id & 0xFFFFFFFF
        params.source_high = (source_id >> 32) & 0xFFFFFFFF
        params.planar_cfg = (
            (planar_min_anchors & 0xFF)
            | (256 if planar_coarse_affine else 0)
            | (512 if planar_ray_crossing_guard else 0)
            | (1024 if planar_coarse_lr_guard else 0)
        )
        params.planar_coarse_lr_tol = planar_coarse_lr_tol
        params.planar_patch_luma_tol = planar_patch_luma_tol
        params.planar_max_rmse = planar_max_rmse

        params_addr = ctypes.addressof(params)
        if stages == (1, 2, 3, 4):
            self._lib.sf_cpu_run_refinement_pipeline(
                params_addr,
                width,
                height,
                1 if ultra_fast else 0,
                1 if planar_recovery else 0,
            )
        else:
            for stage_id in stages:
                self._lib.sf_cpu_dispatch_1d(stage_id, params_addr, n2)
            if planar_recovery:
                self._lib.sf_cpu_dispatch_planar(params_addr, width, height)

        return {
            "census": outputs[24],
            "census_validity": outputs[25],
            "raw_disparity": outputs[21],
            "raw_confidence": outputs[22],
            "raw_validity": outputs[23],
            "disparity": outputs[0],
            "confidence": outputs[1],
            "validity": outputs[2],
            "depth": outputs[3],
            "metric_confidence": outputs[4],
            "metric_validity": outputs[5],
            "provenance": outputs[6],
            "completed_depth": completed[0] if planar_recovery else outputs[3].copy(),
            "completed_confidence": completed[1] if planar_recovery else outputs[4].copy(),
            "completed_validity": completed[2] if planar_recovery else outputs[5].copy(),
            "completed_provenance": completed[3] if planar_recovery else outputs[6].copy(),
            "completed_disparity": completed[6] if planar_recovery else outputs[0].copy(),
        }

    def run_full_camera_to_depth_pipeline(
        self,
        *,
        raw_yuyv: np.ndarray,
        target_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        coarse_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        eye_width: int,
        source_height: int,
        stride_bytes: int,
        width: int,
        height: int,
        coarse_width: int,
        coarse_height: int,
        fx: float,
        baseline: float,
        principal_offset: float,
        coarse_fx: float | None = None,
        coarse_principal_offset: float | None = None,
        disparities: int = 128,
        radius: int = 2,
        subpixel: bool = True,
        ortho_hybrid: bool = True,
        fast_mode: bool = True,
        ultra_fast: bool = True,
        selective_repair: bool = True,
        planar_recovery: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Executes the complete camera-to-depth pipeline on CPU via `slangc -target cpp`:
        1. `preprocess.slang::decode_yuyv` (raw YUYV -> decoded RGBA left/right)
        2. `preprocess.slang::rectify_gray` (coarse 1/4-scale and target full-res rectification + luma + support)
        3. `stereo_hierarchical.slang` (`coarse_prep_1to8_1to4` -> `coarse_match_1to8` -> `coarse_lr_risk_1to8` -> `coarse_selective_fallback_1to4`)
        4. `depth.slang` (`lr_consistency` at coarse scale)
        5. `refinement.slang` (`census` -> `match` -> `lr` -> `metric` -> `planar` at target scale)
        All stages execute inside a single fused OpenMP parallel region (`sf_cpu_run_full_pipeline`).
        """
        raw_buf = np.ascontiguousarray(raw_yuyv, dtype=np.uint8)
        ws_key = (eye_width, source_height, width, height, coarse_width, coarse_height)
        if ws_key not in self._full_ws_cache:
            n_src = eye_width * source_height
            n_t = width * height
            n_c = coarse_width * coarse_height
            w8 = max(1, coarse_width >> 1)
            h8 = max(1, coarse_height >> 1)
            self._full_ws_cache[ws_key] = {
                "orig_l": np.zeros(n_src, dtype=np.uint32),
                "orig_r": np.zeros(n_src, dtype=np.uint32),
                "rect_t_l": np.zeros(n_t, dtype=np.uint32),
                "rect_t_r": np.zeros(n_t, dtype=np.uint32),
                "luma_t": np.zeros(2 * n_t, dtype=np.uint32),
                "supp_t": np.zeros(2 * n_t, dtype=np.uint32),
                "rect_c_l": np.zeros(n_c, dtype=np.uint32),
                "rect_c_r": np.zeros(n_c, dtype=np.uint32),
                "luma_c_l": np.zeros(n_c, dtype=np.uint32),
                "luma_c_r": np.zeros(n_c, dtype=np.uint32),
                "supp_c_l": np.zeros(n_c, dtype=np.uint32),
                "supp_c_r": np.zeros(n_c, dtype=np.uint32),
                "desc4_l": np.zeros(n_c, dtype=np.uint32),
                "desc4_r": np.zeros(n_c, dtype=np.uint32),
                "valid4_l": np.zeros(n_c, dtype=np.uint32),
                "valid4_r": np.zeros(n_c, dtype=np.uint32),
                "raw_c_disp_l": np.zeros(n_c, dtype=np.float32),
                "raw_c_disp_r": np.zeros(n_c, dtype=np.float32),
                "raw_c_conf_l": np.zeros(n_c, dtype=np.float32),
                "raw_c_conf_r": np.zeros(n_c, dtype=np.float32),
                "raw_c_val_l": np.zeros(n_c, dtype=np.uint32),
                "raw_c_val_r": np.zeros(n_c, dtype=np.uint32),
                "pen_scratch": np.zeros(16 * w8 * h8, dtype=np.uint32),
                "c_lr_disp": np.zeros(2 * n_c, dtype=np.float32),
                "c_lr_conf": np.zeros(2 * n_c, dtype=np.float32),
                "c_lr_val": np.zeros(2 * n_c, dtype=np.uint32),
                "c_depth": np.zeros(2 * n_c, dtype=np.float32),
                "c_mconf": np.zeros(2 * n_c, dtype=np.float32),
                "c_mval": np.zeros(2 * n_c, dtype=np.uint32),
                "c_prov": np.zeros(2 * n_c, dtype=np.uint32),
                "c_sids": np.zeros(2 * n_c, dtype=np.int64),
                "c_age": np.zeros(2 * n_c, dtype=np.int64),
                "t_maps": [np.ascontiguousarray(m, dtype=np.float32) for m in target_maps],
                "c_maps": [np.ascontiguousarray(m, dtype=np.float32) for m in coarse_maps],
            }
        ws = self._full_ws_cache[ws_key]
        n_t = width * height
        luma_t = ws["luma_t"]
        supp_t = ws["supp_t"]

        prep_t = _CpuPreprocessParameters()
        prep_t.raw = raw_buf.ctypes.data
        prep_t.original[0] = ws["orig_l"].ctypes.data
        prep_t.original[1] = ws["orig_r"].ctypes.data
        prep_t.rectified[0] = ws["rect_t_l"].ctypes.data
        prep_t.rectified[1] = ws["rect_t_r"].ctypes.data
        prep_t.luma[0] = luma_t[:n_t].ctypes.data
        prep_t.luma[1] = luma_t[n_t:].ctypes.data
        prep_t.support[0] = supp_t[:n_t].ctypes.data
        prep_t.support[1] = supp_t[n_t:].ctypes.data
        for k in range(4):
            prep_t.maps[k] = ws["t_maps"][k].ctypes.data
        prep_t.eye_width = eye_width
        prep_t.source_height = source_height
        prep_t.stride = stride_bytes
        prep_t.width = width
        prep_t.height = height

        prep_c = _CpuPreprocessParameters()
        prep_c.raw = raw_buf.ctypes.data
        prep_c.original[0] = ws["orig_l"].ctypes.data
        prep_c.original[1] = ws["orig_r"].ctypes.data
        prep_c.rectified[0] = ws["rect_c_l"].ctypes.data
        prep_c.rectified[1] = ws["rect_c_r"].ctypes.data
        prep_c.luma[0] = ws["luma_c_l"].ctypes.data
        prep_c.luma[1] = ws["luma_c_r"].ctypes.data
        prep_c.support[0] = ws["supp_c_l"].ctypes.data
        prep_c.support[1] = ws["supp_c_r"].ctypes.data
        for k in range(4):
            prep_c.maps[k] = ws["c_maps"][k].ctypes.data
        prep_c.eye_width = eye_width
        prep_c.source_height = source_height
        prep_c.stride = stride_bytes
        prep_c.width = coarse_width
        prep_c.height = coarse_height

        scale_factor = 4 if width == coarse_width * 4 else 2
        coarse_disp = max(1, disparities // scale_factor)
        sp_l = _CpuStereoParameters()
        sp_r = _CpuStereoParameters()
        for sp, d_out, c_out, v_out in (
            (sp_l, ws["raw_c_disp_l"], ws["raw_c_conf_l"], ws["raw_c_val_l"]),
            (sp_r, ws["raw_c_disp_r"], ws["raw_c_conf_r"], ws["raw_c_val_r"]),
        ):
            sp.left_address = ws["luma_c_l"].ctypes.data
            sp.right_address = ws["luma_c_r"].ctypes.data
            sp.left_desc_address = ws["desc4_l"].ctypes.data
            sp.right_desc_address = ws["desc4_r"].ctypes.data
            sp.left_valid_address = ws["valid4_l"].ctypes.data
            sp.right_valid_address = ws["valid4_r"].ctypes.data
            sp.disparity_address = d_out.ctypes.data
            sp.confidence_address = c_out.ctypes.data
            sp.validity_address = v_out.ctypes.data
            sp.width = coarse_width
            sp.height = coarse_height
            sp.disparities = coarse_disp
            sp.subpixel = (1 if subpixel else 0) | (2 if ortho_hybrid else 0)
            sp.penalties_address = ws["pen_scratch"].ctypes.data

        dp = _CpuDepthParameters()
        dp.inputs[0] = ws["raw_c_disp_l"].ctypes.data
        dp.inputs[1] = ws["raw_c_conf_l"].ctypes.data
        dp.inputs[2] = ws["raw_c_val_l"].ctypes.data
        dp.inputs[3] = ws["raw_c_disp_r"].ctypes.data
        dp.inputs[4] = ws["raw_c_conf_r"].ctypes.data
        dp.inputs[5] = ws["raw_c_val_r"].ctypes.data
        dp.inputs[6] = ws["supp_c_l"].ctypes.data
        dp.inputs[7] = ws["supp_c_r"].ctypes.data
        dp.outputs[0] = ws["c_lr_disp"].ctypes.data
        dp.outputs[1] = ws["c_lr_conf"].ctypes.data
        dp.outputs[2] = ws["c_lr_val"].ctypes.data
        dp.outputs[3] = ws["c_depth"].ctypes.data
        dp.outputs[4] = ws["c_mconf"].ctypes.data
        dp.outputs[5] = ws["c_mval"].ctypes.data
        dp.outputs[6] = ws["c_prov"].ctypes.data
        dp.outputs[7] = ws["c_sids"].ctypes.data
        dp.outputs[8] = ws["c_age"].ctypes.data
        dp.width = coarse_width
        dp.height = coarse_height
        dp.tolerance = 1.0
        dp.fx = coarse_fx if coarse_fx is not None else (fx / float(scale_factor))
        dp.baseline = baseline
        dp.principal_offset = (
            coarse_principal_offset
            if coarse_principal_offset is not None
            else (principal_offset / float(scale_factor))
        )
        dp.source_low = 1
        dp.source_high = 0

        ws_outputs, completed = self._get_refinement_ws(width, height)
        outputs: Dict[int, np.ndarray] = dict(ws_outputs)
        outputs[27] = luma_t
        outputs[28] = supp_t

        ref_p = _CpuRefinementParameters()
        ref_p.coarse[0] = ws["c_lr_disp"].ctypes.data
        ref_p.coarse[1] = ws["c_lr_conf"].ctypes.data
        ref_p.coarse[2] = ws["c_lr_val"].ctypes.data
        for p_idx in range(30):
            ref_p.output[p_idx] = outputs[p_idx].ctypes.data
        if planar_recovery:
            for c_idx in range(7):
                ref_p.completed[c_idx] = completed[c_idx].ctypes.data
        ref_p.width = width
        ref_p.height = height
        ref_p.source_width = width
        ref_p.source_height = height
        ref_p.coarse_width = coarse_width
        ref_p.coarse_height = coarse_height
        ref_p.radius = radius
        ref_p.disparities = disparities
        ref_p.min_margin = 1
        ref_p.subpixel = (
            (1 if subpixel else 0)
            | (2 if ortho_hybrid else 0)
            | (4 if fast_mode else 0)
            | (8 if selective_repair else 0)
            | (16 if ultra_fast else 0)
        )
        ref_p.prior_floor = 0.05
        ref_p.max_span = 2.0
        ref_p.lr_tolerance = 1.0
        ref_p.fx = fx
        ref_p.baseline = baseline
        ref_p.principal_offset = principal_offset
        ref_p.source_low = 1
        ref_p.source_high = 0
        ref_p.planar_cfg = 10 | 256 | 512 | 1024
        ref_p.planar_coarse_lr_tol = 2.0
        ref_p.planar_patch_luma_tol = 12.0
        ref_p.planar_max_rmse = 0.65

        self._lib.sf_cpu_run_full_pipeline(
            ctypes.addressof(prep_t),
            ctypes.addressof(prep_c),
            ctypes.addressof(sp_l),
            ctypes.addressof(sp_r),
            ctypes.addressof(dp),
            ctypes.addressof(ref_p),
            eye_width,
            source_height,
            width,
            height,
            coarse_width,
            coarse_height,
            1 if ultra_fast else 0,
            1 if planar_recovery else 0,
        )

        return {
            "census": outputs[24],
            "census_validity": outputs[25],
            "raw_disparity": outputs[21],
            "raw_confidence": outputs[22],
            "raw_validity": outputs[23],
            "disparity": outputs[0],
            "confidence": outputs[1],
            "validity": outputs[2],
            "depth": outputs[3],
            "metric_confidence": outputs[4],
            "metric_validity": outputs[5],
            "provenance": outputs[6],
            "completed_depth": completed[0] if planar_recovery else outputs[3].copy(),
            "completed_confidence": completed[1] if planar_recovery else outputs[4].copy(),
            "completed_validity": completed[2] if planar_recovery else outputs[5].copy(),
            "completed_provenance": completed[3] if planar_recovery else outputs[6].copy(),
            "completed_disparity": completed[6] if planar_recovery else outputs[0].copy(),
            "coarse_disparity": ws["c_lr_disp"],
            "coarse_validity": ws["c_lr_val"],
            "rectified_left": ws["rect_t_l"],
            "rectified_right": ws["rect_t_r"],
        }
