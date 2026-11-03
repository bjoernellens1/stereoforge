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
from typing import Dict

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
        # Wrap each generated entrypoint in its own namespace so symbol names do not collide.
        lines = [
            line
            for line in src.splitlines()
            if not line.strip().startswith("#include")
        ]
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
    if (n > 8) omp_set_num_threads(8);
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

}}
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "g++",
            "-O3",
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
    """Executes the `.slang` refinement pipeline on CPU via `slangc -target cpp`."""

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

        float_planes = {0, 1, 3, 4, 9, 21, 22}
        u64_planes = {7, 8}
        single_eye_planes = {26, 29}
        outputs: Dict[int, np.ndarray] = {}
        for p_idx in range(30):
            cnt = (width * height) if p_idx in single_eye_planes else n2
            if initial_outputs and p_idx in initial_outputs:
                outputs[p_idx] = np.ascontiguousarray(initial_outputs[p_idx].copy())
            elif p_idx == 27:
                outputs[p_idx] = luma_buf
            elif p_idx == 28:
                outputs[p_idx] = support_buf
            elif p_idx in float_planes:
                outputs[p_idx] = np.zeros(cnt, dtype=np.float32)
            elif p_idx in u64_planes:
                outputs[p_idx] = np.zeros(cnt, dtype=np.int64)
            else:
                outputs[p_idx] = np.zeros(cnt, dtype=np.uint32)

        completed: Dict[int, np.ndarray] = {}
        if planar_recovery:
            for c_idx in range(7):
                if c_idx in (0, 1, 6):
                    completed[c_idx] = np.zeros(n2, dtype=np.float32)
                elif c_idx in (4, 5):
                    completed[c_idx] = np.zeros(n2, dtype=np.int64)
                else:
                    completed[c_idx] = np.zeros(n2, dtype=np.uint32)

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
        4. `depth.slang` (`lr_consistency` -> `metric_observation` at coarse scale)
        5. `refinement.slang` (`census` -> `match` -> `lr` -> `metric` -> `planar` at target scale)
        """
        raw_buf = np.ascontiguousarray(raw_yuyv, dtype=np.uint8)
        n_src = eye_width * source_height
        orig_l = np.zeros(n_src, dtype=np.uint32)
        orig_r = np.zeros(n_src, dtype=np.uint32)

        n_t = width * height
        rect_t_l = np.zeros(n_t, dtype=np.uint32)
        rect_t_r = np.zeros(n_t, dtype=np.uint32)
        luma_t = np.zeros(2 * n_t, dtype=np.uint32)
        supp_t = np.zeros(2 * n_t, dtype=np.uint32)
        t_maps = [np.ascontiguousarray(m, dtype=np.float32) for m in target_maps]

        prep_t = _CpuPreprocessParameters()
        prep_t.raw = raw_buf.ctypes.data
        prep_t.original[0] = orig_l.ctypes.data
        prep_t.original[1] = orig_r.ctypes.data
        prep_t.rectified[0] = rect_t_l.ctypes.data
        prep_t.rectified[1] = rect_t_r.ctypes.data
        prep_t.luma[0] = luma_t[:n_t].ctypes.data
        prep_t.luma[1] = luma_t[n_t:].ctypes.data
        prep_t.support[0] = supp_t[:n_t].ctypes.data
        prep_t.support[1] = supp_t[n_t:].ctypes.data
        for k in range(4):
            prep_t.maps[k] = t_maps[k].ctypes.data
        prep_t.eye_width = eye_width
        prep_t.source_height = source_height
        prep_t.stride = stride_bytes
        prep_t.width = width
        prep_t.height = height

        # Stage 1: decode_yuyv + full-res rectify_gray
        prep_t_addr = ctypes.addressof(prep_t)
        self._lib.sf_cpu_dispatch_preprocess(0, prep_t_addr, eye_width, source_height)
        self._lib.sf_cpu_dispatch_preprocess(1, prep_t_addr, width, height)

        # Stage 2: coarse rectify_gray (reuses decoded orig_l / orig_r!)
        n_c = coarse_width * coarse_height
        rect_c_l = np.zeros(n_c, dtype=np.uint32)
        rect_c_r = np.zeros(n_c, dtype=np.uint32)
        luma_c_l = np.zeros(n_c, dtype=np.uint32)
        luma_c_r = np.zeros(n_c, dtype=np.uint32)
        supp_c_l = np.zeros(n_c, dtype=np.uint32)
        supp_c_r = np.zeros(n_c, dtype=np.uint32)
        c_maps = [np.ascontiguousarray(m, dtype=np.float32) for m in coarse_maps]

        prep_c = _CpuPreprocessParameters()
        prep_c.raw = raw_buf.ctypes.data
        prep_c.original[0] = orig_l.ctypes.data
        prep_c.original[1] = orig_r.ctypes.data
        prep_c.rectified[0] = rect_c_l.ctypes.data
        prep_c.rectified[1] = rect_c_r.ctypes.data
        prep_c.luma[0] = luma_c_l.ctypes.data
        prep_c.luma[1] = luma_c_r.ctypes.data
        prep_c.support[0] = supp_c_l.ctypes.data
        prep_c.support[1] = supp_c_r.ctypes.data
        for k in range(4):
            prep_c.maps[k] = c_maps[k].ctypes.data
        prep_c.eye_width = eye_width
        prep_c.source_height = source_height
        prep_c.stride = stride_bytes
        prep_c.width = coarse_width
        prep_c.height = coarse_height
        self._lib.sf_cpu_dispatch_preprocess(1, ctypes.addressof(prep_c), coarse_width, coarse_height)

        # Stage 3: stereo_hierarchical.slang (1/8-scale + selective 1/4-scale fallback)
        scale_factor = 4 if width == coarse_width * 4 else 2
        coarse_disp = max(1, disparities // scale_factor)
        desc4_l = np.zeros(n_c, dtype=np.uint32)
        desc4_r = np.zeros(n_c, dtype=np.uint32)
        valid4_l = np.zeros(n_c, dtype=np.uint32)
        valid4_r = np.zeros(n_c, dtype=np.uint32)
        raw_c_disp_l = np.zeros(n_c, dtype=np.float32)
        raw_c_disp_r = np.zeros(n_c, dtype=np.float32)
        raw_c_conf_l = np.zeros(n_c, dtype=np.float32)
        raw_c_conf_r = np.zeros(n_c, dtype=np.float32)
        raw_c_val_l = np.zeros(n_c, dtype=np.uint32)
        raw_c_val_r = np.zeros(n_c, dtype=np.uint32)
        w8 = max(1, coarse_width >> 1)
        h8 = max(1, coarse_height >> 1)
        pen_Scratch = np.zeros(16 * w8 * h8, dtype=np.uint32)

        sp_l = _CpuStereoParameters()
        sp_r = _CpuStereoParameters()
        for sp, d_out, c_out, v_out in (
            (sp_l, raw_c_disp_l, raw_c_conf_l, raw_c_val_l),
            (sp_r, raw_c_disp_r, raw_c_conf_r, raw_c_val_r),
        ):
            sp.left_address = luma_c_l.ctypes.data
            sp.right_address = luma_c_r.ctypes.data
            sp.left_desc_address = desc4_l.ctypes.data
            sp.right_desc_address = desc4_r.ctypes.data
            sp.left_valid_address = valid4_l.ctypes.data
            sp.right_valid_address = valid4_r.ctypes.data
            sp.disparity_address = d_out.ctypes.data
            sp.confidence_address = c_out.ctypes.data
            sp.validity_address = v_out.ctypes.data
            sp.width = coarse_width
            sp.height = coarse_height
            sp.disparities = coarse_disp
            sp.subpixel = (1 if subpixel else 0) | (2 if ortho_hybrid else 0)
            sp.penalties_address = pen_Scratch.ctypes.data

        lp_addr = ctypes.addressof(sp_l)
        rp_addr = ctypes.addressof(sp_r)
        self._lib.sf_cpu_dispatch_hier_1d(0, lp_addr, rp_addr, n_c)
        self._lib.sf_cpu_dispatch_hier_3d(0, lp_addr, rp_addr, w8, h8)
        self._lib.sf_cpu_dispatch_hier_1d(1, lp_addr, rp_addr, w8 * h8)
        self._lib.sf_cpu_dispatch_hier_3d(1, lp_addr, rp_addr, coarse_width, coarse_height)

        # Stage 4: depth.slang (coarse LR consistency + metric observation)
        c_lr_disp = np.zeros(2 * n_c, dtype=np.float32)
        c_lr_conf = np.zeros(2 * n_c, dtype=np.float32)
        c_lr_val = np.zeros(2 * n_c, dtype=np.uint32)
        c_depth = np.zeros(2 * n_c, dtype=np.float32)
        c_mconf = np.zeros(2 * n_c, dtype=np.float32)
        c_mval = np.zeros(2 * n_c, dtype=np.uint32)
        c_prov = np.zeros(2 * n_c, dtype=np.uint32)
        c_sids = np.zeros(2 * n_c, dtype=np.int64)
        c_age = np.zeros(2 * n_c, dtype=np.int64)

        dp = _CpuDepthParameters()
        dp.inputs[0] = raw_c_disp_l.ctypes.data
        dp.inputs[1] = raw_c_conf_l.ctypes.data
        dp.inputs[2] = raw_c_val_l.ctypes.data
        dp.inputs[3] = raw_c_disp_r.ctypes.data
        dp.inputs[4] = raw_c_conf_r.ctypes.data
        dp.inputs[5] = raw_c_val_r.ctypes.data
        dp.inputs[6] = supp_c_l.ctypes.data
        dp.inputs[7] = supp_c_r.ctypes.data
        dp.outputs[0] = c_lr_disp.ctypes.data
        dp.outputs[1] = c_lr_conf.ctypes.data
        dp.outputs[2] = c_lr_val.ctypes.data
        dp.outputs[3] = c_depth.ctypes.data
        dp.outputs[4] = c_mconf.ctypes.data
        dp.outputs[5] = c_mval.ctypes.data
        dp.outputs[6] = c_prov.ctypes.data
        dp.outputs[7] = c_sids.ctypes.data
        dp.outputs[8] = c_age.ctypes.data
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

        dp_addr = ctypes.addressof(dp)
        self._lib.sf_cpu_dispatch_1d(5, dp_addr, 2 * n_c)
        self._lib.sf_cpu_dispatch_1d(6, dp_addr, 2 * n_c)

        # Stage 5: refinement.slang (target full-res census -> match -> lr -> metric -> planar)
        out = self.run_pipeline(
            luma=luma_t,
            support=supp_t,
            coarse_disparity=c_lr_disp,
            coarse_confidence=c_lr_conf,
            coarse_validity=c_lr_val,
            width=width,
            height=height,
            coarse_width=coarse_width,
            coarse_height=coarse_height,
            fx=fx,
            baseline=baseline,
            principal_offset=principal_offset,
            radius=radius,
            disparities=disparities,
            subpixel=subpixel,
            ortho_hybrid=ortho_hybrid,
            fast_mode=fast_mode,
            ultra_fast=ultra_fast,
            selective_repair=selective_repair,
            planar_recovery=planar_recovery,
        )
        out["coarse_disparity"] = c_lr_disp
        out["coarse_validity"] = c_lr_val
        out["rectified_left"] = rect_t_l
        out["rectified_right"] = rect_t_r
        return out
