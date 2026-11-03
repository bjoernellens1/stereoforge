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
REFINEMENT_OPS_SLANG = ROOT / "python" / "stereoforge" / "shaders" / "ops" / "refinement.slang"


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
    for p in (REFINEMENT_SLANG, PREPROCESS_SLANG, REFINEMENT_OPS_SLANG):
        h.update(p.read_bytes())
    digest = h.hexdigest()[:16]
    so_path = cache_dir / f"libstereoforge_slang_cpu_{digest}.so"
    if so_path.exists():
        return so_path

    entries = ("rectify", "census", "match", "lr", "metric", "planar")
    generated_sections = []
    for entry in entries:
        cpp_out = cache_dir / f"gen_{entry}.cpp"
        subprocess.run(
            [
                str(SLANGC),
                str(REFINEMENT_SLANG),
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
        }}
    }}
    #undef SF_CALL_1D
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
