"""Build and structural checks for the read-only Vulkan property probe."""
from pathlib import Path
import shutil
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "apps/vulkan_external_capabilities.cpp"


def test_probe_queries_the_intended_handles_and_usage():
    source = SOURCE.read_text()
    for required in (
        "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT",
        "VK_BUFFER_USAGE_TRANSFER_SRC_BIT",
        "VK_BUFFER_USAGE_TRANSFER_DST_BIT",
        "VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT",
        "VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT",
        "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT",
        "VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT",
        "VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT",
        "vkGetPhysicalDeviceExternalBufferProperties",
        "vkGetPhysicalDeviceExternalSemaphoreProperties",
        "exportFromImportedHandleTypes",
        "timeline_compatibility",
        "not_queried",
    ):
        assert required in source
    # A properties query must never evolve into allocation, dispatch or IPC.
    for forbidden in ("vkCreateDevice(", "vkAllocateMemory(", "vkCreateBuffer(",
                      "vkCreateSemaphore(", "vkQueueSubmit(", "vkGetMemoryFdKHR(",
                      "vkImportSemaphoreFdKHR("):
        assert forbidden not in source


def test_probe_compiles_without_gpu():
    compiler = shutil.which("c++")
    if not compiler or not Path("/usr/include/vulkan/vulkan.h").exists():
        pytest.skip("C++ compiler or Vulkan development headers unavailable")
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(SOURCE)],
        check=True,
        capture_output=True,
        text=True,
    )
