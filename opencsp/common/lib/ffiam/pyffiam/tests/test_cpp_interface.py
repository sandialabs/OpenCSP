# Copyright Sandia National Laboratories. All rights reserved.

"""Unit tests for pyffiam.cpp_interface: MemoryPool lifecycle, AnalysisParams
marshalling, library-load failure modes, and ctypes signature wiring.

These tests stay below run_analysis() — they don't require building the FFIAM
DLL or invoking a kernel. The few that do load the DLL (signature checks) are
gated by skipping when the external/ directory hasn't been populated.
"""

from __future__ import annotations

import ctypes as cts
import sys
from pathlib import Path

import pytest


# =============================================================================
# AnalysisParams: dataclass field mapping
# =============================================================================

def _make_params(**overrides):
    """Build a minimal AnalysisParams with sensible defaults for all fields."""
    from pyffiam.cpp_interface import AnalysisParams
    defaults = dict(
        year=2025, month=6, day=21, hour=12.0,
        latitude=34.96, longitude=-106.51, timezone=-7.0,
        field_radius=600, min_height=4, max_height=100, tower_height=61.0,
        heliostat_file="heliostats.csv", num_heliostats=218,
        facet_file="facets.csv", num_facets=25, num_facet_cols=5,
        facet_width=1.2, facet_height=1.2,
        aim_strategy=1, aim_param_0=0.0, aim_param_1=0.0, aim_param_2=90.0,
        aim_file="",
        reflectivity=0.9, peak_dni=0.1, beta=0.0094,
        voxel_size=2, ambient=0.0,
        min_attenuation=1.0, irrad_exponent=2.0, n_rays_per_facet=12.0,
        flux_correction_scale=1.0, pre_focal_scale=1.0,
        num_voxels=100000,
    )
    defaults.update(overrides)
    return AnalysisParams(**defaults)


def test_analysis_params_round_trip_fields():
    """All AnalysisParams fields must round-trip with their declared types."""
    p = _make_params(hour=10.5, aim_strategy=6, num_heliostats=42)
    assert p.year == 2025
    assert p.hour == pytest.approx(10.5)
    assert p.aim_strategy == 6
    assert p.num_heliostats == 42
    # File fields stay as strings (encoded only at the ctypes boundary).
    assert isinstance(p.heliostat_file, str)
    assert isinstance(p.aim_file, str)


def test_analysis_params_str_fields_encode_to_bytes():
    """heliostat_file / facet_file / aim_file must encode to bytes for c_char_p."""
    p = _make_params(heliostat_file="nsttf.csv", aim_file="aims.csv")
    # This mirrors what _call_analysis does internally.
    encoded = p.heliostat_file.encode()
    assert isinstance(encoded, bytes)
    assert encoded == b"nsttf.csv"
    assert p.aim_file.encode() == b"aims.csv"


# =============================================================================
# MemoryPool: allocation, free, exception-safety, double-use guarding
# =============================================================================

@pytest.fixture
def libc():
    """Real libc handle so MemoryPool's malloc/free actually run."""
    libc_name = "msvcrt" if sys.platform == "win32" else "libc.so.6"
    libc = cts.CDLL(libc_name)
    libc.malloc.restype = cts.c_void_p
    return libc


def test_memory_pool_allocates_and_frees(libc):
    from pyffiam.cpp_interface import MemoryPool
    with MemoryPool(libc, 4096, name="t") as pool:
        # Inside the context: ptr is valid and not None.
        assert pool.ptr is not None
        assert pool.ptr.value is not None
        assert pool.size.value == 4096
    # After exit: ptr access must raise.
    with pytest.raises(RuntimeError, match="not allocated or already freed"):
        _ = pool.ptr


def test_memory_pool_ptr_before_enter_raises(libc):
    """Accessing .ptr before the context manager is entered must raise."""
    from pyffiam.cpp_interface import MemoryPool
    pool = MemoryPool(libc, 4096, name="t")
    with pytest.raises(RuntimeError, match="not allocated"):
        _ = pool.ptr


def test_memory_pool_frees_on_exception(libc):
    """If an exception is raised inside the context, memory must still be freed
    and the pool's internal ptr must reset to None (so a later ptr access
    raises the standard "not allocated" RuntimeError)."""
    from pyffiam.cpp_interface import MemoryPool
    pool = MemoryPool(libc, 4096, name="t")
    with pytest.raises(ValueError):
        with pool:
            assert pool.ptr is not None
            raise ValueError("simulated failure")
    with pytest.raises(RuntimeError, match="not allocated"):
        _ = pool.ptr


def test_memory_pool_size_returns_c_uint(libc):
    """The .size property must return a ctypes c_uint for PyAnalysis."""
    from pyffiam.cpp_interface import MemoryPool
    pool = MemoryPool(libc, 1024 * 1024, name="t")
    s = pool.size
    assert isinstance(s, cts.c_uint)
    assert s.value == 1024 * 1024


# =============================================================================
# FFIAMLibrary load behavior
# =============================================================================

def test_cuda_available_respects_pretend_env(monkeypatch):
    """The FFIAM_TEST_PRETEND_NO_CUDA escape hatch must short-circuit detection."""
    from pyffiam.cpp_interface import FFIAMLibrary
    monkeypatch.setenv("FFIAM_TEST_PRETEND_NO_CUDA", "1")
    assert FFIAMLibrary._cuda_available() is False


def test_cpu_library_missing_raises_clear_error(tmp_path, monkeypatch):
    """If both backends fail to locate their shared library, the loader must
    raise OSError with a message pointing at the missing CPU artifact (since
    the CPU fallback is the final attempt)."""
    from pyffiam.cpp_interface import FFIAMLibrary
    monkeypatch.setenv("FFIAM_TEST_PRETEND_NO_CUDA", "1")
    # tmp_path is empty -> no CPU library will be found there.
    with pytest.raises(OSError, match="CPU library not found"):
        FFIAMLibrary(dll_dir=tmp_path, force_cpu=True)


def test_force_cpu_true_skips_cuda_branch(monkeypatch, tmp_path):
    """force_cpu=True must not even attempt the CUDA load path. We verify this
    indirectly by ensuring the OSError mentions the CPU library only — the
    CUDA branch would have produced a different message first."""
    from pyffiam.cpp_interface import FFIAMLibrary
    # Even with CUDA pretend-available, force_cpu bypasses the CUDA path.
    monkeypatch.delenv("FFIAM_TEST_PRETEND_NO_CUDA", raising=False)
    with pytest.raises(OSError) as exc_info:
        FFIAMLibrary(dll_dir=tmp_path, force_cpu=True)
    assert "CPU library not found" in str(exc_info.value)


# =============================================================================
# ctypes signature wiring (requires the DLL to load)
# =============================================================================

def _external_has_cpu_lib() -> bool:
    """The CPU shared library is the most reliably present artifact."""
    name = "ffiam_lib_cpu.dll" if sys.platform == "win32" else "libffiam_lib_cpu.so"
    return (Path(__file__).parents[1] / "src" / "pyffiam" / "external" / name).exists()


@pytest.mark.skipif(not _external_has_cpu_lib(),
                    reason="ffiam_lib_cpu not built")
def test_py_analysis_argtypes_count_matches_signature():
    """PyAnalysis must be configured with the full argument list. If anyone
    adds or drops a parameter in C++ without updating _setup_signatures, this
    test fails. Count is asserted exactly to catch silent drift."""
    from pyffiam.cpp_interface import FFIAMLibrary
    lib = FFIAMLibrary(force_cpu=True)
    sig = lib._ffiam.PyAnalysis
    # 4 pool (void*, uint, void*, uint)
    # + 4 time (year, month, day, hour)
    # + 3 location (lat, lng, timezone)
    # + 4 field (radius, zMin, zMax, towerH)
    # + 2 helio (file, nHelios)
    # + 5 facet (file, nFacets, nFacetCols, facetW, facetH)
    # + 5 aim (stratId, aim1, aim2, aim3, aimFile)
    # + 3 optical (refl, dni, beta)
    # + 7 analysis (voxelSize, ambient, minAttenuation, irradExponent,
    #              nRaysPerFacet, fluxCorrectionScale, preFocalScale)
    # + 2 trailing flags (verbose, useCpu) = 39.
    assert len(sig.argtypes) == 39
    assert sig.restype is cts.c_int


@pytest.mark.skipif(not _external_has_cpu_lib(),
                    reason="ffiam_lib_cpu not built")
def test_py_analysis_first_args_are_pool_pointers():
    """First four ctypes args are (void*, c_uint, void*, c_uint) — the two pools."""
    from pyffiam.cpp_interface import FFIAMLibrary
    lib = FFIAMLibrary(force_cpu=True)
    argtypes = lib._ffiam.PyAnalysis.argtypes
    assert argtypes[0] is cts.c_void_p
    assert argtypes[1] is cts.c_uint
    assert argtypes[2] is cts.c_void_p
    assert argtypes[3] is cts.c_uint


# =============================================================================
# RawResults dataclass
# =============================================================================

def test_raw_results_dataclass_fields():
    """RawResults must expose the seven documented arrays/scalars."""
    from pyffiam.cpp_interface import RawResults
    import numpy as np
    r = RawResults(
        helio_locs=np.zeros((2, 3)),
        helio_aim_vs=np.zeros((2, 3)),
        helio_angles=np.zeros((2,)),
        irrads=np.zeros((10,)),
        facet_data=np.zeros((2, 50, 3)),
        misc_data=np.zeros((10,)),
        movement=1.0,
    )
    assert r.helio_locs.shape == (2, 3)
    assert r.irrads.shape == (10,)
    assert r.movement == 1.0
