# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Tests for the CPU backend: force_cpu kwarg, env var, fallback banner, parity.

The CUDA-vs-CPU parity test is marked requires_cuda and skipped on machines
without a CUDA runtime.
"""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# Library-load behavior
# =============================================================================

def test_force_cpu_kwarg_loads_cpu_library():
    """force_cpu=True must select the CPU backend regardless of CUDA presence."""
    from pyffiam.cpp_interface import FFIAMLibrary
    lib = FFIAMLibrary(force_cpu=True)
    assert lib.backend == "cpu"


def test_env_var_forces_cpu(monkeypatch):
    """FFIAM_FORCE_CPU=1 must also select the CPU backend."""
    from pyffiam.cpp_interface import FFIAMLibrary
    monkeypatch.setenv("FFIAM_FORCE_CPU", "1")
    lib = FFIAMLibrary()
    assert lib.backend == "cpu"


def test_fallback_banner_printed_when_cuda_missing(capsys, monkeypatch):
    """When CUDA is reported missing, the loader must fall back and print the banner."""
    from pyffiam.cpp_interface import FFIAMLibrary
    monkeypatch.setenv("FFIAM_TEST_PRETEND_NO_CUDA", "1")
    lib = FFIAMLibrary()
    captured = capsys.readouterr()
    assert lib.backend == "cpu"
    assert "No CUDA device detected" in captured.err


@pytest.mark.requires_cuda
def test_default_uses_cuda_when_available():
    """With CUDA installed and no force_cpu flag, the default backend is CUDA."""
    from pyffiam.cpp_interface import FFIAMLibrary
    lib = FFIAMLibrary()
    assert lib.backend == "cuda"


# =============================================================================
# End-to-end integration
# =============================================================================

def _run_small_analysis(force_cpu: bool):
    """Run a tiny preset analysis and return AnalysisResults-like object with .irrads."""
    from pyffiam.analysis import analysis
    from pyffiam.ffiam_types import CspSite, AimType
    return analysis(
        site=CspSite.NSTTF,
        year=2025, month=6, day=21, hour=12.0,
        threshold=4,
        aim_strat=AimType.Point,
        aim_params=np.array([0, 0, 90]),
        create_xls=False, create_gifs=False, create_plots=False,
        open_output_dir=False,
        force_cpu=force_cpu,
    )


def test_analysis_force_cpu_runs():
    """analysis() with force_cpu=True must complete and produce positive irradiance."""
    result = _run_small_analysis(force_cpu=True)
    assert result is not None
    irrads = getattr(result, "irrads", None)
    if irrads is None and hasattr(result, "irradiance"):
        irrads = result.irradiance.values
    arr = np.asarray(irrads)
    assert arr.size > 0
    assert float(arr.sum()) > 0.0


@pytest.mark.requires_cuda
def test_cpu_matches_cuda_within_tolerance(capsys):
    """CPU and CUDA backends must agree on the physics that matters: total flux
    drift within 0.5%, peak irradiance within 1%, and per-voxel relative error
    within 1% for voxels that carry meaningful magnitude. Sub-noise voxels
    (below 0.1% of peak) are excluded — their absolute differences are tiny
    but blow up under relative-error arithmetic, and they sit far below the
    glare-relevant range anyway."""
    cuda_res = _run_small_analysis(force_cpu=False)
    cpu_res  = _run_small_analysis(force_cpu=True)

    def _get_irrads(r):
        irrads = getattr(r, "irrads", None)
        if irrads is None and hasattr(r, "irradiance"):
            irrads = r.irradiance.values
        return np.asarray(irrads).flatten()

    cuda_arr = _get_irrads(cuda_res)
    cpu_arr  = _get_irrads(cpu_res)
    assert cuda_arr.shape == cpu_arr.shape

    cuda_peak = float(cuda_arr.max())
    cpu_peak  = float(cpu_arr.max())
    peak_drift = abs(cpu_peak - cuda_peak) / cuda_peak

    # Voxels above 0.1% of peak are the ones that carry the irradiance signal.
    # Below that the values are FP noise where relative error is meaningless.
    mask_threshold = max(1.0, 0.001 * cuda_peak)
    mask = cuda_arr > mask_threshold
    rel = np.abs(cpu_arr[mask] - cuda_arr[mask]) / cuda_arr[mask]
    max_rel = float(rel.max())

    total_cuda = float(cuda_arr.sum())
    total_cpu  = float(cpu_arr.sum())
    total_drift = abs(total_cpu - total_cuda) / total_cuda

    # Print diagnostics so a future failure is easy to investigate.
    if max_rel >= 0.01 or total_drift >= 0.005 or peak_drift >= 0.01:
        worst_local = int(rel.argmax())
        worst_global = int(np.flatnonzero(mask)[worst_local])
        print(
            f"\nCPU vs CUDA parity diagnostics:\n"
            f"  total_cuda={total_cuda:.2f}, total_cpu={total_cpu:.2f}, drift={total_drift:.6f}\n"
            f"  peak_cuda={cuda_peak:.2f}, peak_cpu={cpu_peak:.2f}, drift={peak_drift:.6f}\n"
            f"  masked voxels: {mask.sum()} / {cuda_arr.size} (threshold {mask_threshold:.3f})\n"
            f"  worst voxel #{worst_global}: cuda={cuda_arr[worst_global]:.4f}, "
            f"cpu={cpu_arr[worst_global]:.4f}, rel={max_rel:.4f}"
        )

    assert total_drift < 0.005, f"total flux drift {total_drift:.6f} exceeds 0.5%"
    assert peak_drift < 0.01,   f"peak drift {peak_drift:.6f} exceeds 1%"
    assert max_rel < 0.01,      f"max rel error {max_rel:.4f} exceeds 1% on voxels > {mask_threshold:.3f}"
