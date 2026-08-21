# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""ctypes interface to the FFIAM C++ library: DLL loading, memory pools, and result extraction."""

from __future__ import annotations

import ctypes as cts
from ctypes import c_char_p, c_void_p, c_int, c_uint, c_float, c_bool
from ctypes.util import find_library
# LPCVOID is just c_void_p on Windows; use c_void_p directly for cross-platform
LPCVOID = cts.c_void_p
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Optional
import logging

import numpy as np

from pyffiam.app_config import (
    GB,
    MB,
    VOXEL_CHUNK_SIZE as _VOXEL_CHUNK_SIZE,
    VOXEL_POOL_SIZE as _VOXEL_POOL_SIZE,
)

log = logging.getLogger(__name__)


_FALLBACK_BANNER = (
    "\n"
    "============================================================\n"
    "FFIAM: No CUDA device detected. Using CPU backend.\n"
    "Expect 50-200x slower analysis vs. GPU.\n"
    "Pass force_cpu=False to surface the underlying load error.\n"
    "============================================================\n"
)


@dataclass
class AnalysisParams:
    """Flat parameter struct marshaled directly to C++ PyAnalysis()."""
    # Time parameters
    year: int
    month: int
    day: int
    hour: float  # Supports fractional hours (e.g., 10.5 = 10:30 AM)

    # Location parameters
    latitude: float
    longitude: float
    timezone: float

    # Field parameters
    field_radius: int
    min_height: int
    max_height: int
    tower_height: float

    # Heliostat parameters
    heliostat_file: str
    num_heliostats: int
    facet_file: str
    num_facets: int
    num_facet_cols: int
    facet_width: float
    facet_height: float

    # Aim strategy
    aim_strategy: int  # AimType.value
    aim_param_0: float
    aim_param_1: float
    aim_param_2: float
    aim_file: str

    # Analysis parameters
    reflectivity: float
    peak_dni: float
    beta: float
    voxel_size: int
    ambient: float  # Ambient irradiance baseline for all voxels [kW/m²]
    min_attenuation: float  # Hybrid blend floor (1.0 = disabled/original, 0.42 = UAS-calibrated)
    irrad_exponent: float   # Beam concentration exponent (2.0 = original, 1.7 = UAS-calibrated)
    n_rays_per_facet: float  # Rays per facet (12 = 9 core + 3 outer, 9 = core only)
    flux_correction_scale: float  # Flux correction scale (1.0 = full correction, 0.0 = disabled)
    pre_focal_scale: float  # Pre-focal attenuation scale (1.0 = original symmetric, 0.6 = UAS-calibrated)
    num_voxels: int
    layout: int = 0  # FieldLayout value (0=grid, 1=radial); used only when no heliostat_file


@dataclass
class RawResults:
    """Copied from C++ shared memory before pools are freed."""
    helio_locs: np.ndarray      # (num_heliostats, 3) heliostat positions
    helio_aim_vs: np.ndarray    # (num_heliostats, 3) heliostat aim vectors
    helio_angles: np.ndarray    # (num_heliostats,) heliostat movement angles
    irrads: np.ndarray          # (num_voxels,) irradiance per voxel
    facet_data: np.ndarray      # (num_heliostats, 2*num_facets, 3) facet origins and normals
    misc_data: np.ndarray       # (10,) miscellaneous data from C++
    movement: float             # Total heliostat movement (degrees)


class MemoryPool:
    """RAII wrapper around malloc/free for the C++ shared memory arenas.

    Use as a context manager; memory is freed on exit even if an exception occurs.
    """

    def __init__(self, libc: cts.CDLL, size_bytes: int, name: str = "pool"):
        """
        Args:
            libc: Loaded C runtime library exposing malloc/free.
            size_bytes: Number of bytes to allocate (typically 2 GB per pool).
            name: Label used in log messages and error strings.
        """
        self._libc = libc
        self._size_bytes = size_bytes
        self._name = name
        self._ptr: Optional[LPCVOID] = None

    @property
    def ptr(self) -> LPCVOID:
        """Return the allocated memory pointer, raising if not yet allocated."""
        if self._ptr is None:
            raise RuntimeError(f"MemoryPool '{self._name}' not allocated or already freed")
        return self._ptr

    @property
    def size(self) -> c_uint:
        """Return the pool size as a ctypes c_uint for passing to C.

        PyAnalysis() declares the arena size as `uint`, so a pool of 4 GiB or
        more would silently wrap to a small (or zero) value. Fail loudly instead.
        """
        if self._size_bytes >= 2 ** 32:
            raise ValueError(
                f"MemoryPool '{self._name}' is {self._size_bytes:,} bytes; PyAnalysis() "
                f"takes the arena size as a C uint, so pools must stay under 4 GiB."
            )
        return c_uint(self._size_bytes)

    def __enter__(self) -> 'MemoryPool':
        # malloc takes size_t; passing a c_uint leaves the upper half of the
        # argument register undefined, which corrupts allocations over 2 GiB.
        raw_ptr = self._libc.malloc(cts.c_size_t(self._size_bytes))
        if not raw_ptr:
            raise MemoryError(f"Failed to allocate {self._size_bytes:,} bytes for {self._name}")
        self._ptr = cts.cast(raw_ptr, LPCVOID)
        log.debug(f"Allocated {self._size_bytes // MB} MB for {self._name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ptr is not None:
            self._libc.free(self._ptr)
            log.debug(f"Freed {self._name}")
            self._ptr = None
        return False  # Don't suppress exceptions


class FFIAMLibrary:
    """Loads the FFIAM DLL and exposes run_analysis()."""

    # Memory configuration. Voxel sizes come from app_config so the pool, the
    # chunk and the MAX_VOXELS guard can never drift apart.
    MAIN_POOL_SIZE = 2 * GB
    MAIN_CHUNK_SIZE = 256 * MB
    VOXEL_POOL_SIZE = _VOXEL_POOL_SIZE
    VOXEL_CHUNK_SIZE = _VOXEL_CHUNK_SIZE

    # Chunk indices for data extraction (chunks are used last-first)
    MAX_CHUNKS = 8
    F3_DATA_IDX = MAX_CHUNKS - 5
    MISC_DATA_IDX = MAX_CHUNKS - 4
    H_ANGLES_IDX = MAX_CHUNKS - 3
    H_AIM_VS_IDX = MAX_CHUNKS - 2
    H_LOCS_IDX = MAX_CHUNKS - 1
    IRRADS_IDX = 0

    def __init__(self, dll_dir: Optional[Path] = None, force_cpu: bool = False):
        self._dll_dir = dll_dir or Path(__file__).parent / 'external'
        self._ffiam = None
        self._libc = None
        self.backend: str = "unknown"
        env_force = os.environ.get("FFIAM_FORCE_CPU", "") == "1"
        self._force_cpu = force_cpu or env_force
        self._load_libraries()
        self._setup_signatures()

    @staticmethod
    def _cuda_available() -> bool:
        """Probe whether the CUDA runtime can be located on this machine.

        Honors FFIAM_TEST_PRETEND_NO_CUDA=1 for fault-injection in tests.
        """
        if os.environ.get("FFIAM_TEST_PRETEND_NO_CUDA", "") == "1":
            return False
        if sys.platform == "win32":
            cuda_name, cuda_check = "cudart64_12", "cudart64"
        else:
            cuda_name, cuda_check = "cudart", "cudart"
        path = find_library(cuda_name)
        return path is not None and cuda_check in path

    def _load_libraries(self):
        """Locate and load the C runtime + FFIAM shared library. Falls back to
        the CPU build when CUDA is unavailable or force_cpu is set."""
        if sys.platform == 'win32':
            libc_name = 'msvcrt'
            gpu_name = 'ffiam_lib.dll'
            cpu_name = 'ffiam_lib_cpu.dll'
        else:
            libc_name = 'libc.so.6'
            gpu_name = 'libffiam_lib.so'
            cpu_name = 'libffiam_lib_cpu.so'

        self._libc = cts.CDLL(libc_name)
        self._libc.malloc.argtypes = [cts.c_size_t]
        self._libc.malloc.restype = c_void_p
        self._libc.free.argtypes = [c_void_p]
        self._libc.free.restype = None

        use_cpu = self._force_cpu or not self._cuda_available()

        if not use_cpu:
            try:
                if sys.platform == 'win32':
                    cuda_path = find_library('cudart64_12')
                else:
                    cuda_path = find_library('cudart')
                cts.cdll.LoadLibrary(cuda_path)
                gpu_path = self._dll_dir / gpu_name
                if not gpu_path.exists():
                    raise OSError(
                        f"FFIAM CUDA library not found at {gpu_path}. "
                        "Build ffiam_lib and ensure it is copied to external/."
                    )
                self._ffiam = cts.cdll.LoadLibrary(gpu_path.as_posix())
                self.backend = "cuda"
                log.info(f"Loaded FFIAM CUDA library from {gpu_path}")
                return
            except OSError as e:
                log.warning(f"Failed to load CUDA backend: {e}; falling back to CPU.")
                use_cpu = True

        cpu_path = self._dll_dir / cpu_name
        if not cpu_path.exists():
            raise OSError(
                f"FFIAM CPU library not found at {cpu_path}. "
                "Build the ffiam_lib_cpu target so it is copied to external/."
            )
        self._ffiam = cts.cdll.LoadLibrary(cpu_path.as_posix())
        self.backend = "cpu"
        print(_FALLBACK_BANNER, file=sys.stderr)
        log.warning(f"Loaded FFIAM CPU library from {cpu_path}")

    def _setup_signatures(self):
        """Declare ctypes argtypes and restype for PyAnalysis."""
        self._ffiam.PyAnalysis.argtypes = (
            # Main memory arena and size
            LPCVOID, c_uint,
            # Voxel memory arena and size
            LPCVOID, c_uint,
            # Time: year, month, day, hour (float for fractional hours)
            c_int, c_int, c_int, c_float,
            # Location: lat, lng, timezone
            c_float, c_float, c_float,
            # Field: radius, min_height, max_height, tower_height
            c_int, c_int, c_int, c_float,
            # Heliostat file
            c_char_p,
            # Num heliostats
            c_int,
            # Facet file
            c_char_p,
            # Facet properties: num_facets, num_cols, width, height
            c_int, c_int, c_float, c_float,
            # Aim strategy: type, param0, param1, param2
            c_int, c_float, c_float, c_float,
            # Aim file
            c_char_p,
            # Analysis params: reflectivity, peak_dni, beta
            c_float, c_float, c_float,
            # Voxel size, ambient, min_attenuation, irrad_exponent, n_rays_per_facet, flux_correction_scale, pre_focal_scale, verbose, useCpu
            c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_bool, c_bool,
            # layout (field_layout_type: 0=grid, 1=radial)
            c_int
        )
        self._ffiam.PyAnalysis.restype = c_int

    @contextmanager
    def _working_directory(self, path: Path):
        """Context manager to temporarily change working directory.

        The C++ library expects data files in 'data/' relative to cwd,
        so we change to the pyffiam source directory before calling it.
        """
        original_cwd = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(original_cwd)

    def run_analysis(self, params: AnalysisParams) -> RawResults:
        """Allocate memory pools, invoke C++ PyAnalysis(), and return copied results.

        Args:
            params: Fully populated AnalysisParams struct to pass to the C++ library.

        Returns:
            RawResults containing numpy arrays copied out of shared memory before deallocation.
        """
        # C++ expects data files in 'data/' relative to cwd
        pyffiam_src_dir = Path(__file__).parent.resolve()

        with self._working_directory(pyffiam_src_dir), \
             MemoryPool(self._libc, self.MAIN_POOL_SIZE, "main_pool") as main_pool, \
             MemoryPool(self._libc, self.VOXEL_POOL_SIZE, "voxel_pool") as voxel_pool:

            # Call C++ analysis
            status = self._call_analysis(params, main_pool, voxel_pool)

            if status != 0:
                raise RuntimeError(f"C++ analysis failed with status {status}")

            return self._extract_results(params, main_pool, voxel_pool)

    def _call_analysis(
        self,
        params: AnalysisParams,
        main_pool: MemoryPool,
        voxel_pool: MemoryPool
    ) -> int:
        """Forward all analysis parameters to C++ PyAnalysis() and return its status code."""
        return self._ffiam.PyAnalysis(
            main_pool.ptr,
            main_pool.size,
            voxel_pool.ptr,
            voxel_pool.size,

            params.year,
            params.month,
            params.day,
            params.hour,

            c_float(params.latitude),
            c_float(params.longitude),
            c_float(params.timezone),

            params.field_radius,
            params.min_height,
            params.max_height,
            c_float(params.tower_height),
            c_char_p(params.heliostat_file.encode()),
            params.num_heliostats,

            c_char_p(params.facet_file.encode()),
            params.num_facets,
            params.num_facet_cols,
            c_float(params.facet_width),
            c_float(params.facet_height),

            params.aim_strategy,
            c_float(params.aim_param_0),
            c_float(params.aim_param_1),
            c_float(params.aim_param_2),
            c_char_p(params.aim_file.encode()),

            c_float(params.reflectivity),
            c_float(params.peak_dni),
            c_float(params.beta),
            params.voxel_size,
            c_float(params.ambient),
            c_float(params.min_attenuation),
            c_float(params.irrad_exponent),
            c_float(params.n_rays_per_facet),
            c_float(params.flux_correction_scale),
            c_float(params.pre_focal_scale),
            True,                          # verbose
            self.backend == "cpu",          # useCpu
            params.layout,                  # layout (0=grid, 1=radial)
        )

    def _extract_results(
        self,
        params: AnalysisParams,
        main_pool: MemoryPool,
        voxel_pool: MemoryPool
    ) -> RawResults:
        """Copy results out of shared memory before pools are freed."""
        def get_array(pool: MemoryPool, chunk_size: int, chunk_idx: int,
                      array_x: int, array_y: int = 1) -> np.ndarray:
            ptr_v = pool.ptr.value + chunk_size * chunk_idx
            ptr_f = cts.cast(ptr_v, cts.POINTER(cts.c_float))
            shape = (int(array_x), int(array_y)) if array_y > 1 else (int(array_x),)
            return np.ctypeslib.as_array(ptr_f, shape=shape)

        c_h_locs = get_array(main_pool, self.MAIN_CHUNK_SIZE, self.H_LOCS_IDX,
                             params.num_heliostats, 3)
        c_h_aim_vs = get_array(main_pool, self.MAIN_CHUNK_SIZE, self.H_AIM_VS_IDX,
                               params.num_heliostats, 3)
        c_h_angles = get_array(main_pool, self.MAIN_CHUNK_SIZE, self.H_ANGLES_IDX,
                               params.num_heliostats, 1)

        n_misc_data = 10
        c_misc_data = get_array(main_pool, self.MAIN_CHUNK_SIZE, self.MISC_DATA_IDX,
                                n_misc_data, 1)

        n_f3_data = params.num_heliostats * 2 * params.num_facets
        c_f3_data = get_array(main_pool, self.MAIN_CHUNK_SIZE, self.F3_DATA_IDX,
                              n_f3_data, 3)

        c_irrads = get_array(voxel_pool, self.VOXEL_CHUNK_SIZE, self.IRRADS_IDX,
                             params.num_voxels, 1)

        log.info(f'Python memory: {self.MAIN_CHUNK_SIZE:,d} bytes/chunk')
        log.info(f'Helio locs shape: {c_h_locs.shape}')
        log.warning(f'Calculation results: total irradiance {int(np.nansum(c_irrads)):,d}. '
                    f'Impacted voxels {int(np.nansum(c_irrads > 0)):,d}')

        misc_data = np.copy(c_misc_data)

        return RawResults(
            helio_locs=np.copy(c_h_locs),
            helio_aim_vs=np.copy(c_h_aim_vs),
            helio_angles=np.copy(c_h_angles),
            irrads=np.copy(c_irrads),
            facet_data=np.copy(c_f3_data),
            misc_data=misc_data,
            movement=float(misc_data[0])
        )
