# pyffiam Developer Guide

Python wrapper for the FFIAM C++/CUDA library. Performs volumetric irradiance analysis of CSP airspace by partitioning volume into voxels and computing per-voxel flux.

**Runtime:** Python 3.10+, CUDA 12 runtime. Works on Windows (native) and Linux (Docker). The C++ library is loaded via `ctypes` from `src/pyffiam/external/`.

## Architecture

### Analysis Pipeline

1. `analysis.py`: main entry point. `analysis(...)` assembles config, invokes the C++ library, post-processes results, writes outputs.
2. `cpp_interface.py`: ctypes layer. Handles DLL/.so resolution (`ffiam_lib.dll` on Windows, `libffiam_lib.so` on Linux), argtype setup, memory allocation/free, and the `PyAnalysis` call.
3. `config.py`: input dataclasses (site, field, aim strategy, datetime, runtime model params).
4. `results.py`: output dataclasses (`AnalysisResults`, `PathResults`, voxel arrays, statistics).
5. `analysis_data.py`: legacy `AnalysisData` / `PathData` containers still used by some call sites.

### Types and Conventions

- `ffiam_types.py` exposes `CspSite` (Custom, NSTTF, CrescentDunesSmall, CrescentDunes, SampleV1/V2/V3), `AimType` (Point, Ring, SplitRing, Vector, CsvData, FixedNormal, Null), and `Direction`.
- Coordinate system: Cartesian `(x, y, z)` = `(east, north, up)`, tower at origin, meters.
- `FixedNormal` / STOW aims the heliostat surface normal at a fixed vector; beam direction follows the sun. All other aim types tilt heliostats to redirect sunlight toward a target.

### Supporting Modules

- `plots.py`: Plotly voxel-scatter plots, heatmaps, path plots, GIF animations.
- `xls.py`: `ExcelWriter` class for the .xlsx summary.
- `output.py`: output directory / slug management.
- `app_config.py`: cross-platform app data path resolution.
- `utils.py`: voxel indexing (`get_voxel_loc_from_index`, `get_voxel_indexes_from_locs`), interpolation, unit conversions.
- `run_site_analysis_from_file.py`: batch runner over `site_configs/loaded/*.json`.

### Site Configurations

JSON files in `src/pyffiam/site_configs/`:
- `presets/`: NSTTF, CrescentDunes, CrescentDunesSmall, SampleV1/V2/V3
- `loaded/`: active batch configs
- `unloaded/`: inactive configs

See the README "JSON Configuration Format" section for the full schema.

### Data Files

`src/pyffiam/data/` holds heliostat and facet CSVs for preset sites (X, Y, Z in cols B–D, row 2 onward).

## Usage

### Python API

```python
from pyffiam.analysis import analysis
from pyffiam.ffiam_types import CspSite, AimType
import numpy as np

result = analysis(
    site=CspSite.NSTTF,
    year=2025, month=6, day=21, hour=12,
    threshold=4,
    aim_strat=AimType.Point,
    aim_params=np.array([0, 0, 90]),
    create_plots=True,
    create_xls=True,
)
```

### Batch from JSON

```bash
cd pyffiam/src
python -m pyffiam.run_site_analysis_from_file
```

Processes every `.json` in `site_configs/loaded/`.

### Docker

```bash
docker build -t ffiam .
docker run --rm --gpus all -v $(pwd)/output:/output ffiam -c "..."
```

See README for full Docker examples.

## Testing

```bash
cd pyffiam
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=pyffiam   # with coverage
```

Single test:
```bash
python -m pytest tests/test_integration.py -v
```

## Implementation Notes

### Memory

Two 2 GB memory pools are allocated via `libc.malloc()` from `cpp_interface.py`:
- **Main pool**: heliostat data, aim vectors, angles
- **Voxel pool**: per-voxel irradiance arrays

The C++ library uses a chunked arena pattern; pools are explicitly freed after the analysis call.

### Adding a C++ Entry Point

1. Add the function to `ffiam/src/main/ffiam.cu` and export via `FFIAM_API` macro.
2. Declare it in `ffiam/include/ffiam/ffiam.h`.
3. In `cpp_interface.py`: set `argtypes`, `restype`, encode strings with `.encode()`, cast pointers with `ctypes.cast()`, and convert arrays with `np.ctypeslib.as_array()`.

### Runtime Model Parameters

Five parameters flow from Python -> ctypes -> C++ -> CUDA kernel:
- `min_attenuation` (hybrid blend floor)
- `irrad_exponent` (beam concentration)
- `n_rays_per_facet` (ray count + f2 normalization)
- `flux_correction_scale` (piecewise flux corrections)
- `pre_focal_scale` (pre-focal beam scale)

The kernel also derives `useOuterRays` (n_rays > 9) and `useCrossPattern` (n_rays ≤ 5).

### Backend selection (CUDA vs CPU)

`FFIAMLibrary` auto-detects whether the CUDA runtime is available. On a CUDA
machine it loads `ffiam_lib.dll` / `libffiam_lib.so` and runs the GPU
kernel. Without a CUDA runtime it loads `ffiam_lib_cpu.dll` /
`libffiam_lib_cpu.so` and prints a banner to stderr indicating CPU mode.

The CPU backend mirrors the CUDA model exactly (same attenuation curve, ray
patterns, and flux correction) and is OpenMP-parallel over heliostats when
the build picked up OpenMP. It's slower but produces results within ~1% of
CUDA on the standard validation suites.

To force CPU on a machine that has CUDA available:

- `analysis(force_cpu=True, ...)`, or
- set `FFIAM_FORCE_CPU=1` in the environment before running.

The Python-side flag and env var are mirrored on the C++ side via a new
`bool useCpu` parameter on `PyAnalysis` / `InitAnalysis` /
`InitPresetAnalysis` / `FieldAnalysis`. The library load mechanism and the
runtime parameter are independent: a CPU library can still receive
`useCpu=false` (it'll just ignore it and use the CPU path), and a CUDA
library can receive `useCpu=true` to exercise the CPU branch in-process.

### Logging

All modules log to `ffiam.log` in the app data directory. Handlers are configured in `analysis.py`.

### Output Layout

Per-analysis directory under `<app_data>/SimsIndustries/FFIAM/<slug>_<timestamp>/` containing Excel summary, PNG plots, GIFs, and log. Override with the `output_dir` analysis param.

## Constraints

- CUDA 12 runtime must be resolvable (`cudart64_12.dll` on Windows, `libcudart.so.12` on Linux).
- NVIDIA GPU required for performance; CPU fallback exists in the C++ library.
- `kaleido==0.1.0.post1` pinned on Windows (plotly `write_image` bug); Docker overrides to `0.2.1` on Linux.
