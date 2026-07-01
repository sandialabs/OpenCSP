# FFIAM-UE

Unreal Engine 5.7 visualization for the Full-Field Irradiance Analysis Model (FFIAM). Renders CSP fields and irradiance results in an interactive 3D environment, with real-time aim-strategy adjustment and live threshold updates.

![FFIAM-UE](Irradiance.png)

This repository is the visualization workflow only. The C++/CUDA analysis library and Python wrapper live in the main FFIAM repo: [FFIAM](https://gitlab-ex.sandia.gov/casims/ffiam) — see its README for the full project overview, installation, and usage.

## Requirements

- Unreal Engine 5.7
- A built copy of the FFIAM C++ library (`ffiam_lib.dll` on Windows, `libffiam_lib.so` on Linux). Set `FFIAM_PATH` to the directory containing the library so the project can resolve it at runtime.
- Memory: each analysis allocates a 2 GB field pool and a 2 GB voxel pool (~4 GB RAM), plus GPU memory for real-time rendering of the field and voxels.

## Quick Start

1. Build the FFIAM C++ library (see the main FFIAM README).
2. Open `Irradiance.uproject` in Unreal Engine 5.7.
3. The project links against the FFIAM library via the `FFIAM_PATH` environment variable.

## License

See [LICENSE.md](LICENSE.md).
