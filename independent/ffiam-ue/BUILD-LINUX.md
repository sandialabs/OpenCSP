# Building on Linux

The `Irradiance` module links the FFIAM C++/CUDA library, so you build that
first, then the Unreal Editor target. Tested with Unreal Engine 5.7 (source
build) and CUDA 12.x / 13.x.

## Prerequisites

- A source build of Unreal Engine (5.7+), with the Linux toolchain set up
  (`Engine/Build/BatchFiles/Linux/Build.sh` works).
- The NVIDIA driver plus a CUDA toolkit (12.x or 13.x) for building the FFIAM
  library. `nvcc`, `cmake`, and a C++ compiler must be on `PATH`.
- The FFIAM C++ sources (the `ffiam/ffiam/ffiam` tree, containing `include/`,
  `src/`, `external/`).

## 1. Build the FFIAM shared library

```bash
cd /path/to/ffiam/ffiam/ffiam
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ffiam_lib -j"$(nproc)"
# -> build/lib/libffiam_lib.so
```

## 2. Build the Unreal editor target

Point `FFIAM_PATH` at the FFIAM C++ source root, then invoke your engine's
`Build.sh`. `CUDA_PATH` defaults to `/usr/local/cuda` if unset.

```bash
export FFIAM_PATH=/path/to/ffiam/ffiam/ffiam
/path/to/UnrealEngine/Engine/Build/BatchFiles/Linux/Build.sh \
    IrradianceEditor Linux Development \
    -project="$PWD/Irradiance.uproject" -waitmutex
```

`Irradiance.Build.cs` locates `libffiam_lib.so` at `$FFIAM_PATH/build/lib/` by
default, copies it into `Binaries/Linux/`, and links the module against it. The
module only uses `cudaError_t` at compile time; the CUDA runtime lives inside
`libffiam_lib.so`, so the host's CUDA runtime version doesn't have to match the
one the UE module was compiled against.

### Overriding the library path

If your FFIAM `.so` isn't at the standard `$FFIAM_PATH/build/lib/` location
(e.g. you built it into a different directory for a specific toolchain), set
`FFIAM_LIB` to its full path:

```bash
export FFIAM_LIB=/path/to/libffiam_lib.so
```

## 3. Launch the editor

```bash
/path/to/UnrealEngine/Engine/Binaries/Linux/UnrealEditor "$PWD/Irradiance.uproject"
```

`EngineAssociation` in `Irradiance.uproject` is empty, so the project builds and
launches with whatever engine you invoke above (the source-build convention).
