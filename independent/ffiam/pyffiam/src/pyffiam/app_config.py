# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

GB = 1024**3
MB = 1024**2

MAX_NUM_HELIOSTATS = 11600
MAX_HEIGHT = 310
MIN_HEIGHT = 0
MAX_FIELD_RADIUS = 1700

# Voxel size constraints
MIN_VOXEL_SIZE = 1
MAX_VOXEL_SIZE = 10

# Voxel pool sizing. Kept in parity with the UE viz tool's pool
# (Irradiance/IrradianceMode.h) so a site config that runs there also runs here.
#
# The C++ side makes the whole voxel arena a single chunk holding the irradiance
# array, so pool capacity IS the voxel budget: 3.5 GiB / 4 B = 939M voxels. That
# covers the 1 km site at 1 m voxels (788M).
#
# Hard ceiling: PyAnalysis() takes the arena size as a C `uint`, so the pool can
# never reach 4 GiB. Raising it past that needs the C API widened to size_t.
VOXEL_POOL_SIZE = 3584 * MB  # 3.5 GiB; must stay < 4 GiB (c_uint ABI limit)
VOXEL_CHUNK_SIZE = VOXEL_POOL_SIZE  # one chunk per arena
MAX_VOXELS = VOXEL_POOL_SIZE // 4  # float32 irradiance

# NB: this bounds the *analysis*, not post-processing. pyffiam's plotting/Excel
# stages peak around 20x the raw array (1.6 km @ 2 m = 381M voxels needed ~32 GB
# RSS), so large grids still need a high-RAM host or create_plots/create_xls off.

app_name = "FFIAM"
app_author = "SimsIndustries"

FFIAM_VERSION = "3.0"
