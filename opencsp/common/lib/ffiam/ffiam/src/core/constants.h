// Copyright Sandia National Laboratories. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ffiam {
namespace constants {

// =============================================================================
// Unit Conversion Factors
// =============================================================================

/// W/cm² to kW/m² (multiply by 10)
constexpr float WATTS_CM2_TO_KW_M2 = 10.0f;

/// Degrees to radians
constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;

/// Radians to degrees
constexpr float RAD_TO_DEG = 180.0f / 3.14159265358979323846f;


// =============================================================================
// Environment Defaults
// =============================================================================

/// Default heliostat reflectivity [0-1]
constexpr float DEFAULT_REFLECTIVITY = 0.9f;

/// Default Direct Normal Irradiance [W/cm²] (0.1 = 1000 W/m²)
constexpr float DEFAULT_DNI = 0.1f;

/// Default beam spread factor (beta) [radians]; calibrated from NSTTF UAS validation
constexpr float DEFAULT_BETA = 0.018f;


// =============================================================================
// Field/Airspace Defaults
// =============================================================================

/// Default field radius [meters]
constexpr int DEFAULT_RADIUS = 600;

/// Default minimum altitude for airspace analysis [meters]
constexpr int DEFAULT_Z_MIN = 4;

/// Default maximum altitude for airspace analysis [meters]
constexpr int DEFAULT_Z_MAX = 100;

/// Default voxel side length [meters]
constexpr int DEFAULT_VOXEL_SIZE = 2;

/// Default tower/receiver height [meters]
constexpr float DEFAULT_TOWER_HEIGHT = 61.0f;


// =============================================================================
// Heliostat Defaults
// =============================================================================

/// Default number of facets per heliostat
constexpr int DEFAULT_N_FACETS = 25;

/// Default number of facet columns
constexpr int DEFAULT_N_FACET_COLS = 5;

/// Default facet width [meters]
constexpr float DEFAULT_FACET_WIDTH = 1.2f;

/// Default facet height [meters]
constexpr float DEFAULT_FACET_HEIGHT = 1.2f;


// =============================================================================
// Ray Tracing Parameters
// =============================================================================

/// Ray step size as fraction of voxel size; must be < 1/sqrt(3) for full diagonal coverage
constexpr float RAY_STEP_MULTIPLIER = 0.5f;

/// Recent voxel hit cache size (prevents double-counting within a ray walk)
constexpr int VOXEL_HIT_CACHE_SIZE = 8;

/// Plane normal Z component for top-of-field intersection
constexpr float PLANE_NORMAL_Z = -1.0f;

/// Divisor for ray origin offsets from facet center (2.0 = facet edge)
constexpr float OFFSET_DIVISOR = 2.0f;


// =============================================================================
// Flux Correction Coefficients
// =============================================================================
// Piecewise-linear correction vs. fR = dist / focalLen, calibrated to SolTrace.
//   fR < 0.6:  1.0  |  fR < 0.8:  -0.85*fR + 1.01  |  fR < 1.2:  1.925*fR - 1.21
//   fR < 1.3:  7.0*fR - 7.3  |  fR < 2.0:  4.57*fR - 4.14  |  fR >= 2.0:  5.0

namespace flux {

/// Minimum attenuation floor for the pre-focal hybrid model (runtime param; kept for reference).
/// Calibrated from F2/F7 UAS data: 0.42 balances full-field predictions.
constexpr float MIN_ATTENUATION = 0.42f;

/// fR below which hybrid blending begins (pre-focal attenuation floor kicks in)
constexpr float BLEND_START_FR = 0.60f;

/// Irradiance exponent (runtime param; kept for reference). 1.7 balances F2/F7 UAS fit.
constexpr float IRRAD_EXPONENT = 1.7f;

constexpr float THRESHOLD_1 = 0.6f;
constexpr float THRESHOLD_2 = 0.8f;
constexpr float THRESHOLD_3 = 1.2f;
constexpr float THRESHOLD_4 = 1.3f;
constexpr float THRESHOLD_5 = 2.0f;

constexpr float COEFF_A1 = -0.85f;
constexpr float COEFF_B1 = 1.01f;

constexpr float COEFF_A2 = 1.925f;
constexpr float COEFF_B2 = -1.21f;

constexpr float COEFF_A3 = 7.0f;
constexpr float COEFF_B3 = -7.3f;

constexpr float COEFF_A4 = 4.57f;
constexpr float COEFF_B4 = -4.14f;

constexpr float MAX_FACTOR = 5.0f;

}  // namespace flux


// =============================================================================
// Memory Configuration
// =============================================================================

/// Size constants
constexpr size_t KB = 1024ULL;
constexpr size_t MB = 1024ULL * 1024ULL;
constexpr size_t GB = 1024ULL * 1024ULL * 1024ULL;

constexpr size_t DEFAULT_POOL_SIZE = 2ULL * GB;
constexpr size_t DEFAULT_CHUNK_SIZE = 256ULL * MB;
constexpr size_t DEFAULT_VOXEL_POOL_SIZE = 2ULL * GB;
constexpr size_t DEFAULT_VOXEL_CHUNK_SIZE = 256ULL * 6ULL * MB;


// =============================================================================
// CUDA Configuration
// =============================================================================

constexpr uint32_t THREADS_PER_BLOCK = 1024;
constexpr size_t SHARED_MEM_SIZE = 1024 * sizeof(float);


// =============================================================================
// Timezone Bounds
// =============================================================================

constexpr float TIMEZONE_MIN = -14.0f;
constexpr float TIMEZONE_MAX = 12.0f;


}  // namespace constants
}  // namespace ffiam
