#pragma once

#include <string>

#include "fmt/core.h"

#include "ffiam/types.h"
#include "memory/pool.h"
#include "util/cuda_utils.h"
#include "vector_types.h"
#include "util/helper_math.h"


// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

__global__
void Cuda_ComputeHeliostatIrradiance(float *irrads,
                                     heliostat *helios,
                                     field_layout field,
                                     heliostat_design helioDesign,
                                     float3 *facetFlatOrigins,
                                     float sdni,
                                     float f1,
                                     float f2,
                                     float beta,
                                     float minAttenuation,
                                     float irradExponent,
                                     int useOuterRays,
                                     int useCrossPattern,
                                     float fluxCorrectionScale,
                                     float preFocalScale);


// ============================================================================
// Internal Function Declarations
// ============================================================================

// Norm and DistToPlane are in util/math.h (inline).


void CPU_ComputeIrradFromFacetRay(float* irrads,
                                  field_layout* field,
                                  float3 focalPt,
                                  float3 rayOrigin,
                                  float focalLen,
                                  float sdni,
                                  float f1,
                                  float f2,
                                  float minAttenuation,
                                  float irradExponent,
                                  float fluxCorrectionScale,
                                  float preFocalScale);

void CPU_ComputeHeliostatIrradiance(float *irrads,
                                    field_layout field,
                                    const heliostat_design &heliostat,
                                    const float3 *facetFlatOrigins,
                                    float sdni,
                                    float f1,
                                    float f2,
                                    float beta,
                                    float minAttenuation,
                                    float irradExponent,
                                    int useOuterRays,
                                    int useCrossPattern,
                                    float fluxCorrectionScale,
                                    float preFocalScale);


// Compute per-voxel irradiance for all heliostats in the field.
// @param field       Field geometry and voxel grid parameters.
// @param helio       Heliostat and facet optical design.
// @param aimStrat    Aim strategy (point, individual, file-driven, etc.).
// @param dt          Date/time and site location for solar position.
// @param irrads      Output: flat voxel irradiance array [kW/m²], caller-allocated.
// @param miscData    Output: miscellaneous scalar results (sun angles, timing, etc.).
// @param float3Data  Output: float3 vector results (sun direction, etc.).
// @param refl        Mirror reflectivity [0, 1].
// @param dni         Direct normal irradiance [W/cm²].
// @param beta        Beam spread [rad] (subtended sun angle + facet slope error).
// @param ambient     Constant irradiance added to every voxel [kW/m²]; 0 = disabled.
// @param minAttenuation  Floor on pre-focal attenuation term; 1.0 = no floor (original).
// @param irradExponent   Distance-attenuation power law exponent (2.0 = inverse square).
// @param nRaysPerFacet   12 = 9 core + 3 outer rays; 9 = 9 core only.
// @param fluxCorrectionScale  Scales the piecewise flux correction factor [0, 1]; 0 = disabled.
// @param preFocalScale   Scales pre-focal attenuation; 1.0 = symmetric behavior.
cudaError_t FieldAnalysis(field_layout* field,
                          heliostat_design* helio,
                          aim_strategy* aimStrat,
                          date_info* dt,

                          float* irrads,
                          float* miscData,
                          float3* float3Data,

                          bool verbose = false,
                          bool coutToFile = false,

                          // environment params
                          float refl = 0.9f,
                          float dni = 0.1f,
                          float beta = 0.0094f,
                          float ambient = 0.0f,
                          float minAttenuation = 1.0f,
                          float irradExponent = 2.0f,
                          float nRaysPerFacet = 12.0f,
                          float fluxCorrectionScale = 1.0f,
                          float preFocalScale = 1.0f,

                          bool useCpu = false
                          );


// Verify DLL access.
cudaError_t TestPrint(float TestNum=1);


// ============================================================================
// Public API - C++ Interface
// ============================================================================

// Parse parameters, initialize structs and memory, then run FieldAnalysis.
// @param arena        Pre-allocated main memory pool (caller owns, typically 2 GB).
// @param nMemory      Size of arena [bytes].
// @param voxelArena   Pre-allocated voxel memory pool.
// @param nVoxelMemory Size of voxelArena [bytes].
// @param year/month/day  Calendar date (UTC).
// @param hour         Fractional hours [0, 24), e.g. 10.5 = 10:30 AM.
// @param coords       Site latitude/longitude [decimal degrees].
// @param timezone     UTC offset [hours], e.g. -7 for MST.
// @param radius       Heliostat field radius [m].
// @param zMin/zMax    Airspace altitude range [m] above ground.
// @param towerH       Receiver tower height [m].
// @param helioFile    Path to heliostat layout CSV.
// @param nHelios      Number of heliostats to load from CSV.
// @param facetFile    Path to facet offset CSV.
// @param nFacets      Total facets per heliostat.
// @param nFacetCols   Facet columns per heliostat (used when not imported).
// @param facetDims    Facet width and height [m].
// @param aimStratId   Aim strategy enum value (see AimType).
// @param aimParams    Strategy-specific aim parameters (e.g. target xyz [m]).
// @param aimFile      Path to per-heliostat aim CSV (used when aimStratId requires it).
// @param refl         Mirror reflectivity [0, 1].
// @param dni          Direct normal irradiance [W/cm²].
// @param beta         Beam spread [rad].
// @param voxelSize    Voxel side length [m]; valid range 1–4.
// @param ambient      Constant irradiance floor [kW/m²]; 0 = disabled.
// @param minAttenuation  Pre-focal attenuation floor; 1.0 = no floor.
// @param irradExponent   Distance-attenuation exponent (2.0 = inverse square).
// @param nRaysPerFacet   12 = 9 core + 3 outer; 9 = core only.
// @param fluxCorrectionScale  Piecewise flux correction blend [0, 1]; 0 = disabled.
// @param preFocalScale   Pre-focal attenuation scale; 1.0 = symmetric.
extern "C" int InitAnalysis(void* arena,
                            uint nMemory,
                            void* voxelArena,
                            uint nVoxelMemory,

                            int year,
                            int month,
                            int day,
                            float hour,
                            float2 coords,
                            float timezone,

                            int radius = 600,
                            int zMin = 4,
                            int zMax = 100,
                            float towerH = 61,

                            std::string helioFile = "",
                            int nHelios = 2000,

                            std::string facetFile = "",
                            int nFacets = 25,
                            int nFacetCols = 5,
                            float2 facetDims = { 1, 1 },

                            int aimStratId = 1,
                            float3 aimParams = {0, 0, 100},
                            std::string aimFile = "",

                            bool verbose = false,

                            float refl = 0.9f,
                            float dni = 0.1f,
                            float beta = 0.0094f,
                            int voxelSize = 2,
                            float ambient = 0.0f,
                            float minAttenuation = 1.0f,
                            float irradExponent = 2.0f,
                            float nRaysPerFacet = 12.0f,
                            float fluxCorrectionScale = 1.0f,
                            float preFocalScale = 1.0f,
                            bool useCpu = false,
                            int layout = layout_grid);  // field_layout_type; used when no helioFile


// Run analysis using a predefined site configuration (NSTTF, radial, sample, etc.).
// Loads heliostat layout, facet design, and field geometry from the built-in site preset.
// @param arena/nMemory            Main memory pool (caller owns, typically 2 GB).
// @param voxelArena/nVoxelMemory  Voxel memory pool.
// @param datetime   Packed date/time: (year, month, day, hour*100) as int4.
// @param siteId     Preset site enum value (see CspSite).
// @param aimId      Aim strategy enum value (see AimType).
// @param aimParams  Strategy-specific aim parameters (e.g. target xyz [m]).
// @param version    Config variant within the preset; controls airspace bounds and voxel size.
// @param refl       Mirror reflectivity [0, 1].
// @param dni        Direct normal irradiance [W/cm²].
// @param beta       Beam spread [rad].
// @param minAttenuation  Pre-focal attenuation floor; 1.0 = no floor.
// @param irradExponent   Distance-attenuation exponent (2.0 = inverse square).
// @param nRaysPerFacet   12 = 9 core + 3 outer; 9 = core only.
// @param fluxCorrectionScale  Piecewise flux correction blend [0, 1]; 0 = disabled.
// @param preFocalScale   Pre-focal attenuation scale; 1.0 = symmetric.
extern "C" int InitPresetAnalysis(void* arena,
                                  uint nMemory,
                                  void* voxelArena,
                                  uint nVoxelMemory,
                                  int4 datetime,

                                  int siteId,
                                  int aimId,
                                  float3 aimParams,

                                  bool verbose = true,
                                  int version = 1,
                                  float refl = 0.9f,
                                  float dni = 0.1f,
                                  float beta = 0.0094f,
                                  float minAttenuation = 1.0f,
                                  float irradExponent = 2.0f,
                                  float nRaysPerFacet = 12.0f,
                                  float fluxCorrectionScale = 1.0f,
                                  float preFocalScale = 1.0f,
                                  bool useCpu = false);


// ============================================================================
// Public API - Python/ctypes Interface
// ============================================================================

// ctypes entry point. Mirrors InitAnalysis with char* strings instead of std::string.
// File paths are relative; the implementation prepends the "data/" directory prefix.
// All pointer arguments (arena, voxelArena, helioFile, facetFile, aimFile) must be valid.
// @param arena/nMemory            Main memory pool (caller allocates via ctypes, typically 2 GB).
// @param voxelArena/nVoxelMemory  Voxel memory pool.
// @param year/month/day  Calendar date (UTC).
// @param hour            Fractional hours [0, 24), e.g. 10.5 = 10:30 AM.
// @param lat/lng         Site latitude and longitude [decimal degrees].
// @param timezone        UTC offset [hours], e.g. -7 for MST.
// @param radius          Heliostat field radius [m].
// @param zMin/zMax       Airspace altitude range [m] above ground.
// @param towerH          Receiver tower height [m].
// @param helioFile       Null-terminated path to heliostat layout CSV.
// @param nHelios         Number of heliostats to load.
// @param facetFile       Null-terminated path to facet offset CSV.
// @param nFacets         Total facets per heliostat.
// @param nFacetCols      Facet columns per heliostat.
// @param facetW/facetH   Facet width and height [m].
// @param aimStratId      Aim strategy enum value (see AimType).
// @param aim1/aim2/aim3  Strategy-specific aim parameters (e.g. target x/y/z [m]).
// @param aimFile         Null-terminated path to per-heliostat aim CSV.
// @param refl            Mirror reflectivity [0, 1].
// @param dni             Direct normal irradiance [W/cm²].
// @param beta            Beam spread [rad].
// @param voxelSize       Voxel side length [m]; valid range 1–4.
// @param ambient         Constant irradiance floor [kW/m²]; 0 = disabled.
// @param minAttenuation  Pre-focal attenuation floor; 1.0 = no floor.
// @param irradExponent   Distance-attenuation exponent (2.0 = inverse square).
// @param nRaysPerFacet   12 = 9 core + 3 outer; 9 = core only.
// @param fluxCorrectionScale  Piecewise flux correction blend [0, 1]; 0 = disabled.
// @param preFocalScale   Pre-focal attenuation scale; 1.0 = symmetric.
extern "C" int PyAnalysis(void* arena,
                          uint nMemory,
                          void* voxelArena,
                          uint nVoxelMemory,

                          int year,
                          int month,
                          int day,
                          float hour,

                          float lat, float lng, float timezone,

                          int radius,
                          int zMin,
                          int zMax,
                          float towerH,

                          char* helioFile,
                          int nHelios,

                          char* facetFile,
                          int nFacets,
                          int nFacetCols,
                          float facetW,
                          float facetH,

                          int aimStratId,
                          float aim1,
                          float aim2,
                          float aim3,
                          char* aimFile,

                          float refl,
                          float dni,
                          float beta,
                          int voxelSize,
                          float ambient,
                          float minAttenuation,
                          float irradExponent,
                          float nRaysPerFacet,
                          float fluxCorrectionScale,
                          float preFocalScale,
                          bool verbose,
                          bool useCpu,
                          int layout = layout_grid);  // field_layout_type; used when no helioFile


// ============================================================================
// Utility Functions
// ============================================================================

// Map a voxel grid position to its 1D array index.
// Coordinate system: (x, y, z) = (east, north, up) [m]; tower at origin.
// @param loc     Integer voxel position [m] in world space.
// @param size    Voxel side length [m].
// @param radius  Half-width of the field grid [m]; grid spans [-radius, +radius] in x and y.
// @param zmin    Lower altitude bound of the grid [m].
inline int GetVoxelIndexFromLoc(const int3 loc, const int size = 4, const int radius = 600, const int zmin = 0)
{
    const float fieldArea = powf(2.0f * static_cast<float>(radius), 2);
    const float voxelArea = powf(static_cast<float>(size), 2);
    const int nPerPlane = static_cast<int>(fieldArea / voxelArea);
    const int nPerSide = static_cast<int>(sqrt(nPerPlane));

    const int z = (loc.z - zmin) / size;
    const int y = (loc.y + radius) / size;
    const int x = (loc.x + radius) / size;
    const int idx = z * nPerPlane + y * nPerSide + x;
    return idx;
}


// Float-coordinate overload.
inline int GetVoxelIndexFromLoc(const float3 loc, const int size = 4, const int radius = 600, const int zmin = 0)
{
    const int3 loc_int = { static_cast<int>(loc.x), static_cast<int>(loc.y), static_cast<int>(loc.z) };
    const int result = GetVoxelIndexFromLoc(loc_int, size, radius, zmin);
    return result;
}


// ============================================================================
// Test/Verification Functions
// ============================================================================

// Verify ctypes integer passing.
extern "C" int Py_TestAdd(int a, int b);


// Verify DLL loading from Python.
extern "C" void Py_TestPrint();
