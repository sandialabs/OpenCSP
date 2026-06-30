#include "ffiam/ffiam.h"
#include "solar/solar_position.h"
#include "core/heliostat.h"
#include "core/voxel.h"
#include "api/preset_sites.h"
#include "util/logging.h"
#include "util/math.h"
#include "io/csv_reader.h"
#include "memory/pool.h"
#include "memory/memory_manager.h"

#include <locale>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstring>


#define MB(x) ((size_t) (x) << 20)
#define GB(x) ((size_t) (x) * 1024 * 1024 * 1024)


// Generates a centered nRows x nCols grid of facet offsets (heliostat-local
// plane) when no facet CSV is given. Pitch = facetWidth/facetHeight.
static void ComputeFacetGrid(heliostat_design* design, float3* facetOrigins)
{
    const int nCols = design->nCols > 0 ? design->nCols : 1;
    const int nRows = design->nFacets / nCols;
    const float w = design->facetWidth;
    const float h = design->facetHeight;
    const float x0 = -0.5f * static_cast<float>(nCols - 1) * w;
    const float y0 = -0.5f * static_cast<float>(nRows - 1) * h;
    for (int r = 0; r < nRows; ++r)
    {
        for (int c = 0; c < nCols; ++c)
        {
            facetOrigins[r * nCols + c] = {
                x0 + static_cast<float>(c) * w,
                y0 + static_cast<float>(r) * h,
                0.0f
            };
        }
    }
}


// Core analysis: compute solar position, configure heliostats, trace rays, fill irrads array.
cudaError_t FieldAnalysis(field_layout* field,
                          heliostat_design* helioDesign,
                          aim_strategy* aimStrat,
                          date_info* dt,

                          float* irrads,
                          float* miscData,
                          float3* float3Data,

                          bool verbose,
                          bool coutToFile,

                          // environment params
                          float refl,
                          float dni,
                          float beta,
                          float ambient,
                          float minAttenuation,
                          float irradExponent,
                          float nRaysPerFacet,
                          float fluxCorrectionScale,
                          float preFocalScale,
                          bool useCpu)
{
    Log(fmt::format("Verbose: {}, log: {}", verbose, coutToFile));
    Log(fmt::format("Backend: {}", useCpu ? "CPU" : "CUDA"));
    // Use "" for system default locale; "en_US" may not exist on all platforms
    try { std::locale::global(std::locale("")); }
    catch (...) { /* fall back to C locale if system locale unavailable */ }

    if (coutToFile)
    {
        freopen("ffiamlog.txt", "w", stdout);
    }

    Log("Beginning FieldAnalysis call");
    Log("CWD: " + std::filesystem::current_path().string());

    // Prep CUDA and memory
    cudaError_t cudaStatus = cudaSuccess;
#ifndef FFIAM_CPU_ONLY
    cudaStatus = cudaSetDevice(0);
#endif

    // Compute solar position using extracted module
    Log("Computing solar position...");
    SolarData solarData;
    bool sunOk = ComputeSolarPosition(field->lat, field->lng, field->timezone,
                                      dt->year, dt->month, dt->day, dt->hour,
                                      dni, &solarData);
    if (!sunOk)
    {
        return cudaStatus;
    }

    float3 sunVec = solarData.sunVector;
    float sdni = solarData.scaledDni;

    for (int i = 0; i < field->nHelios; ++i)
    {
        auto h = &field->helios[i];
        h->id = i;
        field->locs[i] = {0, 0, 0};
        field->aimVs[i] = {0, 0, 0};
    }

    // Prep heliostat and mirror facet data
    Log("Prepping heliostat and mirror data");
    const int nFacetsMax = 50;
    float3 facetOrigins[nFacetsMax];

    int nHeliosPerSide = static_cast<int>(sqrt(field->nHelios));

    Log(fmt::format("Data filenames provided:\n\t\t {} (heliostat positions)\n\t\t {} (facet origins)\n",
                    static_cast<const char*>(field->helioFile),
                    static_cast<const char*>(helioDesign->facetFile)));

    // Import or calculate heliostat Cartesian locations centered on tower
    if (field->helioFile[0] != '\0')
    {
        Log("Loading heliostats from file");
        field->nHelios = ImportHeliostatData(field->helioFile, field->locs, helioDesign);
        for (int i = 0; i < field->nHelios; ++i)
        {
            auto h = &field->helios[i];
            h->loc = field->locs[i];
        }
        Log("Loading complete");
    }
    else if (field->layout == layout_radial)
    {
        helioDesign->pivotHeight = 0;
        helioDesign->pivotOffset = 0;

        const float R = static_cast<float>(field->radius);
        const int targetN = field->nHelios;
        if (targetN <= 0 || R <= 0.0f)
        {
            Log("Radial field needs nHelios > 0 and radius > 0!");
            return cudaErrorInvalidValue;
        }

        // Concentric half-staggered rings filling the annulus [r0, R].
        // Ring pitch s ~ sqrt(area / targetN); azimuthal count = 2*pi*r/s.
        // Places exactly targetN heliostats (final ring truncated).
        const float r0 = 0.06f * R;            // tower keep-out
        const float usableArea = static_cast<float>(M_PI) * (R * R - r0 * r0);
        float s = std::sqrt(usableArea / static_cast<float>(targetN));
        if (s <= 0.0f) s = 1.0f;

        int count = 0;
        int nRings = 0;
        for (float r = r0 + 0.5f * s; count < targetN; r += s)
        {
            int nOnRing = static_cast<int>(std::lround(2.0 * M_PI * r / s));
            if (nOnRing < 1) nOnRing = 1;
            // half-slot stagger on alternate rings
            const float phase = (nRings % 2) ? static_cast<float>(M_PI) / nOnRing : 0.0f;
            for (int k = 0; k < nOnRing && count < targetN; ++k)
            {
                const float theta = phase + (2.0f * static_cast<float>(M_PI) * k) / nOnRing;
                const float3 pos = { r * std::cos(theta), r * std::sin(theta), 0.0f };
                field->locs[count] = pos;
                field->helios[count].loc = pos;
                ++count;
            }
            ++nRings;
        }
        field->nHelios = count;
        Log(fmt::format("Generated radial field: {} heliostats in {} rings "
                        "(r0={:.1f} m, R={:.1f} m, spacing={:.2f} m)",
                        count, nRings, r0, R, s));
    }
    else  // layout_grid
    {
        helioDesign->pivotHeight = 0;
        helioDesign->pivotOffset = 0;

        const float fieldR = static_cast<float>(field->radius);
        const size_t n = static_cast<size_t>(nHeliosPerSide);
        if (n == 0)
        {
            Log("Num heliostats cannot be zero!");
            return cudaErrorInvalidValue;
        }

        // place heliostats at cell centers, covering [-R, R) with equal spacing
        const float step = (2.0f * fieldR) / static_cast<float>(n);
        const float start = -fieldR + 0.5f * step;

        for (size_t y = 0; y < n; ++y)
        {
            for (size_t x = 0; x < n; ++x)
            {
                const float3 pos = {
                    start + static_cast<float>(x) * step,
                    start + static_cast<float>(y) * step,
                    0.0f
                };

                const size_t idx = y * n + x;

                field->locs[idx] = pos;

                auto* h = &field->helios[idx];
                h->loc = pos;
            }
        }
        field->nHelios = static_cast<int>(n * n);  // actual placed count
    }

    // If user defined aim strategy via CSV file, import per-heliostat azimuth and elevation angles
    if (aimStrat->type == aim_data_csv)
    {
        if (aimStrat->file[0] == '\0')
        {
            Log("Aim strategy type set as aim_data_csv but .CSV file not provided.");
            return cudaStatus;
        }

        int nRowsImported = ImportAimData(aimStrat->file, field->aimVs);
        if (nRowsImported != field->nHelios)
        {
            Log(fmt::format("Aim data file {} has {} rows, but {} heliostats were provided.",
                            static_cast<const char*>(aimStrat->file), nRowsImported, field->nHelios));
            return cudaStatus;
        }
    }

    // either import facet origin coordinates from file or calculate them based on given geometry
    if (helioDesign->facetFile[0] != '\0')
    {
        Log("Importing facet origin coordinates...");
        helioDesign->nFacets = ImportFacetCoordinates(helioDesign->facetFile, facetOrigins);
        if (!helioDesign->nFacets) return cudaStatus;
        helioDesign->facetsImported = true;
        Log("Facet import complete.");
    }
    else
    {
        Log("Facet data file not provided. Computing facet grid from geometry.");
        ComputeFacetGrid(helioDesign, facetOrigins);
        helioDesign->facetsImported = true;
    }

    helioDesign->nRows = helioDesign->nFacets / helioDesign->nCols;

    if (verbose)
    {
        Log("==== ENVIRONMENT ====");
        PrintParam("Refl", refl, 4, "");
        PrintParam("DNI", dni, 4, "W/cm^2");
        PrintParam("Scaled DNI", sdni, 4, "W/cm^2");
        PrintParam("Beta", beta, 4, "rad\n");
    }

    float facetArea = helioDesign->facetHeight * helioDesign->facetWidth;
    float facetDiam = std::pow(4 * facetArea / M_PI, 0.5f);

    float f1 = refl * sdni / (helioDesign->nFacets);
    float scaler = 1.f;
    f1 *= scaler;
    float f2 = beta / facetDiam / nRaysPerFacet;

    // Compute voxel grid dimensions using extracted module
    int nVoxelsX, nVoxelsY, nVoxelsZ;
    int nVoxels = ComputeVoxelGridDimensions(field->radius, field->zMin, field->zMax,
                                             field->voxelSize, &nVoxelsX, &nVoxelsY, &nVoxelsZ);
    field->nVoxels = nVoxels;

    if (verbose)
    {
        Log(fmt::format("\n====== FIELD SETUP ====="));
        Log(fmt::format("radius {:L} m, height {:L} m, voxel size {:L} m \n",
                        field->radius, field->zMax, field->voxelSize));
        Log(fmt::format("{:L} heliostats ({:L} / side)\n", field->nHelios, nHeliosPerSide));
    }

    // Ensure tower is at origin
    float eps = 0.5f;
    if (fabs(field->towerPos.x) > eps ||
        fabs(field->towerPos.y) > eps ||
        fabs(field->towerPos.z > eps))
    {
        for (int i = 0; i < field->nHelios; ++i)
        {
            field->locs[i] -= field->towerPos;
            field->helios[i].loc -= field->towerPos;
        }
        field->towerPos = {0, 0, 0};
    }

    // For split-ring method, determine row index by comparing y-values
    int nRows = 0;
    for (int i = 1; i < field->nHelios; ++i)
    {
        float y0 = field->helios[i - 1].loc.y;
        float y1 = field->helios[i].loc.y;
        if (fabs(y1 - y0) > 4.0f)
        {
            nRows++;
        }
    }
    Log(fmt::format("{} rows", nRows));

    int rowIdx = 0;
    float y0 = field->helios[0].loc.y;

    // Compute heliostat configurations using extracted module
    for (int i = 0; i < field->nHelios; ++i)
    {
        if (i > 0) y0 = field->helios[i - 1].loc.y;
        float y1 = field->helios[i].loc.y;
        if (i > 0 && fabs(y1 - y0 > 4.0f))
        {
            rowIdx++;
        }

        ComputeHeliostatConfigurations(i, field, aimStrat, sunVec, rowIdx, nRows);
    }

    // Compute voxel positions
    if (verbose)
    {
        size_t voxelMem = nVoxels * sizeof(float) / 1000;
        Log(fmt::format("Num voxels: {:L} ({}, {}, {}) using {:L} KB / 1D array\n",
                        nVoxels, nVoxelsX, nVoxelsY, nVoxelsZ, voxelMem));
    }

    // Initialize voxels with ambient irradiance (0 if not specified)
    for (int i = 0; i < nVoxels; i++)
    {
        irrads[i] = ambient;
    }

    if (ambient > 0.0f && verbose)
    {
        Log(fmt::format("Ambient irradiance: {:.4f} kW/m² added to all voxels", ambient));
    }

#ifndef FFIAM_CPU_ONLY
    if (!useCpu)
    {
        Log(fmt::format("\nPrepping unified memory\n"));
        float3* gFacetOrigins;
        heliostat* gHelios;
        float* gIrrads;

        cudaMallocManaged(&gFacetOrigins, nFacetsMax * sizeof(float3));
        cudaMallocManaged(&gIrrads, nVoxels * sizeof(float));
        cudaMallocManaged(&gHelios, field->nHelios * sizeof(heliostat));

        CheckStatus(cudaMemcpy(gFacetOrigins, facetOrigins, nFacetsMax * sizeof(float3), cudaMemcpyHostToDevice));
        CheckStatus(cudaMemcpy(gIrrads, irrads, nVoxels * sizeof(float), cudaMemcpyHostToDevice));
        CheckStatus(cudaMemcpy(gHelios, field->helios, field->nHelios * sizeof(heliostat), cudaMemcpyHostToDevice));

        uint32_t nThreadsPerBlock = 1024;
        uint32_t nBlocks = (static_cast<uint32_t>(field->nHelios) + nThreadsPerBlock - 1) / nThreadsPerBlock;
        Log(fmt::format("{:L} blocks of size {:L}", nBlocks, nThreadsPerBlock));

        int useOuterRays = (nRaysPerFacet > 9) ? 1 : 0;
        int useCrossPattern = (nRaysPerFacet <= 5) ? 1 : 0;

        Cuda_ComputeHeliostatIrradiance<<<nBlocks, nThreadsPerBlock>>>(gIrrads,
                                                                       gHelios,
                                                                       *field,
                                                                       *helioDesign,
                                                                       gFacetOrigins,
                                                                       sdni,
                                                                       f1,
                                                                       f2,
                                                                       beta,
                                                                       minAttenuation,
                                                                       irradExponent,
                                                                       useOuterRays,
                                                                       useCrossPattern,
                                                                       fluxCorrectionScale,
                                                                       preFocalScale);

        Log(fmt::format("Synchronizing GPU..."));
        CheckStatus(cudaDeviceSynchronize());

        Log(fmt::format("Transferring irradiance values per voxel..."));
        CheckStatus(cudaMemcpy(irrads, gIrrads, nVoxels * sizeof(float), cudaMemcpyDeviceToHost));
        Log(fmt::format("Transferring facet data..."));
        CheckStatus(cudaMemcpy(facetOrigins, gFacetOrigins, nFacetsMax * sizeof(float3), cudaMemcpyDeviceToHost));
        Log(fmt::format("Transfers done. Freeing CUDA resources..."));

        cudaFree(gHelios);
        cudaFree(gFacetOrigins);
        cudaFree(gIrrads);
    }
    else
#endif
    {
        int useOuterRays = (nRaysPerFacet > 9) ? 1 : 0;
        int useCrossPattern = (nRaysPerFacet <= 5) ? 1 : 0;

        CPU_ComputeHeliostatIrradiance(irrads,
                                       *field,
                                       *helioDesign,
                                       facetOrigins,
                                       sdni,
                                       f1, f2,
                                       beta,
                                       minAttenuation,
                                       irradExponent,
                                       useOuterRays,
                                       useCrossPattern,
                                       fluxCorrectionScale,
                                       preFocalScale);
    }

    Log(fmt::format("Processing field data"));

    float totalMoveAngles = 0;

    for (int i = 0; i < field->nHelios; ++i)
    {
        auto helio = &field->helios[i];

        // Store computed facet origins and aim vectors using extracted module
        ComputeAndStoreFacetData(i, helio, helioDesign, facetOrigins, float3Data);

        // Store heliostat data in arrays (for python use)
        field->locs[i] = helio->loc;
        field->aimVs[i] = helio->aimV;
        field->moveAngles[i] = helio->moveAngle;

        float angle = field->moveAngles[i];
        if (!std::isnan(angle))
        {
            totalMoveAngles += field->moveAngles[i];
        }
    }

    float threshold = 0.8f;
    float totalFlux = 0;
    float maxFlux = 0;
    float minFlux = 1;
    int nImpactedVoxels = 0;

    Log(fmt::format("Processing irradiance data..."));
    for (size_t i = 0; i < nVoxels; ++i)
    {
        irrads[i] /= field->voxelSize;

        float flux = irrads[i];
        if (!std::isnan(flux))
        {
            totalFlux += flux;
        }

        if (flux > maxFlux) maxFlux = flux;
        if (flux < minFlux) minFlux = flux;
        if (flux > threshold) nImpactedVoxels++;
    }
    Log(fmt::format("Done."));

    miscData[0] = totalMoveAngles;

    if (verbose)
    {
        Log(fmt::format("Total irrad: {:L}", totalFlux));
        Log(fmt::format("Min-max: {:.1f} to {:.1f}", minFlux, maxFlux));
        Log(fmt::format("# impacted: {:L}", nImpactedVoxels));
        Log(fmt::format("Total movement: {:.0f} deg", totalMoveAngles));
    }

    Log("FieldAnalysis call complete");

    return cudaStatus;
}


// Verify DLL load.
cudaError_t TestPrint(float TestNum)
{
    Log(fmt::format("Testing DLL load."));
    Log(fmt::format("%d", TestNum));

    cudaError_t status = cudaSuccess;
    return status;
}


// Initialize structs and memory pools, then call FieldAnalysis.
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

                            int radius,
                            int zMin, int zMax,
                            float towerH,

                            std::string helioFile,
                            int nHelios,

                            std::string facetFile,
                            int nFacets,
                            int nFacetCols,
                            float2 facetDims,

                            int aimStratId,
                            float3 aimParams,
                            std::string aimFile,

                            bool verbose,

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
                            bool useCpu,
                            int layout)
{
    field_layout field = {};
    field.lat = coords.x;
    field.lng = coords.y;
    field.timezone = timezone;
    field.zMin = zMin;
    field.towerPos = {0, 0, 0};
    field.towerHeight = towerH;
    std::snprintf(field.helioFile, kMaxPathLen, "%s", helioFile.c_str());
    field.nHelios = nHelios;
    field.layout = layout;
    field.radius = radius;
    field.zMax = zMax;
    field.voxelSize = voxelSize;
    field.voxelArea = static_cast<float>(voxelSize) * static_cast<float>(voxelSize);

    heliostat_design helio = {};
    helio.nFacets = nFacets;
    helio.nCols = nFacetCols;
    helio.nRows = helio.nFacets / helio.nCols;
    helio.facetWidth = facetDims.x;
    helio.facetHeight = facetDims.y;
    std::snprintf(helio.facetFile, kMaxPathLen, "%s", facetFile.c_str());

    aim_strategy aimStrat = {};
    aimStrat.type = static_cast<aim_strategy_type>(aimStratId);
    aimStrat.params = aimParams;
    std::snprintf(aimStrat.file, kMaxPathLen, "%s", aimFile.c_str());

    if (aimStrat.type == aim_null)
    {
        aimStrat.type = aim_point;
        aimStrat.params = {0, 0, 100.0f};
    }

    date_info dt = {};
    dt.year = year;
    dt.month = month;
    dt.day = day;
    dt.hour = hour;

    // One pool for smaller datasets, and a bigger one for voxels
    Pool pool;
    size_t chunkSize = MB(256);

    Pool voxelPool;
    size_t voxelChunkSize = MB(256*6);

    std::cout << fmt::format("Memory: {} MB Pool, ({} MB / chunk)", static_cast<int>(nMemory / 1e6), static_cast<int>(chunkSize / 1e6)) <<
        std::endl;
    std::cout << fmt::format("Voxel Memory: {} MB Pool, ({} MB / chunk)", static_cast<int>(nVoxelMemory / 1e6),
                             static_cast<int>(voxelChunkSize / 1e6)) <<
        std::endl;

    pool_init(&pool, arena, nMemory, chunkSize, DEFAULT_ALIGNMENT);
    pool_init(&voxelPool, voxelArena, nVoxelMemory, voxelChunkSize, DEFAULT_ALIGNMENT);

    std::cout << fmt::format("Memory allocation successful") << std::endl;
    std::cout << fmt::format("Allocating chunks") << std::endl;

    // Note: allocation order matters here!
    field.locs = static_cast<float3*>(pool_alloc(&pool));
    field.aimVs = static_cast<float3*>(pool_alloc(&pool));
    field.moveAngles = static_cast<float*>(pool_alloc(&pool));

    auto miscData = static_cast<float*>(pool_alloc(&pool));
    auto float3Data = static_cast<float3*>(pool_alloc(&pool));
    field.helios = static_cast<heliostat*>(pool_alloc(&pool));

    auto irrads = static_cast<float*>(pool_alloc(&voxelPool));

    bool coutToFile = false;

    int status = FieldAnalysis(&field,
                               &helio,
                               &aimStrat,
                               &dt,
                               irrads,
                               miscData,
                               float3Data,
                               verbose,
                               coutToFile,
                               refl,
                               dni,
                               beta,
                               ambient,
                               minAttenuation,
                               irradExponent,
                               nRaysPerFacet,
                               fluxCorrectionScale,
                               preFocalScale,
                               useCpu);

    return status;
}


// Load a preset site config and delegate to InitAnalysis.
extern "C" int InitPresetAnalysis(void* arena,
                                  uint nMemory,
                                  void* voxelArena,
                                  uint nVoxelMemory,
                                  int4 datetime,
                                  int siteId,
                                  int aimId,
                                  float3 aimParams,
                                  bool verbose,
                                  int version,
                                  float refl,
                                  float dni,
                                  float beta,
                                  float minAttenuation,
                                  float irradExponent,
                                  float nRaysPerFacet,
                                  float fluxCorrectionScale,
                                  float preFocalScale,
                                  bool useCpu)
{
    int radius, zMin, zMax, voxelSize;
    get_version_params(version, &radius, &zMin, &zMax, &voxelSize);

    preset_site_config config;
    std::string aimFile;
    if (!get_preset_site_config(siteId, aimId, &config, &aimFile))
    {
        Log("Invalid site ID");
        return 1;
    }

    // Override radius if site specifies it
    if (config.radius > 0)
    {
        radius = config.radius;
    }

    int status = InitAnalysis(arena,
                              nMemory,
                              voxelArena,
                              nVoxelMemory,
                              datetime.x,  // year
                              datetime.y,  // month
                              datetime.z,  // day
                              static_cast<float>(datetime.w),  // hour (convert int to float)
                              config.coords,
                              config.timezone,
                              radius,
                              zMin,
                              zMax,
                              config.towerHeight,
                              config.helioFile,
                              config.nHelios,
                              config.facetFile,
                              config.nFacets,
                              config.nFacetCols,
                              config.facetDims,
                              aimId,
                              aimParams,
                              aimFile,
                              verbose,
                              refl,
                              dni,
                              beta,
                              voxelSize,
                              0.0f,  // ambient (preset analyses use default of 0)
                              minAttenuation,
                              irradExponent,
                              nRaysPerFacet,
                              fluxCorrectionScale,
                              preFocalScale,
                              useCpu,
                              config.layout);
    return status;
}


// Verify ctypes integer passing.
int Py_TestAdd(int a, int b)
{
    return a + b;
}


// Verify DLL loading from Python.
void Py_TestPrint()
{
    Log("Test print from DLL");
}


// ctypes entry point. Converts char* paths to std::string (prepends "data/") and calls InitAnalysis.
extern "C" int PyAnalysis(void* arena,
                          uint nMemory,
                          void* voxelArena,
                          uint nVoxelMemory,

                          int year,
                          int month,
                          int day,
                          float hour,

                          float lat,
                          float lng,
                          float timezone,

                          int radius,
                          int zMin, int zMax,
                          float towerH,

                          char* helioFile,
                          int nHelios,

                          char* facetFile,
                          int nFacets,
                          int nFacetCols,
                          float facetW,
                          float facetH,

                          int aimStratId,
                          float aim1, float aim2, float aim3,
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
                          int layout)
{
    std::string helioFileStr;
    if (helioFile != nullptr && strlen(helioFile) > 0)
    {
        helioFileStr = "data/" + std::string(helioFile);
    }

    std::string facetFileStr;
    if (facetFile != nullptr && strlen(facetFile) > 0)
    {
        facetFileStr = "data/" + std::string(facetFile);
    }

    std::string aimFileStr;
    if (aimFile != nullptr && strlen(aimFile) > 0)
    {
        aimFileStr = "data/" + std::string(aimFile);
    }

    const float2 coords = {lat, lng};
    const float2 facetDims = {facetW, facetH};
    const float3 aimParams = {aim1, aim2, aim3};

    const int status = InitAnalysis(arena,
                                    nMemory,
                                    voxelArena,
                                    nVoxelMemory,
                                    year,
                                    month,
                                    day,
                                    hour,
                                    coords,
                                    timezone,
                                    radius,
                                    zMin,
                                    zMax,
                                    towerH,
                                    helioFileStr,
                                    nHelios,
                                    facetFileStr,
                                    nFacets,
                                    nFacetCols,
                                    facetDims,
                                    aimStratId,
                                    aimParams,
                                    aimFileStr,
                                    verbose,
                                    refl, dni, beta,
                                    voxelSize,
                                    ambient,
                                    minAttenuation,
                                    irradExponent,
                                    nRaysPerFacet,
                                    fluxCorrectionScale,
                                    preFocalScale,
                                    useCpu,
                                    layout);  // grid vs radial generation (when no helioFile)
    return status;
}


#if !defined(FFIAM_CPU_ONLY) && !defined(FFIAM_NO_MAIN)
// =============================================================================
// Standalone executable entry point and helpers (CUDA-only)
// =============================================================================
// Pulls in CudaBuffer / CudaUnifiedBuffer for self-test, so it's excluded from
// the CPU-only build. The shared library doesn't need main() either way.
// FFIAM_NO_MAIN is set by the test target so ffiam_tests can link its own main().


// Returns true if a and b are within tol of each other.
bool approxEqual(float a, float b, float tol = 0.01f)
{
    return std::abs(a - b) < tol;
}

// Checks a float3 against expected components and prints PASS/FAIL to stdout.
bool verifyVector(const float3& v, float x, float y, float z, const char* name)
{
    bool pass = approxEqual(v.x, x) && approxEqual(v.y, y) && approxEqual(v.z, z);
    if (pass)
    {
        std::cout << "[PASS] " << name << std::endl;
    }
    else
    {
        std::cout << "[FAIL] " << name << " expected (" << x << "," << y << "," << z
            << ") got " << v << std::endl;
    }
    return pass;
}

// Standalone verification: memory, field analysis, coordinate conversions, voxel grid, RAII wrappers.
int main()
{
    int failCount = 0;

    std::cout << "\n========== FFIAM Quick Verification Test ==========\n\n";

    // Test 1: Memory allocation and field analysis
    std::cout << "--- Test: Field Analysis ---\n";
    constexpr size_t nMemory = GB(2);
    void* arena = malloc(nMemory);
    constexpr size_t nVoxelMemory = GB(2);
    void* voxelArena = malloc(nVoxelMemory);

    if (!arena || !voxelArena)
    {
        std::cout << "[FAIL] Memory allocation failed\n";
        return 1;
    }
    std::cout << "[PASS] Memory allocation (4GB total)\n";

    constexpr int4 datetime = {2025, 6, 21, 12};
    constexpr float3 aimParams = {20, 0, 100};

    const int status = InitPresetAnalysis(arena, nMemory, voxelArena, nVoxelMemory,
                                          datetime, site_radial, aim_point,
                                          aimParams, true, 3);

    if (status != 0)
    {
        std::cout << "[FAIL] InitPresetAnalysis returned " << status << "\n";
        failCount++;
    }
    else
    {
        std::cout << "[PASS] Field analysis completed\n";
    }

    // Test 2: AzEl to Cartesian conversions
    std::cout << "\n--- Test: AzEl to Cartesian Conversions ---\n";

    if (!verifyVector(ConvertAzElToCartesian(0.0f, 0.0f), 0, 1, 0, "North (Az=0, El=0)"))
        failCount++;
    if (!verifyVector(ConvertAzElToCartesian(90.0f, 0.0f), 1, 0, 0, "East (Az=90, El=0)"))
        failCount++;
    if (!verifyVector(ConvertAzElToCartesian(180.0f, 0.0f), 0, -1, 0, "South (Az=180, El=0)"))
        failCount++;
    if (!verifyVector(ConvertAzElToCartesian(270.0f, 0.0f), -1, 0, 0, "West (Az=270, El=0)"))
        failCount++;
    if (!verifyVector(ConvertAzElToCartesian(0.0f, 90.0f), 0, 0, 1, "Zenith (Az=0, El=90)"))
        failCount++;
    if (!verifyVector(ConvertAzElToCartesian(0.0f, -90.0f), 0, 0, -1, "Nadir (Az=0, El=-90)"))
        failCount++;

    // Test 3: Voxel grid computation
    std::cout << "\n--- Test: Voxel Grid Dimensions ---\n";
    int nX, nY, nZ;
    const int nVoxels = ComputeVoxelGridDimensions(600, 4, 100, 2, &nX, &nY, &nZ);

    if (nX == 600 && nY == 600 && nZ == 49 && nVoxels == 600 * 600 * 49)
    {
        std::cout << "[PASS] Voxel grid: " << nX << "x" << nY << "x" << nZ
            << " = " << nVoxels << " voxels\n";
    }
    else
    {
        std::cout << "[FAIL] Voxel grid expected 600x600x49, got "
            << nX << "x" << nY << "x" << nZ << "\n";
        failCount++;
    }

    // Test 4: RAII Memory wrappers
    std::cout << "\n--- Test: RAII Memory Wrappers ---\n";
    {
        // Test MemoryPool (1MB pool, 64KB chunks)
        ffiam::MemoryPool pool(1024 * 1024, 64 * 1024);
        if (pool.valid())
        {
            void* chunk1 = pool.allocate();
            void* chunk2 = pool.allocate();
            if (chunk1 && chunk2 && chunk1 != chunk2)
            {
                std::cout << "[PASS] MemoryPool allocation\n";
                pool.deallocate(chunk1);
                pool.deallocate(chunk2);
            }
            else
            {
                std::cout << "[FAIL] MemoryPool allocation failed\n";
                failCount++;
            }
        }
        else
        {
            std::cout << "[FAIL] MemoryPool initialization failed\n";
            failCount++;
        }

        // Test CudaBuffer
        ffiam::CudaBuffer gpuBuf(1024);
        if (gpuBuf.valid())
        {
            float hostData[256] = {1.0f, 2.0f, 3.0f};
            cudaError_t err = gpuBuf.copyToDevice(hostData, sizeof(hostData));
            if (err == cudaSuccess)
            {
                std::cout << "[PASS] CudaBuffer allocation and copy\n";
            }
            else
            {
                std::cout << "[FAIL] CudaBuffer copy failed\n";
                failCount++;
            }
        }
        else
        {
            std::cout << "[FAIL] CudaBuffer allocation failed\n";
            failCount++;
        }

        // Test CudaUnifiedBuffer
        ffiam::CudaUnifiedBuffer unifiedBuf(1024);
        if (unifiedBuf.valid())
        {
            float* data = unifiedBuf.as<float>();
            data[0] = 42.0f;
            if (data[0] == 42.0f)
            {
                std::cout << "[PASS] CudaUnifiedBuffer allocation\n";
            }
            else
            {
                std::cout << "[FAIL] CudaUnifiedBuffer access failed\n";
                failCount++;
            }
        }
        else
        {
            std::cout << "[FAIL] CudaUnifiedBuffer allocation failed\n";
            failCount++;
        }
    } // RAII cleanup happens here

    // Summary
    std::cout << "\n========== Verification Complete ==========\n";
    if (failCount == 0)
    {
        std::cout << "All tests PASSED\n";
    }
    else
    {
        std::cout << failCount << " test(s) FAILED\n";
    }

    free(arena);
    free(voxelArena);
    return failCount;
}
#endif  // !FFIAM_CPU_ONLY && !FFIAM_NO_MAIN
