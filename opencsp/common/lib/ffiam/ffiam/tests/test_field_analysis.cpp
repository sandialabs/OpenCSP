// Copyright Sandia National Laboratories. All rights reserved.
// Edge-branch tests for FieldAnalysis (src/main/ffiam.cu). All tests run with
// useCpu=true; the pre-compute branches under test run identically on both
// backends. Test target is CUDA-aware (ffiam.cu uses cudaError_t) — gated by
// BUILD_CUDA_TESTS in CMakeLists.txt.

#include "test_framework.h"
#include "ffiam/ffiam.h"
#include "ffiam/types.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>


namespace {

namespace fs = std::filesystem;

struct FieldHarness {
    std::vector<heliostat> helios;
    std::vector<float3> locs;
    std::vector<float3> aimVs;
    std::vector<float> moveAngles;

    std::vector<float> irrads;
    std::vector<float> miscData;
    std::vector<float3> float3Data;

    field_layout field{};
    heliostat_design design{};
    date_info dt{};

    FieldHarness(int nHelios, int radius = 100, int zMin = 0, int zMax = 50, int voxelSize = 4)
        : helios(std::max(nHelios, 1))
        , locs(std::max(nHelios, 1), float3{0,0,0})
        , aimVs(std::max(nHelios, 1), float3{0,0,1})
        , moveAngles(std::max(nHelios, 1), 0.0f)
        , miscData(10, 0.0f)
        , float3Data(std::max(nHelios, 1) * 2 * 50, float3{0,0,0})
    {
        field.lat = 34.96f;
        field.lng = -106.51f;
        field.timezone = -7.0f;
        field.radius = radius;
        field.zMin = zMin;
        field.zMax = zMax;
        field.voxelSize = voxelSize;
        field.voxelArea = static_cast<float>(voxelSize * voxelSize);
        field.towerPos = {0, 0, 0};
        field.towerHeight = 60.0f;

        field.nHelios = nHelios;
        field.helios = helios.data();
        field.locs = locs.data();
        field.aimVs = aimVs.data();
        field.moveAngles = moveAngles.data();

        const int nx = (radius * 2) / voxelSize;
        const int nz = (zMax - zMin) / voxelSize + 1;
        const int nVoxels = nx * nx * nz;
        irrads.assign(nVoxels, 0.0f);

        design.nFacets = 1;
        design.nRows = 1;
        design.nCols = 1;
        design.pivotHeight = 0.0f;
        design.pivotOffset = 0.0f;
        design.facetWidth = 1.2f;
        design.facetHeight = 1.2f;
        design.facetsImported = false;

        dt.year = 2025;
        dt.month = 6;
        dt.day = 21;
        dt.hour = 12.0f;
    }
};

struct TempCsv {
    fs::path path;

    explicit TempCsv(const std::string& content) {
        static int counter = 0;
        path = fs::temp_directory_path() /
               ("ffiam_fa_test_" + std::to_string(++counter) + ".csv");
        std::ofstream f(path);
        f << content;
        f.close();
    }

    ~TempCsv() {
        std::error_code ec;
        fs::remove(path, ec);
    }

    std::string str() const { return path.string(); }
};

inline aim_strategy MakePointAim() {
    aim_strategy a{};
    a.type = aim_point;
    a.params = {0.0f, 0.0f, 60.0f};
    a.file = "";
    return a;
}

}  // namespace


TEST(FieldAnalysis, ZeroHeliosReturnsInvalidValue)
{
    FieldHarness h(0);
    aim_strategy aim = MakePointAim();

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaErrorInvalidValue), static_cast<int>(status));
    return true;
}

TEST(FieldAnalysis, AimCsvEmptyPathReturnsEarly)
{
    FieldHarness h(4);
    aim_strategy aim{};
    aim.type = aim_data_csv;
    aim.params = {0, 0, 0};
    aim.file = "";

    for (auto& v : h.irrads) v = -123.0f;

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaSuccess), static_cast<int>(status));
    EXPECT_NEAR(-123.0f, h.irrads[0], 1e-3f);
    return true;
}

TEST(FieldAnalysis, AimCsvRowMismatchReturnsEarly)
{
    FieldHarness h(4);
    TempCsv csv("id,azimuth,elevation\n0,0,90\n1,0,90\n");
    aim_strategy aim{};
    aim.type = aim_data_csv;
    aim.params = {0, 0, 0};
    aim.file = csv.str();

    for (auto& v : h.irrads) v = -123.0f;

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaSuccess), static_cast<int>(status));
    EXPECT_NEAR(-123.0f, h.irrads[0], 1e-3f);
    return true;
}

TEST(FieldAnalysis, MissingFacetFileReturnsEarly)
{
    FieldHarness h(4);
    aim_strategy aim = MakePointAim();
    h.design.facetFile = "/nonexistent/no_such_facet_file.csv";

    for (auto& v : h.irrads) v = -123.0f;

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaSuccess), static_cast<int>(status));
    EXPECT_NEAR(-123.0f, h.irrads[0], 1e-3f);
    return true;
}

TEST(FieldAnalysis, TowerPositionOffsetTriggersRebase)
{
    FieldHarness h(4, 100, 0, 40, 4);
    h.field.towerPos = {10.0f, 5.0f, 0.0f};
    aim_strategy aim = MakePointAim();

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaSuccess), static_cast<int>(status));
    EXPECT_NEAR(0.0f, h.field.towerPos.x, 1e-3f);
    EXPECT_NEAR(0.0f, h.field.towerPos.y, 1e-3f);
    EXPECT_NEAR(0.0f, h.field.towerPos.z, 1e-3f);
    EXPECT_TRUE(h.field.locs[0].x < -10.0f + 1e-3f);
    EXPECT_TRUE(h.field.locs[0].y < -5.0f + 1e-3f);
    return true;
}

TEST(FieldAnalysis, AimCsvMatchingRunsToCompletion)
{
    FieldHarness h(4);
    TempCsv csv("id,azimuth,elevation\n"
                "0,0,75\n"
                "1,0,75\n"
                "2,0,75\n"
                "3,0,75\n");
    aim_strategy aim{};
    aim.type = aim_data_csv;
    aim.params = {0, 0, 0};
    aim.file = csv.str();

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaSuccess), static_cast<int>(status));
    EXPECT_TRUE(h.miscData[0] >= 0.0f);
    return true;
}

TEST(FieldAnalysis, HelioFileImportOverwritesLocations)
{
    FieldHarness h(4);

    TempCsv csv("id,x,y,z,nFacets,nRows,nCols,pivotHeight,pivotOffset,facetWidth,facetHeight\n"
                "0,17.0,-23.0,0.0,1,1,1,0.0,0.0,1.2,1.2\n"
                "1,-31.0,11.0,0.0,1,1,1,0.0,0.0,1.2,1.2\n"
                "2,5.5,42.5,0.0,1,1,1,0.0,0.0,1.2,1.2\n");
    h.field.helioFile = csv.str();
    h.field.radius = 100;
    aim_strategy aim = MakePointAim();

    cudaError_t status = FieldAnalysis(&h.field, &h.design, &aim, &h.dt,
                                       h.irrads.data(), h.miscData.data(), h.float3Data.data(),
                                       false, false,
                                       0.9f, 0.1f, 0.0094f, 0.0f, 1.0f, 2.0f, 9.0f, 1.0f, 1.0f,
                                       true);
    EXPECT_EQ(static_cast<int>(cudaSuccess), static_cast<int>(status));
    EXPECT_EQ(3, h.field.nHelios);
    EXPECT_NEAR(17.0f, h.helios[0].loc.x, 1e-3f);
    EXPECT_NEAR(-23.0f, h.helios[0].loc.y, 1e-3f);
    EXPECT_NEAR(-31.0f, h.helios[1].loc.x, 1e-3f);
    EXPECT_NEAR(11.0f, h.helios[1].loc.y, 1e-3f);
    EXPECT_NEAR(5.5f, h.helios[2].loc.x, 1e-3f);
    EXPECT_NEAR(42.5f, h.helios[2].loc.y, 1e-3f);
    return true;
}
