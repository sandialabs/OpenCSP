// Unit tests for src/io/csv_reader.cpp — raw and *Safe import variants.

#include "test_framework.h"
#include "ffiam/types.h"
#include "io/csv_reader.h"
#include "util/error.h"
#include "util/math.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace fs = std::filesystem;

// Absolute path bypasses ResolvePath's data/ prefix logic.
struct TempCsv {
    fs::path path;

    explicit TempCsv(const std::string& content) {
        static int counter = 0;
        path = fs::temp_directory_path() /
               ("ffiam_test_" + std::to_string(++counter) + ".csv");
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

constexpr const char* HELIO_HEADER =
    "id,x,y,z,nFacets,nRows,nCols,pivotHeight,pivotOffset,facetWidth,facetHeight\n";

constexpr const char* AIM_HEADER = "id,azimuth,elevation\n";
constexpr const char* FACET_HEADER = "id,x,y,z\n";

} // namespace


TEST(CsvReader, ImportHeliostat_HappyPath)
{
    TempCsv csv(std::string(HELIO_HEADER) +
                "0,10.0,20.0,0.5,25,5,5,0.0,1.0,1.2,1.2\n"
                "1,-5.0,15.5,0.5,25,5,5,0.0,1.0,1.2,1.2\n"
                "2,0.0,-30.0,0.5,25,5,5,0.0,1.0,1.2,1.2\n");
    float3 locs[3]{};
    heliostat_design design{};

    int n = ImportHeliostatData(csv.str(), locs, &design);

    EXPECT_EQ(3, n);
    EXPECT_NEAR(10.0f, locs[0].x, 1e-3f);
    EXPECT_NEAR(20.0f, locs[0].y, 1e-3f);
    EXPECT_NEAR(-5.0f, locs[1].x, 1e-3f);
    EXPECT_NEAR(15.5f, locs[1].y, 1e-3f);
    EXPECT_EQ(25, design.nFacets);
    EXPECT_EQ(5, design.nRows);
    EXPECT_EQ(5, design.nCols);
    EXPECT_NEAR(1.2f, design.facetWidth, 1e-3f);
    EXPECT_NEAR(1.2f, design.facetHeight, 1e-3f);
    return true;
}

TEST(CsvReader, ImportHeliostat_MissingFile)
{
    float3 locs[1]{};
    heliostat_design design{};
    int n = ImportHeliostatData("/nonexistent/path/does_not_exist.csv", locs, &design);
    EXPECT_EQ(0, n);
    return true;
}

TEST(CsvReader, ImportHeliostat_HeaderOnly)
{
    TempCsv csv(HELIO_HEADER);
    float3 locs[1]{};
    heliostat_design design{};
    int n = ImportHeliostatData(csv.str(), locs, &design);
    EXPECT_EQ(0, n);
    return true;
}


TEST(CsvReader, ImportAim_HappyPath)
{
    // az=0,el=90 -> straight up (0,0,1); az=90,el=0 -> east (1,0,0)
    TempCsv csv(std::string(AIM_HEADER) +
                "0,0.0,90.0\n"
                "1,90.0,0.0\n");
    float3 aims[2]{};

    int n = ImportAimData(csv.str(), aims);

    EXPECT_EQ(2, n);
    EXPECT_NEAR(0.0f, aims[0].x, 1e-4f);
    EXPECT_NEAR(0.0f, aims[0].y, 1e-4f);
    EXPECT_NEAR(1.0f, aims[0].z, 1e-4f);
    EXPECT_NEAR(1.0f, aims[1].x, 1e-4f);
    EXPECT_NEAR(0.0f, aims[1].y, 1e-4f);
    EXPECT_NEAR(0.0f, aims[1].z, 1e-4f);
    return true;
}

TEST(CsvReader, ImportAim_MissingFile)
{
    float3 aims[1]{};
    int n = ImportAimData("/nonexistent/aim_file.csv", aims);
    EXPECT_EQ(0, n);
    return true;
}


TEST(CsvReader, ImportFacets_HappyPath)
{
    TempCsv csv(std::string(FACET_HEADER) +
                "0,-1.2,-1.2,0.0\n"
                "1,0.0,-1.2,0.0\n"
                "2,1.2,-1.2,0.0\n"
                "3,-1.2,0.0,0.0\n");
    float3 origins[4]{};

    int n = ImportFacetCoordinates(csv.str(), origins);

    EXPECT_EQ(4, n);
    EXPECT_NEAR(-1.2f, origins[0].x, 1e-3f);
    EXPECT_NEAR(-1.2f, origins[0].y, 1e-3f);
    EXPECT_NEAR(0.0f, origins[0].z, 1e-3f);
    EXPECT_NEAR(1.2f, origins[2].x, 1e-3f);
    return true;
}

TEST(CsvReader, ImportFacets_MissingFile)
{
    float3 origins[1]{};
    int n = ImportFacetCoordinates(std::string("/nonexistent/facet.csv"), origins);
    EXPECT_EQ(0, n);
    return true;
}


TEST(CsvReader, ImportHeliostatSafe_NullLocs)
{
    heliostat_design design{};
    auto r = ImportHeliostatDataSafe("ignored.csv", nullptr, &design);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::InvalidParameter),
              static_cast<int>(r.error));
    return true;
}

TEST(CsvReader, ImportHeliostatSafe_NullDesign)
{
    float3 locs[1]{};
    auto r = ImportHeliostatDataSafe("ignored.csv", locs, nullptr);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::InvalidParameter),
              static_cast<int>(r.error));
    return true;
}

TEST(CsvReader, ImportHeliostatSafe_MissingFile)
{
    float3 locs[1]{};
    heliostat_design design{};
    auto r = ImportHeliostatDataSafe("/nonexistent/path.csv", locs, &design);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::FileNotFound),
              static_cast<int>(r.error));
    EXPECT_EQ(0, r.rowCount);
    return true;
}

TEST(CsvReader, ImportHeliostatSafe_HappyPath)
{
    TempCsv csv(std::string(HELIO_HEADER) +
                "0,10.0,20.0,0.0,25,5,5,0.0,1.0,1.2,1.2\n"
                "1,11.0,21.0,0.0,25,5,5,0.0,1.0,1.2,1.2\n");
    float3 locs[2]{};
    heliostat_design design{};

    auto r = ImportHeliostatDataSafe(csv.str(), locs, &design);

    EXPECT_TRUE(r.ok());
    EXPECT_EQ(2, r.rowCount);
    EXPECT_NEAR(10.0f, locs[0].x, 1e-3f);
    EXPECT_EQ(25, design.nFacets);
    return true;
}

TEST(CsvReader, ImportHeliostatSafe_ShortRowsSkipped)
{
    TempCsv csv(std::string(HELIO_HEADER) +
                "0,10.0,20.0,0.0,25,5,5,0.0,1.0,1.2,1.2\n"
                "1,1.0,2.0,3.0\n"
                "2,30.0,40.0,0.0,25,5,5,0.0,1.0,1.2,1.2\n");
    float3 locs[3]{};
    heliostat_design design{};

    auto r = ImportHeliostatDataSafe(csv.str(), locs, &design);

    EXPECT_TRUE(r.ok());
    EXPECT_EQ(2, r.rowCount);
    EXPECT_NEAR(10.0f, locs[0].x, 1e-3f);
    EXPECT_NEAR(30.0f, locs[1].x, 1e-3f);
    return true;
}


TEST(CsvReader, ImportAimSafe_NullArray)
{
    auto r = ImportAimDataSafe("ignored.csv", nullptr);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::InvalidParameter),
              static_cast<int>(r.error));
    return true;
}

TEST(CsvReader, ImportAimSafe_MissingFile)
{
    float3 aims[1]{};
    auto r = ImportAimDataSafe("/nonexistent/aim.csv", aims);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::FileNotFound),
              static_cast<int>(r.error));
    return true;
}

TEST(CsvReader, ImportAimSafe_HappyPath)
{
    TempCsv csv(std::string(AIM_HEADER) +
                "0,0.0,90.0\n"
                "1,180.0,0.0\n");
    float3 aims[2]{};
    auto r = ImportAimDataSafe(csv.str(), aims);
    EXPECT_TRUE(r.ok());
    EXPECT_EQ(2, r.rowCount);
    EXPECT_NEAR(1.0f, aims[0].z, 1e-4f);
    EXPECT_NEAR(-1.0f, aims[1].y, 1e-4f);
    return true;
}

TEST(CsvReader, ImportAimSafe_ShortRowsSkipped)
{
    TempCsv csv(std::string(AIM_HEADER) +
                "0,0.0,90.0\n"
                "incomplete\n"
                "1,90.0,0.0\n");
    float3 aims[2]{};
    auto r = ImportAimDataSafe(csv.str(), aims);
    EXPECT_TRUE(r.ok());
    EXPECT_EQ(2, r.rowCount);
    return true;
}


TEST(CsvReader, ImportFacetsSafe_NullArray)
{
    auto r = ImportFacetCoordinatesSafe("ignored.csv", nullptr);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::InvalidParameter),
              static_cast<int>(r.error));
    return true;
}

TEST(CsvReader, ImportFacetsSafe_MissingFile)
{
    float3 origins[1]{};
    auto r = ImportFacetCoordinatesSafe("/nonexistent/facet.csv", origins);
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(static_cast<int>(ffiam::ErrorCode::FileNotFound),
              static_cast<int>(r.error));
    return true;
}

TEST(CsvReader, ImportFacetsSafe_HappyPath)
{
    TempCsv csv(std::string(FACET_HEADER) +
                "0,-1.0,-1.0,0.0\n"
                "1,1.0,1.0,0.0\n");
    float3 origins[2]{};
    auto r = ImportFacetCoordinatesSafe(csv.str(), origins);
    EXPECT_TRUE(r.ok());
    EXPECT_EQ(2, r.rowCount);
    EXPECT_NEAR(-1.0f, origins[0].x, 1e-3f);
    EXPECT_NEAR(1.0f, origins[1].x, 1e-3f);
    return true;
}

TEST(CsvReader, ImportFacetsSafe_ShortRowsSkipped)
{
    TempCsv csv(std::string(FACET_HEADER) +
                "0,-1.0,-1.0,0.0\n"
                "bad,row\n"
                "1,1.0,1.0,0.0\n");
    float3 origins[2]{};
    auto r = ImportFacetCoordinatesSafe(csv.str(), origins);
    EXPECT_TRUE(r.ok());
    EXPECT_EQ(2, r.rowCount);
    return true;
}

// CellToFloat falls back to 0.0f on parse failure rather than throwing.
TEST(CsvReader, ImportFacets_ParseFailureZeros)
{
    TempCsv csv(std::string(FACET_HEADER) +
                "0,not_a_number,1.0,0.0\n");
    float3 origins[1]{};
    int n = ImportFacetCoordinates(csv.str(), origins);
    EXPECT_EQ(1, n);
    EXPECT_NEAR(0.0f, origins[0].x, 1e-6f);
    EXPECT_NEAR(1.0f, origins[0].y, 1e-3f);
    return true;
}
