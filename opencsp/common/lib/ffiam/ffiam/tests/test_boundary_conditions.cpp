// Copyright Sandia National Laboratories. All rights reserved.
// Boundary-condition tests for the CPU irradiance backend.

#include "test_framework.h"
#include "ffiam/ffiam.h"
#include "ffiam/types.h"
#include "core/constants.h"

#include <vector>


namespace {

using namespace ffiam::constants;

inline field_layout MakeField(int voxelSize, int radius, int zMin, int zMax)
{
    field_layout f = {};
    f.voxelSize = voxelSize;
    f.radius = radius;
    f.zMin = zMin;
    f.zMax = zMax;
    const int nx = (radius * 2) / voxelSize;
    const int nz = (zMax - zMin) / voxelSize + 1;
    f.nVoxels = nx * nx * nz;
    return f;
}

inline heliostat MakeHelio(float x, float y)
{
    heliostat h = {};
    h.id = 0;
    h.loc = {x, y, 0.0f};
    h.focalPt = {0.0f, 0.0f, 60.0f};
    h.focalLen = 70.0f;
    h.horizV = {1.0f, 0.0f, 0.0f};
    h.vertV  = {0.0f, 0.0f, 1.0f};
    return h;
}

inline heliostat_design MakeDesign()
{
    heliostat_design d = {};
    d.nFacets = 1;
    d.nRows = 1;
    d.nCols = 1;
    d.facetWidth = 1.2f;
    d.facetHeight = 1.2f;
    d.facetsImported = false;
    return d;
}

inline int CountNonZero(const std::vector<float>& v)
{
    int n = 0;
    for (float x : v) if (x > 0.0f) ++n;
    return n;
}

inline float SumOf(const std::vector<float>& v)
{
    float s = 0.0f;
    for (float x : v) s += x;
    return s;
}

}  // namespace


TEST(BoundaryConditions, SingleHelioAtFieldEdge)
{
    const int radius = 100;
    field_layout field = MakeField(2, radius, 0, 100);
    field.nHelios = 1;
    heliostat h = MakeHelio(static_cast<float>(radius - 1), 0.0f);
    field.helios = &h;
    heliostat_design d = MakeDesign();

    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeHeliostatIrradiance(out.data(), field, d, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   0, 1, 0.0f, 1.0f);
    EXPECT_TRUE(CountNonZero(out) > 0);
    return true;
}

TEST(BoundaryConditions, HelioJustOutsideRadius_DoesNotCrash)
{
    // Radius bound is enforced per-voxel on the ray walk, not on the heliostat
    // origin — rays still trace, but out-of-grid voxels don't accumulate.
    const int radius = 50;
    field_layout field = MakeField(2, radius, 0, 100);
    field.nHelios = 1;
    heliostat h = MakeHelio(static_cast<float>(radius + 5), 0.0f);
    field.helios = &h;
    heliostat_design d = MakeDesign();

    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeHeliostatIrradiance(out.data(), field, d, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   0, 1, 0.0f, 1.0f);
    float s = SumOf(out);
    EXPECT_FALSE(std::isnan(s));
    EXPECT_TRUE(s >= 0.0f);
    return true;
}

TEST(BoundaryConditions, RayWalkOverflowsVoxelHitCache)
{
    // The ray walker caches the last VOXEL_HIT_CACHE_SIZE hits to skip
    // duplicates; this ray touches enough unique voxels to wrap the cache.
    field_layout field = MakeField(2, 100, 0, 100);

    const float3 rayOrigin{0.0f, 0.0f, 5.0f};
    const float3 focalPt{0.0f, 0.0f, 100.0f};
    const float focalLen = 95.0f;

    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeIrradFromFacetRay(out.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, 0.001f, 1.0f, 2.0f, 0.0f, 1.0f);

    int hits = CountNonZero(out);
    EXPECT_TRUE(hits > VOXEL_HIT_CACHE_SIZE);
    EXPECT_FALSE(std::isnan(SumOf(out)));
    return true;
}

TEST(BoundaryConditions, FocalPointAtZMin)
{
    field_layout field = MakeField(2, 100, 4, 100);

    const float3 rayOrigin{10.0f, 10.0f, 0.0f};
    const float3 focalPt{0.0f, 0.0f, 4.0f};
    const float focalLen = length(focalPt - rayOrigin);

    // f2=0.05 keeps dist*f2 + atten nonzero at fR=1, dodging the post-focal
    // singularity.
    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeIrradFromFacetRay(out.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, 0.05f, 1.0f, 2.0f, 0.0f, 1.0f);
    EXPECT_FALSE(std::isnan(SumOf(out)));
    return true;
}

TEST(BoundaryConditions, FocalPointAboveZMax)
{
    field_layout field = MakeField(2, 100, 0, 80);

    const float3 rayOrigin{20.0f, 20.0f, 5.0f};
    const float3 focalPt{0.0f, 0.0f, 200.0f};
    const float focalLen = length(focalPt - rayOrigin);

    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeIrradFromFacetRay(out.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, 0.001f, 1.0f, 2.0f, 0.0f, 1.0f);
    EXPECT_TRUE(CountNonZero(out) > 0);
    EXPECT_FALSE(std::isnan(SumOf(out)));
    return true;
}

TEST(BoundaryConditions, RayStartingOutsideRadiusIsSkipped)
{
    field_layout field = MakeField(2, 50, 0, 100);

    const float3 rayOrigin{-60.0f, 0.0f, 0.0f};
    const float3 focalPt{-60.0f, 0.0f, 100.0f};
    const float focalLen = 100.0f;

    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeIrradFromFacetRay(out.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, 0.001f, 1.0f, 2.0f, 0.0f, 1.0f);
    EXPECT_EQ(0, CountNonZero(out));
    return true;
}
