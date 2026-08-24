// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// Unit tests for the CPU irradiance backend.

#include "test_framework.h"
#include "ffiam/ffiam.h"
#include "ffiam/types.h"
#include "compute/ray_math.h"

#include <vector>


namespace {

inline field_layout MakeSmallField(int voxelSize = 2,
                                   int radius = 100,
                                   int zMin = 0,
                                   int zMax = 100)
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

inline float MaxOf(const std::vector<float>& v)
{
    float m = 0.0f;
    for (float x : v) if (x > m) m = x;
    return m;
}

inline float SumOf(const std::vector<float>& v)
{
    float s = 0.0f;
    for (float x : v) s += x;
    return s;
}

inline int CountHitVoxels(const std::vector<float>& v)
{
    int n = 0;
    for (float x : v) if (x > 0.0f) ++n;
    return n;
}

}  // namespace


// preFocalScale sits in the denominator of facetIrrad: halving it raises peak.
// Focal point beyond zMax keeps fR < 0.5 throughout — pure pre-focal regime,
// no singularity.
TEST(CpuIrradiance, PreFocalScale_HalvesAttenuation_AtMidRay) {
    field_layout field = MakeSmallField();
    std::vector<float> sym(field.nVoxels, 0.0f);
    std::vector<float> scaled(field.nVoxels, 0.0f);

    const float3 rayOrigin = {0.0f, 0.0f, 10.0f};
    const float3 focalPt   = {0.0f, 0.0f, 200.0f};
    const float focalLen   = 190.0f;
    const float f1 = 1.0f, f2 = 0.0f;  // f2=0 isolates the attenuation term

    CPU_ComputeIrradFromFacetRay(sym.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, f1, f2, 1.0f, 2.0f, 0.0f, 1.0f);
    CPU_ComputeIrradFromFacetRay(scaled.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, f1, f2, 1.0f, 2.0f, 0.0f, 0.5f);

    EXPECT_TRUE(MaxOf(scaled) > MaxOf(sym));
    return true;
}

// minAtten caps fminf(rawAtten, blendedMin) below rawAtten when
// fR < BLEND_START_FR — smaller denominator -> larger peak.
TEST(CpuIrradiance, MinAttenuation_BlendCap_RaisesPeakIrradiance) {
    field_layout field = MakeSmallField();
    std::vector<float> noFloor(field.nVoxels, 0.0f);
    std::vector<float> withFloor(field.nVoxels, 0.0f);

    const float3 rayOrigin = {0.0f, 0.0f, 10.0f};
    const float3 focalPt   = {0.0f, 0.0f, 200.0f};
    const float focalLen   = 190.0f;

    CPU_ComputeIrradFromFacetRay(noFloor.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f);
    CPU_ComputeIrradFromFacetRay(withFloor.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, 0.0f, 0.25f, 2.0f, 0.0f, 1.0f);

    EXPECT_TRUE(MaxOf(withFloor) > MaxOf(noFloor));
    return true;
}

// Past focus (fR > 2) the piecewise flux correction factor is 5x; once enough
// voxels sit in that regime it dominates the sum. f2 > 0 keeps the denominator
// nonzero at fR=1.
TEST(CpuIrradiance, FluxCorrection_AmplifiesPostFocalSum) {
    field_layout field = MakeSmallField(2, 100, 0, 200);
    std::vector<float> noCorr(field.nVoxels, 0.0f);
    std::vector<float> withCorr(field.nVoxels, 0.0f);

    const float3 rayOrigin = {0.0f, 0.0f, 5.0f};
    const float3 focalPt   = {0.0f, 0.0f, 25.0f};
    const float focalLen   = 20.0f;
    const float f2_safe    = 0.05f;

    CPU_ComputeIrradFromFacetRay(noCorr.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, f2_safe, 1.0f, 2.0f, 0.0f, 1.0f);
    CPU_ComputeIrradFromFacetRay(withCorr.data(), &field, focalPt, rayOrigin, focalLen,
                                 0, 1.0f, f2_safe, 1.0f, 2.0f, 1.0f, 1.0f);

    EXPECT_TRUE(SumOf(withCorr) > SumOf(noCorr) * 1.1f);
    return true;
}


namespace {

inline heliostat MakeTestHelio()
{
    heliostat h = {};
    h.id = 0;
    h.loc = {30.0f, 30.0f, 0.0f};
    h.focalPt = {0.0f, 0.0f, 60.0f};
    h.focalLen = 70.0f;
    h.horizV = {1.0f, 0.0f, 0.0f};
    h.vertV  = {0.0f, 0.0f, 1.0f};
    return h;
}

inline heliostat_design MakeTestDesign()
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

}  // namespace

// The 3x3 grid is a superset of the cross pattern.
TEST(CpuIrradiance, RayPatterns_GridSupersetsCross) {
    field_layout field = MakeSmallField(2, 100, 0, 100);
    field.nHelios = 1;
    heliostat h = MakeTestHelio();
    field.helios = &h;
    heliostat_design design = MakeTestDesign();

    std::vector<float> cross(field.nVoxels, 0.0f);
    std::vector<float> grid(field.nVoxels, 0.0f);

    CPU_ComputeHeliostatIrradiance(cross.data(), field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.0f, 1.0f, 2.0f,
                                   0, 1, 0.0f, 1.0f);
    CPU_ComputeHeliostatIrradiance(grid.data(),  field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.0f, 1.0f, 2.0f,
                                   0, 0, 0.0f, 1.0f);

    EXPECT_TRUE(CountHitVoxels(grid) >= CountHitVoxels(cross));
    EXPECT_TRUE(CountHitVoxels(cross) > 0);
    return true;
}

// 3x3 facets give outerDir a non-zero offsetMag so outer rays actually widen.
TEST(CpuIrradiance, OuterRays_WidenFootprint) {
    field_layout field = MakeSmallField(2, 100, 0, 100);
    field.nHelios = 1;
    heliostat h = MakeTestHelio();
    field.helios = &h;

    heliostat_design design = {};
    design.nFacets = 9;
    design.nRows = 3;
    design.nCols = 3;
    design.facetWidth = 1.2f;
    design.facetHeight = 1.2f;
    design.facetsImported = false;

    std::vector<float> noOuter(field.nVoxels, 0.0f);
    std::vector<float> withOuter(field.nVoxels, 0.0f);

    CPU_ComputeHeliostatIrradiance(noOuter.data(), field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   0, 0, 0.0f, 1.0f);
    CPU_ComputeHeliostatIrradiance(withOuter.data(), field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   1, 0, 0.0f, 1.0f);

    EXPECT_TRUE(CountHitVoxels(withOuter) > CountHitVoxels(noOuter));
    return true;
}

// Atomic accumulation under OpenMP guarantees correctness, but per-voxel sum
// order can drift between runs — allow a tolerance rather than bit-exact match.
TEST(CpuIrradiance, RepeatedRuns_AreCloseEnough) {
    field_layout field = MakeSmallField(2, 100, 0, 100);
    field.nHelios = 8;
    heliostat helios[8];
    for (int k = 0; k < 8; ++k)
    {
        helios[k] = MakeTestHelio();
        helios[k].id = k;
        helios[k].loc = {20.0f + 5.0f * k, 20.0f + 5.0f * k, 0.0f};
    }
    field.helios = helios;
    heliostat_design design = MakeTestDesign();

    std::vector<float> a(field.nVoxels, 0.0f);
    std::vector<float> b(field.nVoxels, 0.0f);

    CPU_ComputeHeliostatIrradiance(a.data(), field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   1, 0, 1.0f, 1.0f);
    CPU_ComputeHeliostatIrradiance(b.data(), field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   1, 0, 1.0f, 1.0f);

    EXPECT_TRUE(CountHitVoxels(a) > 0);
    EXPECT_EQ(CountHitVoxels(a), CountHitVoxels(b));

    const float tol = 0.001f * MaxOf(a);
    for (size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_NEAR(a[i], b[i], tol);
    }
    return true;
}

// Helios within +/- 5 m of the tower are physically invalid (mirrors CUDA).
TEST(CpuIrradiance, NearTower_Skipped) {
    field_layout field = MakeSmallField(2, 100, 0, 100);
    field.nHelios = 1;
    heliostat h = MakeTestHelio();
    h.loc = {0.0f, 0.0f, 0.0f};
    field.helios = &h;
    heliostat_design design = MakeTestDesign();

    std::vector<float> out(field.nVoxels, 0.0f);
    CPU_ComputeHeliostatIrradiance(out.data(), field, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   1, 0, 0.0f, 1.0f);
    EXPECT_EQ(0, CountHitVoxels(out));
    return true;
}
