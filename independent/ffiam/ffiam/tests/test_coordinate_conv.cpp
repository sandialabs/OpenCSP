// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// Tests for coordinate conversion functions

#include "test_framework.h"
// Note: util/math.h is included via test_framework.h -> util/cuda_utils.h
#include "util/math.h"  // for ConvertAzElToCartesian, ConvertCartesianToAzEl
#include "compute/ray_math.h"  // for RotateAroundAxis, GetVoxelIndexFromLocCPU

// =============================================================================
// AzEl to Cartesian Tests
// =============================================================================

TEST(CoordinateConv, AzElToCartesian_North) {
    float3 result = ConvertAzElToCartesian(0.0f, 0.0f);
    float3 expected = {0.0f, 1.0f, 0.0f};
    EXPECT_FLOAT3_NEAR(expected, result, 0.01f);
    return true;
}

TEST(CoordinateConv, AzElToCartesian_East) {
    float3 result = ConvertAzElToCartesian(90.0f, 0.0f);
    float3 expected = {1.0f, 0.0f, 0.0f};
    EXPECT_FLOAT3_NEAR(expected, result, 0.01f);
    return true;
}

TEST(CoordinateConv, AzElToCartesian_South) {
    float3 result = ConvertAzElToCartesian(180.0f, 0.0f);
    float3 expected = {0.0f, -1.0f, 0.0f};
    EXPECT_FLOAT3_NEAR(expected, result, 0.01f);
    return true;
}

TEST(CoordinateConv, AzElToCartesian_West) {
    float3 result = ConvertAzElToCartesian(270.0f, 0.0f);
    float3 expected = {-1.0f, 0.0f, 0.0f};
    EXPECT_FLOAT3_NEAR(expected, result, 0.01f);
    return true;
}

TEST(CoordinateConv, AzElToCartesian_Zenith) {
    float3 result = ConvertAzElToCartesian(0.0f, 90.0f);
    float3 expected = {0.0f, 0.0f, 1.0f};
    EXPECT_FLOAT3_NEAR(expected, result, 0.01f);
    return true;
}

TEST(CoordinateConv, AzElToCartesian_Nadir) {
    float3 result = ConvertAzElToCartesian(0.0f, -90.0f);
    float3 expected = {0.0f, 0.0f, -1.0f};
    EXPECT_FLOAT3_NEAR(expected, result, 0.01f);
    return true;
}

TEST(CoordinateConv, AzElToCartesian_NortheastUp) {
    // Az=45, El=30 should give roughly (0.61, 0.61, 0.5)
    float3 result = ConvertAzElToCartesian(45.0f, 30.0f);
    EXPECT_NEAR(0.612f, result.x, 0.02f);
    EXPECT_NEAR(0.612f, result.y, 0.02f);
    EXPECT_NEAR(0.5f, result.z, 0.02f);
    return true;
}

// =============================================================================
// Cartesian to AzEl Tests
// =============================================================================

TEST(CoordinateConv, CartesianToAzEl_North) {
    float3 input = {0.0f, 1.0f, 0.0f};
    float2 result = ConvertCartesianToAzEl(input);
    EXPECT_NEAR(0.0f, result.x, 0.1f);   // azimuth
    EXPECT_NEAR(0.0f, result.y, 0.1f);   // elevation
    return true;
}

TEST(CoordinateConv, CartesianToAzEl_East) {
    float3 input = {1.0f, 0.0f, 0.0f};
    float2 result = ConvertCartesianToAzEl(input);
    EXPECT_NEAR(90.0f, result.x, 0.1f);  // azimuth
    EXPECT_NEAR(0.0f, result.y, 0.1f);   // elevation
    return true;
}

TEST(CoordinateConv, CartesianToAzEl_Zenith) {
    float3 input = {0.0f, 0.0f, 1.0f};
    float2 result = ConvertCartesianToAzEl(input);
    EXPECT_NEAR(90.0f, result.y, 0.1f);  // elevation should be 90
    return true;
}

// =============================================================================
// Round-trip Tests
// =============================================================================

TEST(CoordinateConv, RoundTrip_Az45El30) {
    float az_in = 45.0f;
    float el_in = 30.0f;

    float3 cartesian = ConvertAzElToCartesian(az_in, el_in);
    float2 azel_out = ConvertCartesianToAzEl(cartesian);

    EXPECT_NEAR(az_in, azel_out.x, 0.1f);
    EXPECT_NEAR(el_in, azel_out.y, 0.1f);
    return true;
}

TEST(CoordinateConv, RoundTrip_Az135El60) {
    float az_in = 135.0f;
    float el_in = 60.0f;

    float3 cartesian = ConvertAzElToCartesian(az_in, el_in);
    float2 azel_out = ConvertCartesianToAzEl(cartesian);

    EXPECT_NEAR(az_in, azel_out.x, 0.1f);
    EXPECT_NEAR(el_in, azel_out.y, 0.1f);
    return true;
}

TEST(CoordinateConv, RoundTrip_Az270El15) {
    float az_in = 270.0f;
    float el_in = 15.0f;

    float3 cartesian = ConvertAzElToCartesian(az_in, el_in);
    float2 azel_out = ConvertCartesianToAzEl(cartesian);

    EXPECT_NEAR(az_in, azel_out.x, 0.1f);
    EXPECT_NEAR(el_in, azel_out.y, 0.1f);
    return true;
}

// =============================================================================
// Shared ray-math helpers (compute/ray_math.h)
// =============================================================================

TEST(RayMath, RotateAroundAxis_QuarterTurnAroundZ) {
    float3 v = {1.0f, 0.0f, 0.0f};
    float3 axis = {0.0f, 0.0f, 1.0f};
    float angle = 3.14159265f / 2.0f;  // 90 deg
    float3 result = ffiam::RotateAroundAxis(v, axis, angle);
    EXPECT_NEAR(0.0f, result.x, 1e-5f);
    EXPECT_NEAR(1.0f, result.y, 1e-5f);
    EXPECT_NEAR(0.0f, result.z, 1e-5f);
    return true;
}

TEST(RayMath, RotateAroundAxis_NoRotationAtZeroAngle) {
    float3 v = {2.0f, -3.0f, 5.0f};
    float3 axis = {0.0f, 1.0f, 0.0f};
    float3 result = ffiam::RotateAroundAxis(v, axis, 0.0f);
    EXPECT_NEAR(v.x, result.x, 1e-5f);
    EXPECT_NEAR(v.y, result.y, 1e-5f);
    EXPECT_NEAR(v.z, result.z, 1e-5f);
    return true;
}

TEST(RayMath, GetVoxelIndexFromLocCPU_OriginPlaneCenter) {
    // size=2, radius=600, zmin=0: nPerSide=600, nPerPlane=360000.
    // For loc=(0.5, 0.5, 1.5): z-bucket=0, y-bucket=300, x-bucket=300.
    float3 loc = {0.5f, 0.5f, 1.5f};
    int idx = ffiam::GetVoxelIndexFromLocCPU(loc, 2, 600, 0);
    int expected = 300 * 600 + 300;  // z=0 plane center
    EXPECT_EQ(expected, idx);
    return true;
}

TEST(RayMath, GetVoxelIndexFromLocCPU_NonZeroZmin) {
    // size=2, radius=100, zmin=10: voxel at (-99.5, -99.5, 11) -> (0, 0, 0)
    float3 loc = {-99.5f, -99.5f, 11.0f};
    int idx = ffiam::GetVoxelIndexFromLocCPU(loc, 2, 100, 10);
    EXPECT_EQ(0, idx);
    return true;
}
