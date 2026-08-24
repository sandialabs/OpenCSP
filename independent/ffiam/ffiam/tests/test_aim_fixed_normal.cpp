// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// Tests for aim_fixed_normal aim strategy

#include "test_framework.h"
#include "ffiam/types.h"
#include "util/helper_math.h"  // For float3 operators and dot/length functions
#include "util/math.h"

// =============================================================================
// Reflection Direction Tests
// Tests the reflection formula: R = 2*(N·S)*N - S
// =============================================================================

// Helper function to compute reflected direction (same as in heliostat.cpp)
inline float3 computeReflectedDirection(const float3& normal, const float3& sunVec) {
    float3 aimV = normal;
    Norm(&aimV);

    float NdotS = dot(aimV, sunVec);
    float3 refV = aimV * (2.0f * NdotS) - sunVec;
    Norm(&refV);
    return refV;
}

TEST(AimFixedNormal, FaceUp_SunAtZenith) {
    // Face-up heliostat (normal pointing straight up)
    float3 normal = {0.0f, 0.0f, 1.0f};
    // Sun at zenith (directly above)
    float3 sunVec = {0.0f, 0.0f, 1.0f};

    float3 reflected = computeReflectedDirection(normal, sunVec);

    // Reflected should also be straight up
    float3 expected = {0.0f, 0.0f, 1.0f};
    EXPECT_FLOAT3_NEAR(expected, reflected, 0.01f);
    return true;
}

TEST(AimFixedNormal, FaceUp_SunFromEast) {
    // Face-up heliostat
    float3 normal = {0.0f, 0.0f, 1.0f};
    // Sun from East at 45 deg elevation (Az=90, El=45)
    // sunVec = (cos(45)*sin(90), cos(45)*cos(90), sin(45)) = (0.707, 0, 0.707)
    float3 sunVec = {0.707f, 0.0f, 0.707f};
    Norm(&sunVec);

    float3 reflected = computeReflectedDirection(normal, sunVec);

    // For face-up, R = 2*(0,0,1)·S*(0,0,1) - S = 2*Sz*(0,0,1) - S
    // R = (0, 0, 2*0.707) - (0.707, 0, 0.707) = (-0.707, 0, 0.707)
    // Reflected should point West and Up
    EXPECT_TRUE(reflected.x < -0.5f);  // West component
    EXPECT_NEAR(0.0f, reflected.y, 0.1f);  // No North/South
    EXPECT_TRUE(reflected.z > 0.5f);  // Up component
    return true;
}

TEST(AimFixedNormal, FaceUp_SunFromSoutheast) {
    // Face-up heliostat
    float3 normal = {0.0f, 0.0f, 1.0f};
    // Sun from Southeast at 55 deg elevation (approximately Az=114.5)
    // This mimics the NSTTF STOW validation conditions
    float3 sunVec = ConvertAzElToCartesian(114.5f, 55.0f);

    float3 reflected = computeReflectedDirection(normal, sunVec);

    // Reflected should point Northwest and Up
    // From beam_geometry_analysis: expected ~(-0.52, 0.24, 0.82)
    EXPECT_TRUE(reflected.x < -0.4f);  // West component
    EXPECT_TRUE(reflected.y > 0.1f);   // North component
    EXPECT_TRUE(reflected.z > 0.7f);   // Up component (dominant)

    // Check the magnitude of the tilt
    float tilt_from_vertical = std::acos(reflected.z) * 180.0f / 3.14159f;
    EXPECT_TRUE(tilt_from_vertical > 30.0f && tilt_from_vertical < 40.0f);  // ~35 deg
    return true;
}

TEST(AimFixedNormal, FaceUp_SunFromNorth) {
    // Face-up heliostat
    float3 normal = {0.0f, 0.0f, 1.0f};
    // Sun from North at 30 deg elevation
    float3 sunVec = ConvertAzElToCartesian(0.0f, 30.0f);

    float3 reflected = computeReflectedDirection(normal, sunVec);

    // Reflected should point South and Up
    // For face-up with sun at 30 deg elevation from North:
    // sunVec = (0, 0.866, 0.5), N.S = 0.5
    // R = 2*0.5*(0,0,1) - (0, 0.866, 0.5) = (0, -0.866, 0.5)
    EXPECT_NEAR(0.0f, reflected.x, 0.1f);    // No East/West
    EXPECT_TRUE(reflected.y < -0.7f);        // Strong South component
    EXPECT_TRUE(reflected.z >= 0.45f);       // Up component (less dominant)
    return true;
}

// =============================================================================
// Horizontal Vector Tests for Vertical Normal
// When aimV = (0,0,1), the standard horizV = {aimV.y, -aimV.x, 0} = (0,0,0)
// The fix should handle this edge case
// =============================================================================

TEST(AimFixedNormal, HorizV_VerticalNormal) {
    // When aimV = (0,0,1), horizV should fallback to (1,0,0)
    float3 aimV = {0.0f, 0.0f, 1.0f};

    // Standard calculation would give zero vector
    float3 horizV = {aimV.y, -aimV.x, 0.0f};
    float horizLen = length(horizV);

    // The fix should detect this and use a fallback
    if (horizLen < 0.001f) {
        horizV = {1.0f, 0.0f, 0.0f};
    } else {
        horizV = horizV / horizLen;
    }

    // Should be East direction
    float3 expected = {1.0f, 0.0f, 0.0f};
    EXPECT_FLOAT3_NEAR(expected, horizV, 0.01f);
    return true;
}

TEST(AimFixedNormal, HorizV_TiltedNormal) {
    // For a tilted normal, standard calculation should work
    float3 aimV = {0.0f, 0.5f, 0.866f};  // 30 deg tilt toward North
    Norm(&aimV);

    float3 horizV = {aimV.y, -aimV.x, 0.0f};
    float horizLen = length(horizV);

    EXPECT_TRUE(horizLen > 0.001f);  // Should not be near zero

    horizV = horizV / horizLen;

    // horizV should point East (positive x)
    EXPECT_TRUE(horizV.x > 0.9f);
    EXPECT_NEAR(0.0f, horizV.y, 0.1f);
    EXPECT_NEAR(0.0f, horizV.z, 0.01f);
    return true;
}

// =============================================================================
// Aim Type Enum Test
// =============================================================================

TEST(AimFixedNormal, EnumValue) {
    EXPECT_EQ(6, static_cast<int>(aim_fixed_normal));
    return true;
}
