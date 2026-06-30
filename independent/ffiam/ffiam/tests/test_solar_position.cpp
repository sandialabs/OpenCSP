// Tests for solar position calculations

#include "test_framework.h"
#include "solar/solar_position.h"

// =============================================================================
// Solar Position Tests
// =============================================================================

// Test solar position at NSTTF on summer solstice at solar noon
TEST(SolarPosition, NSTTF_SummerSolstice_Noon) {
    // NSTTF coordinates: 34.96348, -106.50964
    // Summer solstice 2025, approximately solar noon (hour 12)
    SolarData data;
    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,  // lat, lng, timezone
        2025, 6, 21, 12,                 // year, month, day, hour
        0.1f,                            // base DNI
        &data
    );

    EXPECT_TRUE(result);

    // At summer solstice near solar noon, elevation should be high (70-80 deg)
    EXPECT_TRUE(data.elevation > 70.0f);
    EXPECT_TRUE(data.elevation < 85.0f);

    // Azimuth should be roughly south (around 180 deg, but varies with time)
    EXPECT_TRUE(data.azimuth > 90.0f);
    EXPECT_TRUE(data.azimuth < 270.0f);

    // Sun vector z-component should be positive (sun above horizon)
    EXPECT_TRUE(data.sunVector.z > 0.0f);

    // DNI should be scaled (close to base at noon)
    EXPECT_TRUE(data.scaledDni > 0.08f);
    EXPECT_TRUE(data.scaledDni <= 0.1f);

    return true;
}

// Test solar position at NSTTF in morning
TEST(SolarPosition, NSTTF_Morning) {
    SolarData data;
    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 6, 21, 9,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);

    // Morning: lower elevation than noon
    EXPECT_TRUE(data.elevation > 30.0f);
    EXPECT_TRUE(data.elevation < 60.0f);

    // Morning: azimuth should be in the east (< 180)
    EXPECT_TRUE(data.azimuth > 45.0f);
    EXPECT_TRUE(data.azimuth < 135.0f);

    // Morning DNI scaled lower than noon
    EXPECT_TRUE(data.scaledDni > 0.05f);
    EXPECT_TRUE(data.scaledDni < 0.1f);

    return true;
}

// Test solar position at NSTTF in evening
TEST(SolarPosition, NSTTF_Evening) {
    SolarData data;
    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 6, 21, 17,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);

    // Evening: lower elevation
    EXPECT_TRUE(data.elevation > 20.0f);
    EXPECT_TRUE(data.elevation < 55.0f);

    // Evening: azimuth should be in the west (> 180)
    EXPECT_TRUE(data.azimuth > 225.0f);
    EXPECT_TRUE(data.azimuth < 315.0f);

    return true;
}

// Test solar position at winter solstice (lower sun angle)
TEST(SolarPosition, NSTTF_WinterSolstice_Noon) {
    SolarData data;
    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 12, 21, 12,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);

    // Winter solstice: much lower elevation than summer
    EXPECT_TRUE(data.elevation > 25.0f);
    EXPECT_TRUE(data.elevation < 45.0f);

    return true;
}

// Test that sun below horizon returns false
TEST(SolarPosition, NightTime_ReturnsFalse) {
    SolarData data;
    // Midnight - sun should be below horizon
    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 6, 21, 0,
        0.1f,
        &data
    );

    // Should return false for nighttime
    EXPECT_FALSE(result);

    return true;
}

// Test different location (higher latitude)
TEST(SolarPosition, HighLatitude_LongerDays) {
    SolarData data;
    // Test at 55 degrees north (e.g., Scotland) summer solstice
    bool result = ComputeSolarPosition(
        55.0f, -4.0f, 0.0f,
        2025, 6, 21, 12,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);

    // Higher latitude = lower max elevation
    EXPECT_TRUE(data.elevation > 50.0f);
    EXPECT_TRUE(data.elevation < 65.0f);

    return true;
}

// Test sun vector normalization
TEST(SolarPosition, SunVector_IsNormalized) {
    SolarData data;
    ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 6, 21, 12,
        0.1f,
        &data
    );

    // Check that sun vector is approximately unit length
    float length = sqrtf(
        data.sunVector.x * data.sunVector.x +
        data.sunVector.y * data.sunVector.y +
        data.sunVector.z * data.sunVector.z
    );

    EXPECT_NEAR(1.0f, length, 0.01f);

    return true;
}

// Test sunrise/sunset times are reasonable
TEST(SolarPosition, SunriseSunset_Reasonable) {
    SolarData data;
    ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 6, 21, 12,
        0.1f,
        &data
    );

    // Summer solstice at ~35N latitude
    // Sunrise should be early (around 5-6 AM = 300-360 minutes)
    EXPECT_TRUE(data.sunrise > 250.0f);
    EXPECT_TRUE(data.sunrise < 400.0f);

    // Sunset should be late (around 8-9 PM = 1200-1260 minutes)
    EXPECT_TRUE(data.sunset > 1150.0f);
    EXPECT_TRUE(data.sunset < 1320.0f);

    // Sunset should be after sunrise
    EXPECT_TRUE(data.sunset > data.sunrise);

    return true;
}

// =============================================================================
// Fractional Hour Tests
// =============================================================================

// Test that fractional hours work (10.5 = 10:30 AM)
TEST(SolarPosition, FractionalHour_HalfHour) {
    SolarData data_integer, data_fractional;

    // Compare hour 10 vs hour 10.5
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.0f, 0.1f, &data_integer);
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.5f, 0.1f, &data_fractional);

    // At 10:30, sun should be higher and more southerly than at 10:00
    EXPECT_TRUE(data_fractional.elevation > data_integer.elevation);

    // Azimuth should have moved toward south (larger value)
    EXPECT_TRUE(data_fractional.azimuth > data_integer.azimuth);

    return true;
}

// Test specific flight time: 10:08 AM
TEST(SolarPosition, FractionalHour_Flight0001) {
    SolarData data;
    // 10:08 AM = 10 + 8/60 = 10.1333...
    float hour = 10.0f + 8.0f / 60.0f;

    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 8, 8, hour,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);

    // Sun should be above horizon in the morning
    EXPECT_TRUE(data.elevation > 40.0f);
    EXPECT_TRUE(data.elevation < 70.0f);

    // Morning azimuth should be east of south
    EXPECT_TRUE(data.azimuth > 90.0f);
    EXPECT_TRUE(data.azimuth < 180.0f);

    return true;
}

// Test specific flight time: 12:43 PM
TEST(SolarPosition, FractionalHour_Flight0004) {
    SolarData data;
    // 12:43 PM = 12 + 43/60 = 12.7166...
    float hour = 12.0f + 43.0f / 60.0f;

    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 8, 8, hour,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);

    // Near solar noon, elevation should be high
    EXPECT_TRUE(data.elevation > 65.0f);
    EXPECT_TRUE(data.elevation < 80.0f);

    // Azimuth should be close to south (180)
    EXPECT_TRUE(data.azimuth > 150.0f);
    EXPECT_TRUE(data.azimuth < 210.0f);

    return true;
}

// Test that fractional hours produce different results than truncated hours
TEST(SolarPosition, FractionalHour_DifferentFromTruncated) {
    SolarData data_truncated, data_fractional;

    // 10:30 should be different from 10:00
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.0f, 0.1f, &data_truncated);
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.5f, 0.1f, &data_fractional);

    // Elevations should differ by several degrees over 30 minutes
    float elevation_diff = std::abs(data_fractional.elevation - data_truncated.elevation);
    EXPECT_TRUE(elevation_diff > 2.0f);  // At least 2 degrees difference

    // Azimuths should also differ
    float azimuth_diff = std::abs(data_fractional.azimuth - data_truncated.azimuth);
    EXPECT_TRUE(azimuth_diff > 3.0f);  // At least 3 degrees difference

    return true;
}

// Test quarter hour precision
TEST(SolarPosition, FractionalHour_QuarterHour) {
    SolarData data_00, data_15, data_30, data_45;

    // Test 10:00, 10:15, 10:30, 10:45
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.0f, 0.1f, &data_00);
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.25f, 0.1f, &data_15);
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.5f, 0.1f, &data_30);
    ComputeSolarPosition(34.96348f, -106.50964f, -7.0f, 2025, 8, 8, 10.75f, 0.1f, &data_45);

    // Elevations should increase monotonically through the morning
    EXPECT_TRUE(data_15.elevation > data_00.elevation);
    EXPECT_TRUE(data_30.elevation > data_15.elevation);
    EXPECT_TRUE(data_45.elevation > data_30.elevation);

    // Azimuths should increase (moving toward south)
    EXPECT_TRUE(data_15.azimuth > data_00.azimuth);
    EXPECT_TRUE(data_30.azimuth > data_15.azimuth);
    EXPECT_TRUE(data_45.azimuth > data_30.azimuth);

    return true;
}

// Test backward compatibility: integer-like floats should work same as before
TEST(SolarPosition, FractionalHour_IntegerCompatibility) {
    SolarData data;

    // 12.0 should work the same as the old integer 12
    bool result = ComputeSolarPosition(
        34.96348f, -106.50964f, -7.0f,
        2025, 6, 21, 12.0f,
        0.1f,
        &data
    );

    EXPECT_TRUE(result);
    EXPECT_TRUE(data.elevation > 70.0f);
    EXPECT_TRUE(data.elevation < 85.0f);

    return true;
}
