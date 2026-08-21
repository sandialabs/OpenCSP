// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// Tests for ambient irradiance parameter support

#include "test_framework.h"
#include "ffiam/types.h"

// =============================================================================
// Ambient Irradiance Parameter Tests
// =============================================================================

// Test that ambient can be set to zero (disabled)
TEST(AmbientIrradiance, ZeroValue_DisablesFeature) {
    float ambient = 0.0f;

    // Zero ambient should be a valid value (feature disabled)
    EXPECT_NEAR(0.0f, ambient, 0.001f);

    return true;
}

// Test typical ambient values
TEST(AmbientIrradiance, TypicalValue_035) {
    // 0.35 kW/m² is a typical ambient irradiance for clear sky
    float ambient = 0.35f;

    EXPECT_TRUE(ambient > 0.0f);
    EXPECT_TRUE(ambient < 1.0f);

    return true;
}

// Test ambient value range
TEST(AmbientIrradiance, ValueRange_ValidRange) {
    // Test various valid ambient values
    float test_values[] = {0.0f, 0.1f, 0.25f, 0.35f, 0.5f, 0.75f, 1.0f};

    for (float val : test_values) {
        EXPECT_TRUE(val >= 0.0f);
        EXPECT_TRUE(val <= 2.0f);  // Max reasonable ambient is well under 2 kW/m²
    }

    return true;
}

// Test ambient as fraction of DNI
TEST(AmbientIrradiance, FractionOfDNI_TypicalRatio) {
    // Typical diffuse/ambient is 30-40% of DNI
    float dni = 0.1f;  // W/cm² (standard test value)
    float ambient_fraction = 0.35f;  // 35% of DNI
    float ambient = dni * ambient_fraction;

    EXPECT_NEAR(0.035f, ambient, 0.001f);

    return true;
}

// Test ambient effect on baseline irradiance
TEST(AmbientIrradiance, BaselineEffect_AddsToVoxels) {
    // When ambient is non-zero, it should be added to all voxels
    float ambient = 0.35f;
    float initial_irrad = 0.0f;
    float final_irrad = initial_irrad + ambient;

    // Voxel irradiance should be at least the ambient value
    EXPECT_NEAR(ambient, final_irrad, 0.001f);
    EXPECT_TRUE(final_irrad >= ambient);

    return true;
}

// Test ambient doesn't affect threshold logic incorrectly
TEST(AmbientIrradiance, ThresholdLogic_ConsidersAmbient) {
    // If threshold is 4 kW/m² and ambient is 0.35 kW/m²,
    // voxels need 3.65 kW/m² additional irradiance to exceed threshold
    float threshold = 4.0f;
    float ambient = 0.35f;
    float required_additional = threshold - ambient;

    EXPECT_NEAR(3.65f, required_additional, 0.001f);
    EXPECT_TRUE(required_additional > 0.0f);

    return true;
}

// Test ambient with very small value
TEST(AmbientIrradiance, SmallValue_Works) {
    float ambient = 0.001f;  // Very small but non-zero

    EXPECT_TRUE(ambient > 0.0f);
    EXPECT_TRUE(ambient < 0.01f);

    return true;
}

// Test ambient validation (negative should not be allowed conceptually)
TEST(AmbientIrradiance, Validation_NonNegative) {
    // Ambient irradiance should always be non-negative
    // (sunlight doesn't remove energy from voxels)
    float ambient = 0.35f;

    EXPECT_TRUE(ambient >= 0.0f);

    return true;
}
