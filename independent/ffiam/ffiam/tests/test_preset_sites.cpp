// Tests for preset site configurations

#include "test_framework.h"
#include "api/preset_sites.h"
#include "ffiam/types.h"

// =============================================================================
// Preset Site Configuration Tests
// =============================================================================

// Test NSTTF configuration
TEST(PresetSites, NSTTF_Configuration) {
    preset_site_config config;
    std::string aimFile;

    bool result = get_preset_site_config(site_nsttf, aim_point, &config, &aimFile);

    EXPECT_TRUE(result);

    // NSTTF is at Sandia National Labs, New Mexico
    EXPECT_NEAR(34.96348f, config.coords.x, 0.01f);   // latitude
    EXPECT_NEAR(-106.50964f, config.coords.y, 0.01f); // longitude
    EXPECT_NEAR(-7.0f, config.timezone, 0.1f);        // Mountain Time

    // NSTTF has 218 heliostats
    EXPECT_EQ(218, config.nHelios);

    // Tower height around 61.5m
    EXPECT_NEAR(61.5f, config.towerHeight, 1.0f);

    // 25 facets per heliostat (5x5 grid)
    EXPECT_EQ(25, config.nFacets);
    EXPECT_EQ(5, config.nFacetCols);

    // Heliostat file should be non-empty for NSTTF
    EXPECT_TRUE(!config.helioFile.empty());

    return true;
}

// Test SampleV1 configuration
TEST(PresetSites, SampleV1_Configuration) {
    preset_site_config config;
    std::string aimFile;

    bool result = get_preset_site_config(site_sampleV1, aim_point, &config, &aimFile);

    EXPECT_TRUE(result);

    // SampleV1 has approximately 1936 heliostats (generic field)
    EXPECT_TRUE(config.nHelios >= 1000);
    EXPECT_TRUE(config.nHelios <= 2500);

    // Field radius should be around 650m
    EXPECT_TRUE(config.radius >= 600);
    EXPECT_TRUE(config.radius <= 700);

    // Tower height (inherits default of 61m if not explicitly set)
    EXPECT_TRUE(config.towerHeight > 50.0f);
    EXPECT_TRUE(config.towerHeight < 150.0f);

    return true;
}

// Test that site_null defaults to NSTTF
TEST(PresetSites, NullSite_DefaultsToNSTTF) {
    preset_site_config config;
    std::string aimFile;

    // site_null should default to NSTTF and return true
    bool result = get_preset_site_config(site_null, aim_point, &config, &aimFile);

    EXPECT_TRUE(result);

    // Should have NSTTF configuration (218 heliostats)
    EXPECT_EQ(218, config.nHelios);

    return true;
}

// Test truly invalid site ID returns false
TEST(PresetSites, InvalidSiteId_ReturnsFalse) {
    preset_site_config config;
    std::string aimFile;

    // Use a site ID that doesn't exist (e.g., 999)
    bool result = get_preset_site_config(999, aim_point, &config, &aimFile);

    EXPECT_FALSE(result);

    return true;
}

// Test version parameters
TEST(PresetSites, VersionParams_V1) {
    int radius, zMin, zMax, voxelSize;

    get_version_params(1, &radius, &zMin, &zMax, &voxelSize);

    // V1: 600m radius, 4-100m altitude, 2m voxels
    EXPECT_EQ(600, radius);
    EXPECT_EQ(4, zMin);
    EXPECT_TRUE(zMax >= 100);
    EXPECT_EQ(2, voxelSize);

    return true;
}

// Test version parameters V2
TEST(PresetSites, VersionParams_V2) {
    int radius, zMin, zMax, voxelSize;

    get_version_params(2, &radius, &zMin, &zMax, &voxelSize);

    // V2: expanded capacity
    EXPECT_TRUE(radius >= 1000);
    EXPECT_TRUE(zMax >= 200);

    return true;
}

// Test version parameters V3
TEST(PresetSites, VersionParams_V3) {
    int radius, zMin, zMax, voxelSize;

    get_version_params(3, &radius, &zMin, &zMax, &voxelSize);

    // V3: largest capacity
    EXPECT_TRUE(radius >= 1600);
    EXPECT_TRUE(zMax >= 300);

    return true;
}

// Generic radial site configuration
TEST(PresetSites, Radial_Configuration) {
    preset_site_config config;
    std::string aimFile;

    bool result = get_preset_site_config(site_radial, aim_point, &config, &aimFile);

    EXPECT_TRUE(result);

    // Large radial field: ~10k heliostats, 1.6 km radius.
    EXPECT_TRUE(config.nHelios >= 10000);
    EXPECT_TRUE(config.radius >= 1600);

    // Generated layout: no helio/facet CSVs, radial generation flag set.
    EXPECT_TRUE(config.helioFile.empty());
    EXPECT_TRUE(config.facetFile.empty());
    EXPECT_EQ(layout_radial, config.layout);

    return true;
}

// Small radial site (~1 km, ~6.4k heliostats).
TEST(PresetSites, RadialSmall_Configuration) {
    preset_site_config config;
    std::string aimFile;

    bool result = get_preset_site_config(site_radialSmall, aim_point, &config, &aimFile);

    EXPECT_TRUE(result);
    EXPECT_TRUE(config.nHelios >= 6000);
    EXPECT_EQ(1000, config.radius);
    EXPECT_TRUE(config.helioFile.empty());
    EXPECT_EQ(layout_radial, config.layout);

    return true;
}

// Test that all aim strategies are accepted for NSTTF
TEST(PresetSites, NSTTF_AllAimStrategies) {
    preset_site_config config;
    std::string aimFile;

    // Point aim
    EXPECT_TRUE(get_preset_site_config(site_nsttf, aim_point, &config, &aimFile));

    // Ring aim
    EXPECT_TRUE(get_preset_site_config(site_nsttf, aim_ring, &config, &aimFile));

    // Vector aim
    EXPECT_TRUE(get_preset_site_config(site_nsttf, aim_vector, &config, &aimFile));

    return true;
}

// Test facet dimensions are physically reasonable
TEST(PresetSites, FacetDimensions_Reasonable) {
    preset_site_config config;
    std::string aimFile;

    get_preset_site_config(site_nsttf, aim_point, &config, &aimFile);

    // Facets should be between 0.5m and 3m
    EXPECT_TRUE(config.facetDims.x > 0.5f);
    EXPECT_TRUE(config.facetDims.x < 3.0f);
    EXPECT_TRUE(config.facetDims.y > 0.5f);
    EXPECT_TRUE(config.facetDims.y < 3.0f);

    return true;
}
