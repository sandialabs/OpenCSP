// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// Tests for voxel indexing functions

#include "test_framework.h"
#include "ffiam/ffiam.h"
#include "core/voxel.h"

// =============================================================================
// GetVoxelIndexFromLoc Tests
// =============================================================================

TEST(VoxelIndexing, IndexFromLoc_Origin) {
    // At origin (0,0,zmin), index depends on radius
    int3 loc = {0, 0, 4};  // zmin=4
    int idx = GetVoxelIndexFromLoc(loc, 2, 600, 4);

    // With radius=600, nPerSide = 1200/2 = 600
    // x = (0 + 600) / 2 = 300
    // y = (0 + 600) / 2 = 300
    // z = (4 - 4) / 2 = 0
    // idx = 0 * 600*600 + 300 * 600 + 300 = 180300
    EXPECT_EQ(180300, idx);
    return true;
}

TEST(VoxelIndexing, IndexFromLoc_Corner) {
    // At corner (-radius, -radius, zmin)
    int3 loc = {-600, -600, 4};
    int idx = GetVoxelIndexFromLoc(loc, 2, 600, 4);

    // x = (-600 + 600) / 2 = 0
    // y = (-600 + 600) / 2 = 0
    // z = (4 - 4) / 2 = 0
    // idx = 0
    EXPECT_EQ(0, idx);
    return true;
}

TEST(VoxelIndexing, IndexFromLoc_UpperCorner) {
    // At upper corner (near +radius, +radius, zmax)
    int3 loc = {598, 598, 100};
    int idx = GetVoxelIndexFromLoc(loc, 2, 600, 4);

    // x = (598 + 600) / 2 = 599
    // y = (598 + 600) / 2 = 599
    // z = (100 - 4) / 2 = 48
    // idx = 48 * 600*600 + 599 * 600 + 599
    int expected = 48 * 600 * 600 + 599 * 600 + 599;
    EXPECT_EQ(expected, idx);
    return true;
}

TEST(VoxelIndexing, IndexFromLoc_Float3Version) {
    float3 loc = {0.0f, 0.0f, 4.0f};
    int idx = GetVoxelIndexFromLoc(loc, 2, 600, 4);
    EXPECT_EQ(180300, idx);
    return true;
}

// =============================================================================
// GetVoxelLocFromIndex Tests
// =============================================================================

TEST(VoxelIndexing, LocFromIndex_Zero) {
    float3 loc = GetVoxelLocFromIndex(0, 2, 600, 4);

    // Index 0 should be at (-radius, -radius, zmin)
    EXPECT_NEAR(-600.0f, loc.x, 0.1f);
    EXPECT_NEAR(-600.0f, loc.y, 0.1f);
    EXPECT_NEAR(4.0f, loc.z, 0.1f);
    return true;
}

TEST(VoxelIndexing, LocFromIndex_Center) {
    // Index at center of first plane
    // nPerSide = 600, center = 300
    // idx = 0 * 600*600 + 300 * 600 + 300 = 180300
    float3 loc = GetVoxelLocFromIndex(180300, 2, 600, 4);

    EXPECT_NEAR(0.0f, loc.x, 2.0f);
    EXPECT_NEAR(0.0f, loc.y, 2.0f);
    EXPECT_NEAR(4.0f, loc.z, 0.1f);
    return true;
}

// =============================================================================
// ComputeVoxelGridDimensions Tests
// =============================================================================

TEST(VoxelIndexing, GridDimensions_Default) {
    int nX, nY, nZ;
    int total = ComputeVoxelGridDimensions(600, 4, 100, 2, &nX, &nY, &nZ);

    EXPECT_EQ(600, nX);
    EXPECT_EQ(600, nY);
    EXPECT_EQ(49, nZ);  // (100-4)/2 + 1 = 49
    EXPECT_EQ(600 * 600 * 49, total);
    return true;
}

TEST(VoxelIndexing, GridDimensions_LargerVoxel) {
    int nX, nY, nZ;
    int total = ComputeVoxelGridDimensions(600, 4, 100, 4, &nX, &nY, &nZ);

    EXPECT_EQ(300, nX);  // 1200 / 4 = 300
    EXPECT_EQ(300, nY);
    EXPECT_EQ(25, nZ);   // (100-4)/4 + 1 = 25
    EXPECT_EQ(300 * 300 * 25, total);
    return true;
}

TEST(VoxelIndexing, GridDimensions_LargerRadius) {
    int nX, nY, nZ;
    int total = ComputeVoxelGridDimensions(1600, 4, 300, 2, &nX, &nY, &nZ);

    EXPECT_EQ(1600, nX);  // 3200 / 2 = 1600
    EXPECT_EQ(1600, nY);
    EXPECT_EQ(149, nZ);   // (300-4)/2 + 1 = 149
    EXPECT_EQ(1600 * 1600 * 149, total);
    return true;
}

// =============================================================================
// Round-trip Tests
// =============================================================================

TEST(VoxelIndexing, RoundTrip_MultipleIndices) {
    // Test several indices to verify round-trip consistency
    int testIndices[] = {0, 100, 10000, 180300, 500000};

    for (int idx : testIndices) {
        float3 loc = GetVoxelLocFromIndex(idx, 2, 600, 4);
        int3 loc_int = {(int)loc.x, (int)loc.y, (int)loc.z};
        int recovered = GetVoxelIndexFromLoc(loc_int, 2, 600, 4);

        if (recovered != idx) {
            std::cout << "  Round-trip failed for index " << idx << std::endl;
            std::cout << "    Loc: (" << loc.x << ", " << loc.y << ", " << loc.z << ")" << std::endl;
            std::cout << "    Recovered: " << recovered << std::endl;
            return false;
        }
    }
    return true;
}
