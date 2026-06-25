// Copyright Sandia National Laboratories. All rights reserved.
// Unit tests for aim strategies: Point, Ring, SplitRing, Vector, CsvData.
// FixedNormal is covered by test_aim_fixed_normal.cpp. aim_null asserts in
// ComputeHeliostatConfigurations and is not tested here.

#include "test_framework.h"
#include "ffiam/types.h"
#include "core/heliostat.h"
#include "util/helper_math.h"
#include "util/math.h"

#include <vector>

namespace {

struct FieldHarness {
    std::vector<heliostat> helios;
    std::vector<float3> locs;
    std::vector<float3> aimVs;
    std::vector<float> moveAngles;
    field_layout field{};

    FieldHarness(const std::vector<float3>& positions, float towerHeight = 60.0f)
        : helios(positions.size())
        , locs(positions)
        , aimVs(positions.size(), float3{0, 0, 1})
        , moveAngles(positions.size(), 0.0f)
    {
        for (size_t i = 0; i < positions.size(); ++i) {
            helios[i].id = static_cast<int>(i);
            helios[i].loc = positions[i];
        }
        field.lat = 35.0f;
        field.lng = -106.0f;
        field.timezone = -7.0f;
        field.radius = 600;
        field.zMin = 0;
        field.zMax = 100;
        field.towerPos = {0, 0, 0};
        field.towerHeight = towerHeight;
        field.helios = helios.data();
        field.locs = locs.data();
        field.aimVs = aimVs.data();
        field.moveAngles = moveAngles.data();
        field.voxelSize = 2;
        field.nVoxels = 0;
        field.voxelArea = 4.0f;
        field.nHelios = static_cast<int>(positions.size());
    }
};

inline float3 SunSE55() {
    return ConvertAzElToCartesian(114.5f, 55.0f);
}

} // namespace


TEST(AimStrategies, Point_ReceiverAimFromSouthHelio)
{
    FieldHarness h({{0.0f, -100.0f, 0.0f}}, 60.0f);
    aim_strategy aimStrat{aim_point, {0.0f, 0.0f, 60.0f}, ""};
    const float3 sunVec = SunSE55();

    ComputeHeliostatConfigurations(0, &h.field, &aimStrat, sunVec, 0, 1);

    const heliostat& he = h.helios[0];
    EXPECT_NEAR(1.0f, length(he.refV), 1e-4f);
    EXPECT_TRUE(he.refV.y > 0.5f);
    EXPECT_TRUE(he.refV.z > 0.4f);

    EXPECT_NEAR(0.0f, he.focalPt.x, 1e-3f);
    EXPECT_NEAR(0.0f, he.focalPt.y, 1e-3f);
    EXPECT_NEAR(60.0f, he.focalPt.z, 1e-3f);
    EXPECT_NEAR(std::sqrt(100.0f * 100.0f + 60.0f * 60.0f), he.focalLen, 1e-2f);

    EXPECT_NEAR(1.0f, length(he.aimV), 1e-4f);
    return true;
}

TEST(AimStrategies, Point_OffsetAimAbovereceiver)
{
    FieldHarness h({{50.0f, 0.0f, 0.0f}}, 60.0f);

    aim_strategy aimAtReceiver{aim_point, {0.0f, 0.0f, 60.0f}, ""};
    ComputeHeliostatConfigurations(0, &h.field, &aimAtReceiver, SunSE55(), 0, 1);
    const float receiverZ = h.helios[0].refV.z;

    aim_strategy aimAbove{aim_point, {0.0f, 0.0f, 90.0f}, ""};
    ComputeHeliostatConfigurations(0, &h.field, &aimAbove, SunSE55(), 0, 1);
    const float aboveZ = h.helios[0].refV.z;

    EXPECT_TRUE(aboveZ > receiverZ);
    return true;
}


TEST(AimStrategies, Ring_PointsPerpendicularToReceiverAxis)
{
    FieldHarness h({{0.0f, -100.0f, 0.0f}}, 60.0f);
    const float ringRadius = 20.0f;
    const float ringHeight = 60.0f;
    aim_strategy aimStrat{aim_ring, {ringRadius, ringHeight, 0.0f}, ""};

    ComputeHeliostatConfigurations(0, &h.field, &aimStrat, SunSE55(), 0, 1);
    const heliostat& he = h.helios[0];

    EXPECT_NEAR(1.0f, length(he.refV), 1e-4f);
    EXPECT_TRUE(he.refV.x < -0.1f);
    EXPECT_TRUE(he.refV.y > 0.7f);
    EXPECT_TRUE(he.refV.z > 0.3f);

    EXPECT_NEAR(ringHeight, he.focalPt.z, 5.0f);
    return true;
}

TEST(AimStrategies, Ring_LargerRadiusYieldsLargerOffset)
{
    FieldHarness h({{0.0f, -100.0f, 0.0f}}, 60.0f);

    aim_strategy small{aim_ring, {5.0f, 60.0f, 0.0f}, ""};
    ComputeHeliostatConfigurations(0, &h.field, &small, SunSE55(), 0, 1);
    const float smallOffsetX = std::abs(h.helios[0].refV.x);

    aim_strategy large{aim_ring, {50.0f, 60.0f, 0.0f}, ""};
    ComputeHeliostatConfigurations(0, &h.field, &large, SunSE55(), 0, 1);
    const float largeOffsetX = std::abs(h.helios[0].refV.x);

    EXPECT_TRUE(largeOffsetX > smallOffsetX);
    return true;
}


TEST(AimStrategies, SplitRing_EastAndWestMirror)
{
    FieldHarness h({
        {50.0f, -100.0f, 0.0f},
        {-50.0f, -100.0f, 0.0f},
    }, 60.0f);
    aim_strategy aimStrat{aim_split_ring, {25.0f, 60.0f, 0.0f}, ""};
    const float3 sunVec = SunSE55();

    ComputeHeliostatConfigurations(0, &h.field, &aimStrat, sunVec, 0, 2);
    ComputeHeliostatConfigurations(1, &h.field, &aimStrat, sunVec, 0, 2);

    const heliostat& east = h.helios[0];
    const heliostat& west = h.helios[1];

    EXPECT_NEAR(1.0f, length(east.refV), 1e-4f);
    EXPECT_NEAR(1.0f, length(west.refV), 1e-4f);
    EXPECT_TRUE(east.refV.y > 0.5f);
    EXPECT_TRUE(west.refV.y > 0.5f);

    EXPECT_TRUE((east.focalPt.x > 0.0f) != (west.focalPt.x > 0.0f) ||
                (std::abs(east.focalPt.x) < 1e-3f && std::abs(west.focalPt.x) < 1e-3f));
    return true;
}

TEST(AimStrategies, SplitRing_RowIdxShiftsAngle)
{
    // nHelios must be >= nRows: aim_split_ring computes nHelioPerSlice =
    // ceil(nHelios/nRows/2), and with nHelios=1, nRows=4 that goes to 0 and
    // `idx % 0` crashes.
    std::vector<float3> positions(4, float3{50.0f, -100.0f, 0.0f});
    FieldHarness h1(positions, 60.0f);
    FieldHarness h2(positions, 60.0f);
    aim_strategy aimStrat{aim_split_ring, {25.0f, 60.0f, 0.0f}, ""};
    const float3 sunVec = SunSE55();

    ComputeHeliostatConfigurations(0, &h1.field, &aimStrat, sunVec, 0, 4);
    ComputeHeliostatConfigurations(0, &h2.field, &aimStrat, sunVec, 3, 4);

    EXPECT_FALSE(ffiam::test::float3_near(h1.helios[0].refV, h2.helios[0].refV, 1e-3f));
    return true;
}


TEST(AimStrategies, Vector_SharedAimAcrossHeliostats)
{
    FieldHarness h({
        {50.0f, -100.0f, 0.0f},
        {-50.0f, -100.0f, 0.0f},
        {0.0f, -200.0f, 0.0f},
    }, 60.0f);

    float3 sharedDir = {0.5f, 0.5f, 0.707f};
    Norm(&sharedDir);
    aim_strategy aimStrat{aim_vector, sharedDir, ""};
    const float3 sunVec = SunSE55();

    for (int i = 0; i < h.field.nHelios; ++i) {
        ComputeHeliostatConfigurations(i, &h.field, &aimStrat, sunVec, 0, 1);
    }

    EXPECT_FLOAT3_NEAR(h.helios[0].refV, h.helios[1].refV, 1e-4f);
    EXPECT_FLOAT3_NEAR(h.helios[0].refV, h.helios[2].refV, 1e-4f);
    EXPECT_NEAR(1.0f, length(h.helios[0].refV), 1e-4f);

    EXPECT_FLOAT3_NEAR(sharedDir, h.helios[0].refV, 1e-3f);
    return true;
}


TEST(AimStrategies, CsvData_UsesPerHelioAimVector)
{
    FieldHarness h({
        {50.0f, -100.0f, 0.0f},
        {-50.0f, -100.0f, 0.0f},
    }, 60.0f);

    h.aimVs[0] = ConvertAzElToCartesian(0.0f, 50.0f);
    h.aimVs[1] = ConvertAzElToCartesian(45.0f, 40.0f);
    aim_strategy aimStrat{aim_data_csv, {0, 0, 0}, "ignored"};
    const float3 sunVec = SunSE55();

    ComputeHeliostatConfigurations(0, &h.field, &aimStrat, sunVec, 0, 1);
    ComputeHeliostatConfigurations(1, &h.field, &aimStrat, sunVec, 0, 1);

    EXPECT_FLOAT3_NEAR(h.aimVs[0], h.helios[0].refV, 1e-3f);
    EXPECT_FLOAT3_NEAR(h.aimVs[1], h.helios[1].refV, 1e-3f);
    EXPECT_NEAR(1.0f, length(h.helios[0].refV), 1e-4f);
    EXPECT_NEAR(1.0f, length(h.helios[1].refV), 1e-4f);

    EXPECT_FALSE(ffiam::test::float3_near(h.helios[0].refV, h.helios[1].refV, 1e-3f));
    return true;
}


TEST(AimStrategies, EnumValuesStable)
{
    EXPECT_EQ(0, static_cast<int>(aim_null));
    EXPECT_EQ(1, static_cast<int>(aim_point));
    EXPECT_EQ(2, static_cast<int>(aim_ring));
    EXPECT_EQ(3, static_cast<int>(aim_split_ring));
    EXPECT_EQ(4, static_cast<int>(aim_vector));
    EXPECT_EQ(5, static_cast<int>(aim_data_csv));
    EXPECT_EQ(6, static_cast<int>(aim_fixed_normal));
    return true;
}
