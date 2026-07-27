// Unit tests for util/math.h and compute/ray_math.h primitives.

#include "test_framework.h"
#include "util/math.h"
#include "compute/ray_math.h"

#include <cmath>


TEST(MathPrimitives, RotateAroundAxis_180DegreesAroundZ)
{
    float3 v{1.0f, 0.0f, 0.0f};
    float3 axis{0.0f, 0.0f, 1.0f};
    float3 r = ffiam::RotateAroundAxis(v, axis, static_cast<float>(M_PI));
    EXPECT_NEAR(-1.0f, r.x, 1e-5f);
    EXPECT_NEAR(0.0f, r.y, 1e-5f);
    EXPECT_NEAR(0.0f, r.z, 1e-5f);
    return true;
}

TEST(MathPrimitives, RotateAroundAxis_NegativeAngleReverses)
{
    float3 v{1.0f, 0.0f, 0.0f};
    float3 axis{0.0f, 0.0f, 1.0f};
    float3 pos = ffiam::RotateAroundAxis(v, axis, static_cast<float>(M_PI) / 2.0f);
    float3 neg = ffiam::RotateAroundAxis(v, axis, -static_cast<float>(M_PI) / 2.0f);
    EXPECT_NEAR(0.0f, pos.x, 1e-5f);
    EXPECT_NEAR(1.0f, pos.y, 1e-5f);
    EXPECT_NEAR(0.0f, neg.x, 1e-5f);
    EXPECT_NEAR(-1.0f, neg.y, 1e-5f);
    return true;
}

TEST(MathPrimitives, RotateAroundAxis_120DegreesPreservesLength)
{
    float3 v{2.0f, 0.0f, 0.0f};
    float3 axis{0.0f, 0.0f, 1.0f};
    float3 r = ffiam::RotateAroundAxis(v, axis, 2.0f * static_cast<float>(M_PI) / 3.0f);
    EXPECT_NEAR(2.0f, length(r), 1e-4f);
    EXPECT_NEAR(-1.0f, r.x, 1e-4f);
    EXPECT_NEAR(std::sqrt(3.0f), r.y, 1e-4f);
    return true;
}

TEST(MathPrimitives, RotateAroundAxis_AroundVectorParallelToInput)
{
    float3 v{0.3f, 0.4f, 0.5f};
    float3 axis = v;
    float len = length(axis);
    axis = axis / len;
    float3 r = ffiam::RotateAroundAxis(v, axis, 1.234f);
    EXPECT_NEAR(v.x, r.x, 1e-5f);
    EXPECT_NEAR(v.y, r.y, 1e-5f);
    EXPECT_NEAR(v.z, r.z, 1e-5f);
    return true;
}

TEST(MathPrimitives, RotateAroundAxis_ArbitraryAxis_RoundTrip)
{
    float3 v{1.5f, -2.0f, 0.7f};
    float3 axis{0.6f, 0.8f, 0.0f};
    float angle = 0.7f;
    float3 fwd = ffiam::RotateAroundAxis(v, axis, angle);
    float3 back = ffiam::RotateAroundAxis(fwd, axis, -angle);
    EXPECT_NEAR(v.x, back.x, 1e-4f);
    EXPECT_NEAR(v.y, back.y, 1e-4f);
    EXPECT_NEAR(v.z, back.z, 1e-4f);
    return true;
}


TEST(MathPrimitives, GetBisector_EqualVectorsReturnsSame)
{
    float3 a{0.0f, 0.0f, 1.0f};
    float3 b{0.0f, 0.0f, 1.0f};
    float3 r = GetBisector(a, b);
    EXPECT_FLOAT3_NEAR(a, r, 1e-5f);
    EXPECT_NEAR(1.0f, length(r), 1e-5f);
    return true;
}

TEST(MathPrimitives, GetBisector_OrthogonalVectors)
{
    float3 a{1.0f, 0.0f, 0.0f};
    float3 b{0.0f, 0.0f, 1.0f};
    float3 r = GetBisector(a, b);
    const float k = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(k, r.x, 1e-5f);
    EXPECT_NEAR(0.0f, r.y, 1e-5f);
    EXPECT_NEAR(k, r.z, 1e-5f);
    EXPECT_NEAR(1.0f, length(r), 1e-5f);
    return true;
}

TEST(MathPrimitives, GetBisector_ResultIsNormalized)
{
    float3 a = ConvertAzElToCartesian(33.0f, 17.0f);
    float3 b = ConvertAzElToCartesian(115.0f, 42.0f);
    float3 r = GetBisector(a, b);
    EXPECT_NEAR(1.0f, length(r), 1e-4f);
    return true;
}

TEST(MathPrimitives, GetBisector_SymmetryInArgOrder)
{
    float3 a{0.5f, 0.2f, 0.84f};  Norm(&a);
    float3 b{-0.3f, 0.9f, 0.31f}; Norm(&b);
    float3 ab = GetBisector(a, b);
    float3 ba = GetBisector(b, a);
    EXPECT_FLOAT3_NEAR(ab, ba, 1e-5f);
    return true;
}


TEST(MathPrimitives, Norm_UnitLength)
{
    float3 v{3.0f, 4.0f, 0.0f};
    Norm(&v);
    EXPECT_NEAR(0.6f, v.x, 1e-5f);
    EXPECT_NEAR(0.8f, v.y, 1e-5f);
    EXPECT_NEAR(0.0f, v.z, 1e-5f);
    EXPECT_NEAR(1.0f, length(v), 1e-5f);
    return true;
}

TEST(MathPrimitives, Norm_AlreadyUnitIsIdempotent)
{
    float3 v{1.0f, 0.0f, 0.0f};
    Norm(&v);
    EXPECT_NEAR(1.0f, v.x, 1e-6f);
    EXPECT_NEAR(0.0f, v.y, 1e-6f);
    EXPECT_NEAR(0.0f, v.z, 1e-6f);
    return true;
}


TEST(MathPrimitives, DistToPlane_HorizontalPlane)
{
    float3 p{0.0f, 0.0f, 0.0f};
    float3 vec{0.0f, 0.0f, 1.0f};
    float3 planeP{0.0f, 0.0f, 10.0f};
    float3 planeN{0.0f, 0.0f, 1.0f};
    float d = ffiam::DistToPlaneHost(p, vec, planeP, planeN);
    EXPECT_NEAR(10.0f, d, 1e-5f);
    return true;
}

TEST(MathPrimitives, DistToPlane_AngledRay)
{
    float3 p{0.0f, 0.0f, 0.0f};
    float3 vec = {1.0f, 0.0f, 1.0f};
    Norm(&vec);
    float3 planeP{0.0f, 0.0f, 10.0f};
    float3 planeN{0.0f, 0.0f, 1.0f};
    float d = ffiam::DistToPlaneHost(p, vec, planeP, planeN);
    EXPECT_NEAR(10.0f * std::sqrt(2.0f), d, 1e-4f);
    return true;
}


TEST(MathPrimitives, AzEl_RoundTrip)
{
    struct Pair { float az, el; };
    const Pair cases[] = {{45.0f, 30.0f}, {135.0f, 60.0f}, {270.0f, 15.0f}, {350.0f, -5.0f}};
    for (auto c : cases) {
        float3 cart = ConvertAzElToCartesian(c.az, c.el);
        float2 ae = ConvertCartesianToAzEl(cart);
        EXPECT_NEAR(c.az, ae.x, 0.01f);
        EXPECT_NEAR(c.el, ae.y, 0.01f);
    }
    return true;
}

TEST(MathPrimitives, ConvertCartesianToAzEl_ZeroLengthReturnsZero)
{
    // Documented behavior: length < 1e-9 returns (0, 0) rather than NaN.
    float3 zero{0.0f, 0.0f, 0.0f};
    float2 ae = ConvertCartesianToAzEl(zero);
    EXPECT_NEAR(0.0f, ae.x, 1e-6f);
    EXPECT_NEAR(0.0f, ae.y, 1e-6f);
    return true;
}
