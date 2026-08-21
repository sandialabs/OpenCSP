// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "cuda_utils.h"
#include "helper_math.h"  // For float3 operators (dot, length, +, -, *, /)

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef PI
#define PI M_PI
#endif

#ifndef HALF_PI
#define HALF_PI M_PI / 2.f
#endif

#ifndef TWO_PI
#define TWO_PI M_PI * 2.f
#endif

#ifndef TO_DEG
#define TO_DEG 180.0f / PI
#endif

#ifndef TO_RAD
#define TO_RAD PI / 180.0f
#endif


// Overload the << operator for easy printing of float3
inline std::ostream& operator<<(std::ostream& os, const float3& v)
{
    os << "Vector(x=" << v.x << ", y=" << v.y << ", z=" << v.z << ")";
    return os;
}


inline void Norm(float3* a)
{
    float dist = length(*a);
    *a /= dist;
}

// Converts azimuth/elevation [deg] to a Cartesian unit vector.
// Coordinate system: +X east, +Y north, +Z up. Azimuth CW from north.
inline float3 ConvertAzElToCartesian(float azimuthDegrees, float elevationDegrees)
{
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system
    // Using double for precision during conversion.
    constexpr double DEGREES_TO_RADIANS = PI / 180.0;

    double az_rad = azimuthDegrees * DEGREES_TO_RADIANS;
    double el_rad = elevationDegrees * DEGREES_TO_RADIANS;

    double x = cos(el_rad) * sin(az_rad);
    double y = cos(el_rad) * cos(az_rad);
    double z = sin(el_rad);

    float3 result = {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z)
    };

    return result;
}


// Converts a Cartesian unit vector to azimuth/elevation [deg].
// Azimuth returned in [0, 360]; elevation in [-90, 90].
inline float2 ConvertCartesianToAzEl(const float3& vec)
{
    // Define constants for conversion. Using double for precision.
    constexpr double RADIANS_TO_DEGREES = 180.0 / PI;
    double length = std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);

    if (length < 1e-9)
    {
        return {0.0f, 0.0f};
    }

    double el_rad = asin(vec.z / length);
    double az_rad = atan2(vec.x, vec.y);
    float elevationDegrees = static_cast<float>(el_rad * RADIANS_TO_DEGREES);
    float azimuthDegrees = static_cast<float>(az_rad * RADIANS_TO_DEGREES);

    // The result of atan2 is in [-180, 180]. Adjust to [0, 360].
    if (azimuthDegrees < 0.0f)
    {
        azimuthDegrees += 360.0f;
    }

    return {azimuthDegrees, elevationDegrees};
}


// Returns the normalized bisector of two vectors.
inline float3 GetBisector(float3 vm, float3 vs)
{
    float3 vd = vm + vs;
    float dlen = length(vd);
    float3 result = vd / dlen;
    return result;
}
