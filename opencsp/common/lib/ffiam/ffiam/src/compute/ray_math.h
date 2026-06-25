// Copyright Sandia National Laboratories. All rights reserved.
// Host-callable ray-math helpers shared by the CPU and CUDA irradiance backends.

#pragma once

#include "util/cuda_utils.h"
#include "util/helper_math.h"
#include <cmath>

namespace ffiam {

// Rodrigues' rotation formula. axis must be normalized; angle in radians.
inline float3 RotateAroundAxis(float3 v, float3 axis, float angle)
{
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);
    return v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
}

// Distance along `vec` from `p` to the intersection with the plane (planeP, planeN).
inline float DistToPlaneHost(float3 p, float3 vec, float3 planeP, float3 planeN)
{
    return dot((planeP - p), planeN) / dot(vec, planeN);
}

// 1D voxel-array index from a world-space float position. Mirrors Cuda_GetVoxelIndexFromLoc.
inline int GetVoxelIndexFromLocCPU(const float3 loc, int size, int radius, int zmin)
{
    const int nPerSide = (radius * 2) / size;
    const int nPerPlane = nPerSide * nPerSide;
    const int z = static_cast<int>((loc.z - static_cast<float>(zmin)) / static_cast<float>(size));
    const int y = static_cast<int>((loc.y + static_cast<float>(radius)) / static_cast<float>(size));
    const int x = static_cast<int>((loc.x + static_cast<float>(radius)) / static_cast<float>(size));
    return z * nPerPlane + y * nPerSide + x;
}

}  // namespace ffiam
