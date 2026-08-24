// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#include "core/voxel.h"

// Converts a 1D voxel array index to its world-space corner position.
float3 GetVoxelLocFromIndex(const int i, const int size, const int radius, const int zmin)
{
    int nPerSide = (radius * 2) / size;
    int nPerPlane = nPerSide * nPerSide;

    int3 val = {};

    val.z = i / nPerPlane;
    int rem = i - val.z * nPerPlane;

    val.y = rem / nPerSide;
    val.x = rem - val.y * nPerSide;

    // shift so origin at center, and account for voxel size (m)
    float3 result = {0, 0, 0};
    result.x = static_cast<float>(val.x * size) - static_cast<float>(radius);
    result.y = static_cast<float>(val.y * size) - static_cast<float>(radius);
    result.z = static_cast<float>(val.z * size) + static_cast<float>(zmin);

    return result;
}

// Fills nVoxelsX/Y/Z from field extents and returns the total voxel count.
int ComputeVoxelGridDimensions(const int radius, const int zMin, const int zMax, const int voxelSize,
                               int* nVoxelsX, int* nVoxelsY, int* nVoxelsZ)
{
    *nVoxelsZ = (zMax - zMin) / voxelSize + 1; // include zmax altitude
    *nVoxelsY = radius * 2 / voxelSize;
    *nVoxelsX = radius * 2 / voxelSize;
    return (*nVoxelsZ) * (*nVoxelsY) * (*nVoxelsX);
}
