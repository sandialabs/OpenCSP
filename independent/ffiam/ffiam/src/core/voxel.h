#pragma once

#include "ffiam/ffiam.h"

// Returns world-space coordinates (m) of voxel at index i.
float3 GetVoxelLocFromIndex(int i, int size, int radius, int zmin);

// Computes per-axis voxel counts and returns total. Outputs written to nVoxelsX/Y/Z.
int ComputeVoxelGridDimensions(int radius, int zMin, int zMax, int voxelSize,
                               int* nVoxelsX, int* nVoxelsY, int* nVoxelsZ);
