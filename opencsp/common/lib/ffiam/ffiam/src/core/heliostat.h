// Copyright Sandia National Laboratories. All rights reserved.

#pragma once

#include "ffiam/ffiam.h"

// Computes aim vector, orientation vectors, and focal point for a single heliostat.
void ComputeHeliostatConfigurations(int idx,
                                    const field_layout* field,
                                    const aim_strategy* aimStrat,
                                    float3 sunVec,
                                    int rowIdx,
                                    int nHelioRows);

// Computes world-space facet origins and aim vectors, writing results into float3Data.
void ComputeAndStoreFacetData(int helioIdx,
                              const heliostat* helio,
                              const heliostat_design* design,
                              const float3* facetOrigins,
                              float3* float3Data);
