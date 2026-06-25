// Copyright Sandia National Laboratories. All rights reserved.

#pragma once

#include <string>
#include "cuda_runtime.h"
#include "vector_types.h"

typedef int32_t int32;


enum aim_strategy_type
{
    aim_null = 0,
    aim_point = 1,
    aim_ring = 2,
    aim_split_ring = 3,
    aim_vector = 4,
    aim_data_csv = 5,
    aim_fixed_normal = 6
};


enum csp_site
{
    site_null = 0,
    site_nsttf = 1,
    site_crescentDunesSmall = 2,
    site_crescentDunes = 3,
    site_sampleV1 = 4,
    site_sampleV2 = 5,
    site_sampleV3 = 6,
};


struct heliostat_design
{
    int nFacets = 0;
    int nRows = 0;
    int nCols = 0;

    int spacer;

    float pivotHeight;  // not currently used
    float pivotOffset;  // facet offset from center (m)
    float facetWidth;   // (m)
    float facetHeight;  // (m)

    std::string facetFile;
    bool facetsImported = false;
};


struct heliostat
{
    int id;

    float moveAngle;  // angle from standby to active aim position (deg)
    float3 loc;       // position relative to tower origin (m)
    float3 aimV;      // surface normal (bisector of aim & sun vectors)
    float3 horizV;    // horizontal plane unit vector
    float3 vertV;     // vertical plane unit vector
    float3 focalPt;   // focal point (m)
    float3 refV;      // unit vector toward aim point

    float focalLen;
};


struct field_layout
{
    float lat;       // decimal degrees
    float lng;       // decimal degrees
    float timezone;  // UTC offset (hours)

    int radius;      // airspace radius (m)
    int zMin = 0;    // min analysis altitude (m)
    int zMax;        // max analysis altitude (m)

    // Tower position (m); adjusted to origin before analysis.
    float3 towerPos;
    float towerHeight;  // tower height to receiver (m)

    // Data-as-arrays for Python readback.
    heliostat* helios;
    float3* locs;
    float3* aimVs;
    float* moveAngles;

    int voxelSize;   // side length (m); v3 default = 2
    int nVoxels;
    float voxelArea; // (m²)
    int nHelios;

    std::string helioFile;
};


struct aim_strategy
{
    aim_strategy_type type;

    // Interpretation varies by type:
    //   aim_point -> target location (m)
    //   aim_ring  -> [offset, height, unused]
    //   aim_vector -> shared aim unit vector
    float3 params;

    std::string file;  // CSV path for aim_data_csv
};


struct date_info
{
    int year;
    int month;
    int day;
    float hour;  // Supports fractional hours (e.g., 10.5 = 10:30 AM)
};


struct preset_site_config
{
    float2 coords;
    float timezone;
    float towerHeight;
    std::string helioFile;
    int nHelios;
    std::string facetFile;
    int nFacets;
    int nFacetCols;
    float2 facetDims;
    int radius;
};
