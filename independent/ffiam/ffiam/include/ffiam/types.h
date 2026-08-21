// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include <string>
#include "cuda_runtime.h"
#include "vector_types.h"

typedef int32_t int32;

// File-path buffer size for the ABI structs below. They cross the .so boundary
// (UE -> FieldAnalysis), so they use char buffers, not std::string, to stay
// ABI-safe across C++ stdlibs (UE uses libc++, the lib uses libstdc++).
constexpr int kMaxPathLen = 512;


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
    // Values must match pyffiam CspSite and UE ECspSite (cast directly).
    site_radialSmall = 2,   // generated ~1 km radial field
    site_radial = 3,        // generated ~1.6 km radial field
    site_sampleV1 = 4,
    site_sampleV2 = 5,
    site_sampleV3 = 6,
};


// Field generation used when no helioFile is given.
enum field_layout_type
{
    layout_grid = 0,    // square grid over [-R, R)
    layout_radial = 1,  // concentric staggered rings
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

    char facetFile[kMaxPathLen];
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

    char helioFile[kMaxPathLen];

    int layout = layout_grid;  // field_layout_type; used when helioFile is empty
};


struct aim_strategy
{
    aim_strategy_type type;

    // Interpretation varies by type:
    //   aim_point -> target location (m)
    //   aim_ring  -> [inner_radius, outer_radius, height]
    //   aim_split_ring -> [inner_radius, outer_radius, height]
    //   aim_vector -> shared aim unit vector
    float3 params;

    char file[kMaxPathLen];  // CSV path for aim_data_csv
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
    int layout = layout_grid;  // see field_layout_type
};
