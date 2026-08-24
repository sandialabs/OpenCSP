// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#include "core/heliostat.h"
#include "util/math.h"
#include <cassert>
#include <cmath>

constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;

// Writes world-space facet origins and normalized aim vectors into float3Data for Python consumption.
void ComputeAndStoreFacetData(const int helioIdx,
                              const heliostat* helio,
                              const heliostat_design* design,
                              const float3* facetOrigins,
                              float3* float3Data)
{
    for (int fi = 0; fi < design->nFacets; fi++)
    {
        auto [x, y, z] = facetOrigins[fi];
        const float3 facetOrigin = helio->loc + helio->horizV * x + helio->vertV * y;
        const int idx = helioIdx * 2 * design->nFacets + 2 * fi;
        float3Data[idx] = facetOrigin;

        float3 facetAimV = helio->focalPt - facetOrigin;
        Norm(&facetAimV);
        float3Data[idx + 1] = facetAimV;
    }
}


// Computes aim vector, normal, focal point, and on/off-target movement angle for one heliostat.
void ComputeHeliostatConfigurations(const int idx,
                                    const field_layout* field,
                                    const aim_strategy* aimStrat,
                                    const float3 sunVec,
                                    const int rowIdx,
                                    const int nHelioRows)
{
    heliostat* helio = &field->helios[idx];
    float3 refV; // vector pointing from heliostat to aim point
    float3 aimV; // actual heliostat normal vector (i.e. bisector of aim & sun vectors)

    float3 onRefV; // vector pointing from heliostat to receiver; i.e. when on-target
    float3 onAimV; // heliostat normal vector when on-target receiver

    float3 receiverPos = {0, 0, field->towerHeight};
    onRefV = receiverPos - helio->loc;
    helio->focalLen = length(onRefV);
    Norm(&onRefV);

    // compute aim vector and prep for facet calculations with vectors on heliostat plane (vertical and horizontal)
    if (aimStrat->type == aim_point)
    {
        refV = aimStrat->params - helio->loc;
    }
    else if (aimStrat->type == aim_ring)
    {
        // Annular ring aim. params = (inner_radius, outer_radius, height).
        // Each heliostat aims at a point on a circle whose radius is spread
        // linearly across the annulus [innerR, outerR] by heliostat index, at a
        // fixed height. The aim point is offset horizontally, perpendicular to
        // the heliostat->tower vector (tangential offset).
        const float innerR = aimStrat->params.x;
        const float outerR = aimStrat->params.y;
        const float height = aimStrat->params.z;

        const float frac = (field->nHelios > 1)
                               ? static_cast<float>(idx) / static_cast<float>(field->nHelios - 1)
                               : 0.0f;
        const float r = innerR + (outerR - innerR) * frac;

        float3 hV = {-onRefV.y, onRefV.x, 0};
        Norm(&hV);
        float3 aimPt = hV * r;
        aimPt.z = height;
        refV = aimPt - helio->loc;
    }
    else if (aimStrat->type == aim_split_ring)
    {
        // params = (inner_radius, outer_radius, height); shares the ring convention.
        // Within each azimuthal "pizza slice" the aim radius fans across the annulus
        // [inner, outer] (previously a single radius plus a fixed 8 m spread band).
        int nHelioPerRow = static_cast<int>(std::ceil(field->nHelios / nHelioRows));
        int nHelioPerSlice = static_cast<int>(std::ceil(nHelioPerRow / 2.0f)); // half of row east, half west

        const float innerR = aimStrat->params.x;
        const float outerR = aimStrat->params.y;
        const float height = aimStrat->params.z;

        // southernmost rows aim at southernmost edge of circle
        auto helioIndexIntoSlice = static_cast<float>(idx % nHelioPerSlice);
        auto hScaler = helioIndexIntoSlice / static_cast<float>(nHelioPerSlice);

        auto anglePerRow = 180.0f / static_cast<float>(nHelioRows); // pizza slices
        auto angleOffset = anglePerRow * hScaler;
        auto angle = static_cast<float>(rowIdx) * anglePerRow + angleOffset;
        // angle around semicircle; include offset to reduce hotspot
        angle -= 90.0f; // shift origin so counterclockwise from due south

        // spread aim radius across the annulus within the slice
        const float rAdj = innerR + (outerR - innerR) * hScaler;

        float isEast = helio->loc.x >= 0.0f ? 1.0f : -1.0f;
        float3 aimPt = {
            std::cos(DEG_TO_RAD * angle) * rAdj * isEast,
            std::sin(DEG_TO_RAD * angle) * rAdj,
            height
        };

        refV = aimPt - helio->loc;
    }
    else if (aimStrat->type == aim_vector)
    {
        // common aim vector across all heliostats
        refV = aimStrat->params;
    }
    else if (aimStrat->type == aim_data_csv)
    {
        // imported from file
        refV = field->aimVs[idx];
    }
    else if (aimStrat->type == aim_fixed_normal)
    {
        // Fixed heliostat orientation (e.g., STOW position)
        // The params vector IS the surface normal, not the aim direction
        // Reflected beam direction depends on sun position via law of reflection
        aimV = aimStrat->params;
        Norm(&aimV);

        // Compute reflected direction: R = 2*(N·S)*N - S
        // where N = surface normal (aimV), S = sun direction (sunVec)
        float NdotS = dot(aimV, sunVec);
        refV = aimV * (2.0f * NdotS) - sunVec;
        Norm(&refV);

        // Skip the normal bisector calculation below since we already set aimV
        goto skip_bisector;
    }
    else
    {
        assert(false);
    }

    Norm(&refV);
    aimV = GetBisector(refV, sunVec);
    Norm(&aimV);

skip_bisector:

    float3 helioFocalPt = helio->loc + refV * helio->focalLen;

    // Compute horizontal vector perpendicular to heliostat normal
    // Standard case: horizV is perpendicular to aimV in XY plane
    float3 horizV = {aimV.y, -aimV.x, 0};
    float horizLen = length(horizV);

    // Handle edge case when aimV is nearly vertical (face-up/face-down)
    // In this case, horizV would be ~zero, so we use a fixed reference direction
    if (horizLen < 0.001f)
    {
        // For vertical normal, use East direction as horizontal reference
        horizV = {1.0f, 0.0f, 0.0f};
    }
    else
    {
        horizV = horizV / horizLen;  // normalize
    }

    float3 vertV = cross(aimV, horizV);
    Norm(&vertV);
    if (vertV.z < 0)
    {
        vertV *= -1.0f;
    }

    // angle between standby and on-target, for time-to-target optimization
    onAimV = GetBisector(onRefV, sunVec);
    float3 onAimVectorXY = {onAimV.x, onAimV.y, 0};
    Norm(&onAimVectorXY);

    float3 onAimVectorZ = {0, 0, onAimV.z};
    Norm(&onAimVectorZ);

    float moveAngleXY = std::abs(std::acos(dot(horizV, onAimVectorXY)) * TO_DEG);
    float moveAngleZ = std::abs(std::acos(dot(vertV, onAimVectorZ)) * TO_DEG);

    field->moveAngles[idx] = moveAngleXY + moveAngleZ;
    helio->moveAngle = moveAngleXY + moveAngleZ;

    helio->horizV = horizV;
    helio->vertV = vertV;
    helio->focalPt = helioFocalPt;
    helio->aimV = aimV;
    helio->refV = refV;
}
