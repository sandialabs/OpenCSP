// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#include "ffiam/ffiam.h"
#include "core/constants.h"

#ifndef FFIAM_CPU_ONLY

using namespace ffiam::constants;


// Normalizes a float3 in-place.
__device__
void Cu_Norm(float3* a)
{
    float dist = length(*a);
    *a /= dist;
}


// Distance from point p along vec to plane intersection.
__device__
float Cuda_DistToPlane(float3 p, float3 vec, float3 planeP, float3 planeN)
{
    float result = dot((planeP - p), planeN) / dot(vec, planeN);
    return result;
}


// Rodrigues' rotation formula. axis must be normalized; angle in radians.
__device__
float3 Cu_RotateAroundAxis(float3 v, float3 axis, float angle)
{
    float cosA = cosf(angle);
    float sinA = sinf(angle);
    float3 result = v * cosA + cross(axis, v) * sinA + axis * dot(axis, v) * (1.0f - cosA);
    return result;
}


// Returns 1D voxel array index from world-space position.
__device__
int Cuda_GetVoxelIndexFromLoc(const float3 loc, const int size, const int radius, const int zmin)
{
    int nPerSide = (radius * 2) / size;
    int nPerPlane = nPerSide * nPerSide;

    // restore origin and step-size of 1
    const int z = static_cast<int>((loc.z - static_cast<float>(zmin)) / static_cast<float>(size));
    const int y = static_cast<int>((loc.y + static_cast<float>(radius)) / static_cast<float>(size));
    const int x = static_cast<int>((loc.x + static_cast<float>(radius)) / static_cast<float>(size));
    const int idx = z * nPerPlane + y * nPerSide + x;
    return idx;
}


// Walk a ray from rayOrigin toward focalPt and accumulate irradiance into voxels along the path.
// Uses a step-and-test traversal; each voxel hit at most once per ray via a small hit cache.
// @param irrads          Flat voxel irradiance array [kW/m²]; atomically accumulated into.
// @param field           Field/voxel grid geometry.
// @param focalPt         World-space focal point the ray aims at [m].
// @param rayOrigin       Ray start position (facet surface point) [m].
// @param focalLen        Distance from rayOrigin to focalPt [m]; used to compute fR.
// @param sdni            Scaled DNI [W/cm²] (currently unused in irradiance term).
// @param f1              Irradiance amplitude scale (reflectivity × solid-angle factor).
// @param f2              Irradiance distance scale (controls falloff rate).
// @param minAttenuation  Pre-focal attenuation floor; 1.0 = no floor (symmetric model).
// @param irradExponent   Power-law exponent for distance attenuation (2.0 = inverse square).
// @param fluxCorrectionScale  Blends piecewise flux correction on [0, 1]; 0 = disabled.
// @param preFocalScale   Scales pre-focal attenuation; 1.0 = symmetric behavior.
__device__
void Cuda_ComputeIrradFromFacetRay(float* irrads,
                                   field_layout* field,
                                   float3 focalPt,
                                   float3 rayOrigin,
                                   float focalLen,
                                   float sdni,
                                   float f1,
                                   float f2,
                                   float minAttenuation,
                                   float irradExponent,
                                   float fluxCorrectionScale,
                                   float preFocalScale)
{
    float3 planeN = { 0, 0, PLANE_NORMAL_Z};
    const float3 topPlanePt = { 0.0f, 0, static_cast<float>(field->zMax) + field->voxelSize};

    // ray aim vector for this facet, based on focal length
    float3 rayAimV = focalPt - rayOrigin;
    Cu_Norm(&rayAimV);

    // track recent hits via list of indexes (prevents double-counting)
    int voxelHits[VOXEL_HIT_CACHE_SIZE] = { -1 };
    int hitIndex = 0;

    // distance along aim vector to hit top of field (line-plane intersection)
    float distToTop = Cuda_DistToPlane(rayOrigin, rayAimV, topPlanePt, planeN);
    // Step size scales with voxel size to ensure diagonal rays hit all traversed voxels
    const float stepSize = static_cast<float>(field->voxelSize) * RAY_STEP_MULTIPLIER;

    // determine impacted voxels by walking rays.
    for (float dist = 0; dist < distToTop; dist += stepSize)
    {
        float3 rayP = rayOrigin + rayAimV * dist;

        if (rayP.z < field->zMin || rayP.z >= field->voxelSize + field->zMax ||
            rayP.x < -field->radius || rayP.x > field->radius ||
            rayP.y < -field->radius || rayP.y > field->radius)
        {
            continue;
        }

        int vi = Cuda_GetVoxelIndexFromLoc(rayP, field->voxelSize, field->radius, field->zMin);

        if (vi >= field->nVoxels)
        {
            continue;
        }

        if (voxelHits[0] == vi || voxelHits[1] == vi || voxelHits[2] == vi || voxelHits[3] == vi ||
            voxelHits[4] == vi || voxelHits[5] == vi || voxelHits[6] == vi || voxelHits[7] == vi)
        {
            continue;
        }

        // Compute distance ratio and attenuation term
        // Asymmetric: below focal point, beam is converging and more concentrated
        // than symmetric |fR - 1| suggests. preFocalScale reduces attenuation.
        // preFocalScale=1.0 recovers original symmetric behavior.
        float fR = dist / focalLen;
        float rawAtten = (fR < 1.0f)
            ? preFocalScale * (1.0f - fR)   // Pre-focal: scaled attenuation
            : (fR - 1.0f);                  // Post-focal: unchanged
        float attenuation;

        // Hybrid attenuation model: smooth blend toward minimum at low fR
        // When minAttenuation >= 1.0, the blend is a no-op (original behavior)
        if (minAttenuation >= 1.0f || fR >= flux::BLEND_START_FR)
        {
            attenuation = rawAtten;
        }
        else
        {
            float t = fR / flux::BLEND_START_FR;
            float blendedMin = minAttenuation + (1.0f - flux::BLEND_START_FR - minAttenuation) * t;
            attenuation = fminf(rawAtten, blendedMin);
        }

        float facetIrrad = WATTS_CM2_TO_KW_M2 * f1 * powf(dist * f2 + attenuation, -irradExponent);

        if (facetIrrad > 0)
        {
            // NOTE: Uncomment this line to account for irradiance from ambient sunlight.
            // facetIrrad += WATTS_CM2_TO_KW_M2 * sdni;

            // Apply flux correction factor based on distance ratio (fR = dist / focalLen)
            // Coefficients derived from SolTrace validation
            float fluxFactor = 1.0f;
            if (fR < flux::THRESHOLD_1)
            {
                fluxFactor = 1.0f;
            }
            else if (fR < flux::THRESHOLD_2)
            {
                fluxFactor = flux::COEFF_A1 * fR + flux::COEFF_B1;
            }
            else if (fR < flux::THRESHOLD_3)
            {
                fluxFactor = flux::COEFF_A2 * fR + flux::COEFF_B2;
            }
            else if (fR < flux::THRESHOLD_4)
            {
                fluxFactor = flux::COEFF_A3 * fR + flux::COEFF_B3;
            }
            else if (fR < flux::THRESHOLD_5)
            {
                fluxFactor = flux::COEFF_A4 * fR + flux::COEFF_B4;
            }
            else
            {
                fluxFactor = flux::MAX_FACTOR;
            }

            // Apply flux correction (scale=1.0 for full correction, 0.0 to disable)
            float scaledFactor = 1.0f + (fluxFactor - 1.0f) * fluxCorrectionScale;
            facetIrrad *= scaledFactor;

            atomicAdd(&irrads[vi], facetIrrad);
            voxelHits[hitIndex++] = vi;
        }

        // update tracking
        if (hitIndex >= 8) hitIndex = 0;
    }
}

// Computes world-space facetOrigin and local facetOffset for a single facet.
__device__
void Cuda_CalculateFacetOriginAndOffset(int mi, const float3 &helioPos,
                                        const float3 &horizV, const float3 &vertV,
                                        const heliostat_design &helio,
                                        const float3 *facetFlatOrigins,
                                        float3 &facetOrigin, float3 &facetOffset)
{
    if (helio.facetsImported)
    {
        facetOffset = facetFlatOrigins[mi];
        facetOrigin = helioPos + horizV * facetOffset.x + vertV * facetOffset.y;
    }
    else
    {
        int facetIndexX = mi / helio.nRows - helio.nCols / 2;
        int facetIndexZ = mi % helio.nCols - helio.nRows / 2;
        facetOffset = { helio.facetWidth * static_cast<float>(facetIndexX), helio.facetHeight * static_cast<float>(facetIndexZ), 0 };
        facetOrigin = helioPos + horizV * facetOffset.x + vertV * facetOffset.y;
    }
}

// CUDA kernel: one thread per heliostat. Traces core and outer facet rays into the voxel grid.
// Heliostats within 5 m of the tower origin are skipped as physically invalid.
// @param irrads           Flat voxel irradiance array [kW/m²]; written atomically per thread.
// @param helios           Per-heliostat tracking/geometry data (device pointer).
// @param field            Field and voxel grid parameters.
// @param helioDesign      Facet optical design shared by all heliostats.
// @param facetFlatOrigins Per-facet offsets in heliostat-local coords (device pointer; may be null).
// @param sdni             Scaled DNI passed through to ray tracer [W/cm²].
// @param f1               Irradiance amplitude scale.
// @param f2               Irradiance distance scale.
// @param beta             Beam spread angle [rad]; used to tilt outer rays outward.
// @param minAttenuation   Pre-focal attenuation floor; 1.0 = no floor.
// @param irradExponent    Distance-attenuation exponent (2.0 = inverse square).
// @param useOuterRays     1 = trace 3 additional tilted outer rays per facet; 0 = core only.
// @param useCrossPattern  1 = 5-ray cross pattern per facet; 0 = full 3×3 grid (9 rays).
// @param fluxCorrectionScale  Piecewise flux correction blend [0, 1]; 0 = disabled.
// @param preFocalScale    Pre-focal attenuation scale; 1.0 = symmetric behavior.
__global__
void Cuda_ComputeHeliostatIrradiance(float *irrads,
                                     heliostat *helios,
                                     field_layout field,
                                     heliostat_design helioDesign,
                                     float3 *facetFlatOrigins,
                                     float sdni,
                                     float f1,
                                     float f2,
                                     float beta,
                                     float minAttenuation,
                                     float irradExponent,
                                     int useOuterRays,
                                     int useCrossPattern,
                                     float fluxCorrectionScale,
                                     float preFocalScale)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < field.nHelios)
    {
        auto helio = &helios[idx];
        auto [hx, hy, hzy] = helio->loc;

        // ignore unrealistic heliostats within/near tower position (origin)
        if (hx < 5 && hx > -5 && hy < 5 && hy > -5)
        {
            return;
        }

        for (int facetIndex = 0; facetIndex < helioDesign.nFacets; ++facetIndex)
        {
            float3 facetOrigin, facetOffset;
            Cuda_CalculateFacetOriginAndOffset(facetIndex, helio->loc, helio->horizV, helio->vertV,
                                               helioDesign,
                                               facetFlatOrigins, facetOrigin, facetOffset);

            float adjustedF1 = f1;
            float adjustedF2 = f2;

            // Core rays: 3x3 grid (9 rays) or cross pattern (5 rays: center + 4 cardinal)
            // Cross pattern skips diagonal positions for original SolTrace-validated behavior
            float hStep = helioDesign.facetWidth / OFFSET_DIVISOR;
            float vStep = helioDesign.facetHeight / OFFSET_DIVISOR;

            for (int vi = -1; vi <= 1; ++vi)
            {
                for (int hi = -1; hi <= 1; ++hi)
                {
                    // Skip diagonal positions when using cross pattern (5 rays)
                    if (useCrossPattern && vi != 0 && hi != 0)
                        continue;
                    float3 rayOrigin = facetOrigin
                                     + helio->horizV * (hStep * static_cast<float>(hi))
                                     + helio->vertV * (vStep * static_cast<float>(vi));

                    Cuda_ComputeIrradFromFacetRay(irrads,
                                                  &field,
                                                  helio->focalPt,
                                                  rayOrigin,
                                                  helio->focalLen,
                                                  sdni,
                                                  adjustedF1,
                                                  adjustedF2,
                                                  minAttenuation,
                                                  irradExponent,
                                                  fluxCorrectionScale,
                                                  preFocalScale);
                }
            }

            // Outer tilted rays: 3 rays per facet on the outer edge, tilted outward by beta.
            // Represents beam spread/diffusion at facet edges.
            if (useOuterRays)
            {
                float offsetMag = sqrtf(facetOffset.x * facetOffset.x + facetOffset.y * facetOffset.y);
                if (offsetMag > 0.01f)  // skip center facet (no clear outer direction)
                {
                    float2 outerDir2D = { facetOffset.x / offsetMag, facetOffset.y / offsetMag };
                    float3 outerDir3D = helio->horizV * outerDir2D.x + helio->vertV * outerDir2D.y;
                    Cu_Norm(&outerDir3D);

                    float3 perpDir3D = helio->horizV * (-outerDir2D.y) + helio->vertV * outerDir2D.x;
                    Cu_Norm(&perpDir3D);

                    float3 surfaceNormal = cross(helio->horizV, helio->vertV);
                    Cu_Norm(&surfaceNormal);

                    float tiltAngle = beta;
                    float outerF1 = adjustedF1 * 0.25f;  // calibrated against UAS validation

                    for (int oi = -1; oi <= 1; ++oi)
                    {
                        float3 rayOrigin = facetOrigin
                                         + outerDir3D * hStep
                                         + perpDir3D * (vStep * static_cast<float>(oi));

                        float3 baseAimV = helio->focalPt - rayOrigin;
                        Cu_Norm(&baseAimV);

                        // tilt ray outward in the plane containing aim and outer vectors
                        float3 rotAxis = cross(baseAimV, outerDir3D);
                        float rotAxisMag = length(rotAxis);
                        if (rotAxisMag > 0.001f)
                        {
                            rotAxis /= rotAxisMag;
                            float3 tiltedAimV = Cu_RotateAroundAxis(baseAimV, rotAxis, tiltAngle);
                            float3 tiltedFocalPt = rayOrigin + tiltedAimV * helio->focalLen;

                            Cuda_ComputeIrradFromFacetRay(irrads,
                                                          &field,
                                                          tiltedFocalPt,
                                                          rayOrigin,
                                                          helio->focalLen,
                                                          sdni,
                                                          outerF1,
                                                          adjustedF2,
                                                          minAttenuation,
                                                          irradExponent,
                                                          fluxCorrectionScale,
                                                          preFocalScale);
                        }
                    }
                }
            }
        }
    }
}

#endif  // FFIAM_CPU_ONLY
