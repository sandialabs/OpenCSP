#include "ffiam/ffiam.h"
#include "compute/ray_math.h"
#include "core/constants.h"
#include "util/math.h"

#include <cmath>

#ifdef FFIAM_HAVE_OPENMP
#include <omp.h>
#endif


using namespace ffiam::constants;


// CPU mirror of Cuda_ComputeIrradFromFacetRay. Walks a ray from rayOrigin toward
// focalPt and accumulates irradiance into voxels along the path. The full CUDA
// model is reproduced: asymmetric pre-focal attenuation with optional floor
// blend, configurable distance exponent, and piecewise flux correction.
void CPU_ComputeIrradFromFacetRay(float* irrads,
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
    (void)sdni;  // mirrors CUDA: sdni is plumbed for future ambient term but unused

    const float3 planeN = { 0.0f, 0.0f, PLANE_NORMAL_Z };
    const float3 topPlanePt = { 0.0f, 0.0f, static_cast<float>(field->zMax) + field->voxelSize };

    float3 aimV = focalPt - rayOrigin;
    Norm(&aimV);

    int voxelHits[VOXEL_HIT_CACHE_SIZE];
    for (int i = 0; i < VOXEL_HIT_CACHE_SIZE; ++i) voxelHits[i] = -1;
    int hitIndex = 0;

    const float distToTop = ffiam::DistToPlaneHost(rayOrigin, aimV, topPlanePt, planeN);
    const float stepSize = static_cast<float>(field->voxelSize) * RAY_STEP_MULTIPLIER;

    for (float dist = 0.0f; dist < distToTop; dist += stepSize)
    {
        const float3 rayP = rayOrigin + aimV * dist;

        if (rayP.z < field->zMin || rayP.z >= field->voxelSize + field->zMax ||
            rayP.x < -field->radius || rayP.x > field->radius ||
            rayP.y < -field->radius || rayP.y > field->radius)
        {
            continue;
        }

        const int vi = ffiam::GetVoxelIndexFromLocCPU(rayP, field->voxelSize, field->radius, field->zMin);
        if (vi >= field->nVoxels) continue;

        bool alreadyHit = false;
        for (int k = 0; k < VOXEL_HIT_CACHE_SIZE; ++k)
        {
            if (voxelHits[k] == vi) { alreadyHit = true; break; }
        }
        if (alreadyHit) continue;

        // Asymmetric pre-focal attenuation with optional floor blend.
        const float fR = dist / focalLen;
        const float rawAtten = (fR < 1.0f)
            ? preFocalScale * (1.0f - fR)
            : (fR - 1.0f);
        float attenuation;
        if (minAttenuation >= 1.0f || fR >= flux::BLEND_START_FR)
        {
            attenuation = rawAtten;
        }
        else
        {
            const float t = fR / flux::BLEND_START_FR;
            const float blendedMin = minAttenuation +
                                     (1.0f - flux::BLEND_START_FR - minAttenuation) * t;
            attenuation = std::fmin(rawAtten, blendedMin);
        }

        float facetIrrad = WATTS_CM2_TO_KW_M2 * f1
                         * std::pow(dist * f2 + attenuation, -irradExponent);

        if (facetIrrad > 0.0f)
        {
            // Piecewise flux correction, calibrated to SolTrace.
            float fluxFactor;
            if      (fR < flux::THRESHOLD_1) fluxFactor = 1.0f;
            else if (fR < flux::THRESHOLD_2) fluxFactor = flux::COEFF_A1 * fR + flux::COEFF_B1;
            else if (fR < flux::THRESHOLD_3) fluxFactor = flux::COEFF_A2 * fR + flux::COEFF_B2;
            else if (fR < flux::THRESHOLD_4) fluxFactor = flux::COEFF_A3 * fR + flux::COEFF_B3;
            else if (fR < flux::THRESHOLD_5) fluxFactor = flux::COEFF_A4 * fR + flux::COEFF_B4;
            else                              fluxFactor = flux::MAX_FACTOR;

            const float scaledFactor = 1.0f + (fluxFactor - 1.0f) * fluxCorrectionScale;
            facetIrrad *= scaledFactor;

#ifdef FFIAM_HAVE_OPENMP
            #pragma omp atomic
#endif
            irrads[vi] += facetIrrad;
            voxelHits[hitIndex++] = vi;
            if (hitIndex >= VOXEL_HIT_CACHE_SIZE) hitIndex = 0;
        }
    }
}


// Computes world-space facetOrigin and local facetOffset for a single facet.
static void CalculateFacetOriginAndOffset(int mi, const float3& helioPos,
                                          const float3& horizV, const float3& vertV,
                                          const heliostat_design& helio,
                                          const float3* facetFlatOrigins,
                                          float3& facetOrigin, float3& facetOffset)
{
    if (helio.facetsImported)
    {
        facetOffset = facetFlatOrigins[mi];
        facetOrigin = helioPos + horizV * facetOffset.x + vertV * facetOffset.y;
    }
    else
    {
        const int facetIndexX = mi / helio.nRows - helio.nCols / 2;
        const int facetIndexZ = mi % helio.nCols - helio.nRows / 2;
        facetOffset = { helio.facetWidth  * static_cast<float>(facetIndexX),
                        helio.facetHeight * static_cast<float>(facetIndexZ),
                        0.0f };
        facetOrigin = helioPos + horizV * facetOffset.x + vertV * facetOffset.y;
    }
}


// CPU mirror of Cuda_ComputeHeliostatIrradiance. Traces core 3x3 (or cross) ray
// pattern and optional outer tilted rays per facet, accumulating irradiance into
// the voxel grid. Skips heliostats within +/-5 m of the tower (physically invalid).
void CPU_ComputeHeliostatIrradiance(float *irrads,
                                    field_layout field,
                                    const heliostat_design &heliostat,
                                    const float3 *facetFlatOrigins,
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
#ifdef FFIAM_HAVE_OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int i = 0; i < field.nHelios; ++i)
    {
        auto helio = &field.helios[i];
        const auto [hx, hy, hz] = helio->loc;

        if (hx < 5 && hx > -5 && hy < 5 && hy > -5)
        {
            continue;
        }

        for (int facetIndex = 0; facetIndex < heliostat.nFacets; ++facetIndex)
        {
            float3 facetOrigin, facetOffset;
            CalculateFacetOriginAndOffset(facetIndex, helio->loc, helio->horizV, helio->vertV,
                                          heliostat, facetFlatOrigins, facetOrigin, facetOffset);

            const float adjustedF1 = f1;
            const float adjustedF2 = f2;

            // Core rays: full 3x3 grid (9 rays) or 5-ray cross (center + 4 cardinal).
            const float hStep = heliostat.facetWidth / OFFSET_DIVISOR;
            const float vStep = heliostat.facetHeight / OFFSET_DIVISOR;

            for (int vi = -1; vi <= 1; ++vi)
            {
                for (int hi = -1; hi <= 1; ++hi)
                {
                    if (useCrossPattern && vi != 0 && hi != 0) continue;
                    const float3 rayOrigin = facetOrigin
                                           + helio->horizV * (hStep * static_cast<float>(hi))
                                           + helio->vertV  * (vStep * static_cast<float>(vi));
                    CPU_ComputeIrradFromFacetRay(irrads, &field, helio->focalPt, rayOrigin,
                                                 helio->focalLen, sdni, adjustedF1, adjustedF2,
                                                 minAttenuation, irradExponent,
                                                 fluxCorrectionScale, preFocalScale);
                }
            }

            // Outer tilted rays: 3 per facet on the outer edge, tilted outward by beta.
            if (useOuterRays)
            {
                const float offsetMag = std::sqrt(facetOffset.x * facetOffset.x +
                                                  facetOffset.y * facetOffset.y);
                if (offsetMag > 0.01f)
                {
                    const float2 outerDir2D = { facetOffset.x / offsetMag,
                                                facetOffset.y / offsetMag };
                    float3 outerDir3D = helio->horizV * outerDir2D.x +
                                        helio->vertV  * outerDir2D.y;
                    Norm(&outerDir3D);

                    float3 perpDir3D = helio->horizV * (-outerDir2D.y) +
                                       helio->vertV  *   outerDir2D.x;
                    Norm(&perpDir3D);

                    const float tiltAngle = beta;
                    const float outerF1 = adjustedF1 * 0.25f;  // calibrated against UAS validation

                    for (int oi = -1; oi <= 1; ++oi)
                    {
                        const float3 rayOrigin = facetOrigin
                                               + outerDir3D * hStep
                                               + perpDir3D  * (vStep * static_cast<float>(oi));

                        float3 baseAimV = helio->focalPt - rayOrigin;
                        Norm(&baseAimV);

                        float3 rotAxis = cross(baseAimV, outerDir3D);
                        const float rotAxisMag = length(rotAxis);
                        if (rotAxisMag > 0.001f)
                        {
                            rotAxis = rotAxis / rotAxisMag;
                            const float3 tiltedAimV = ffiam::RotateAroundAxis(baseAimV, rotAxis, tiltAngle);
                            const float3 tiltedFocalPt = rayOrigin + tiltedAimV * helio->focalLen;
                            CPU_ComputeIrradFromFacetRay(irrads, &field, tiltedFocalPt, rayOrigin,
                                                         helio->focalLen, sdni, outerF1, adjustedF2,
                                                         minAttenuation, irradExponent,
                                                         fluxCorrectionScale, preFocalScale);
                        }
                    }
                }
            }
        }
    }
}
