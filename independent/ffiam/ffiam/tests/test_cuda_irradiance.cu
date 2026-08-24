// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// CUDA-gated unit tests for Cuda_ComputeHeliostatIrradiance. Asserts CUDA↔CPU
// parity at the kernel level for the scenarios covered by test_cpu_irradiance.
// Built only when BUILD_CUDA_TESTS is set in CMakeLists.txt.

#include "test_framework.h"
#include "ffiam/ffiam.h"
#include "ffiam/types.h"

#include <cuda_runtime.h>
#include <vector>


namespace {

inline field_layout MakeSmallField(int voxelSize = 2,
                                   int radius = 100,
                                   int zMin = 0,
                                   int zMax = 100)
{
    field_layout f = {};
    f.voxelSize = voxelSize;
    f.radius = radius;
    f.zMin = zMin;
    f.zMax = zMax;
    const int nx = (radius * 2) / voxelSize;
    const int nz = (zMax - zMin) / voxelSize + 1;
    f.nVoxels = nx * nx * nz;
    return f;
}

inline heliostat MakeTestHelio()
{
    heliostat h = {};
    h.id = 0;
    h.loc = {30.0f, 30.0f, 0.0f};
    h.focalPt = {0.0f, 0.0f, 60.0f};
    h.focalLen = 70.0f;
    h.horizV = {1.0f, 0.0f, 0.0f};
    h.vertV  = {0.0f, 0.0f, 1.0f};
    return h;
}

inline heliostat_design MakeTestDesign(int nFacets = 1, int rows = 1, int cols = 1)
{
    heliostat_design d = {};
    d.nFacets = nFacets;
    d.nRows = rows;
    d.nCols = cols;
    d.facetWidth = 1.2f;
    d.facetHeight = 1.2f;
    d.facetsImported = false;
    return d;
}

inline int CountHitVoxels(const std::vector<float>& v)
{
    int n = 0;
    for (float x : v) if (x > 0.0f) ++n;
    return n;
}

inline float MaxOf(const std::vector<float>& v)
{
    float m = 0.0f;
    for (float x : v) if (x > m) m = x;
    return m;
}

#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cout << "  CUDA error: " << cudaGetErrorString(_e) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while (0)

bool RunCudaKernel(std::vector<float>& irradsOut,
                   field_layout fieldHost,
                   const std::vector<heliostat>& helios,
                   const heliostat_design& design,
                   float sdni, float f1, float f2, float beta,
                   float minAtten, float exp,
                   int useOuter, int useCross,
                   float fluxCorrection, float preFocal)
{
    const int nHelios = static_cast<int>(helios.size());
    fieldHost.nHelios = nHelios;

    heliostat* d_helios = nullptr;
    float* d_irrads = nullptr;
    CUDA_OK(cudaMalloc(&d_helios, sizeof(heliostat) * nHelios));
    CUDA_OK(cudaMalloc(&d_irrads, sizeof(float) * fieldHost.nVoxels));
    CUDA_OK(cudaMemset(d_irrads, 0, sizeof(float) * fieldHost.nVoxels));
    CUDA_OK(cudaMemcpy(d_helios, helios.data(), sizeof(heliostat) * nHelios, cudaMemcpyHostToDevice));

    fieldHost.helios = d_helios;

    const int threadsPerBlock = 32;
    const int blocks = (nHelios + threadsPerBlock - 1) / threadsPerBlock;
    Cuda_ComputeHeliostatIrradiance<<<blocks, threadsPerBlock>>>(
        d_irrads, d_helios, fieldHost, design, nullptr,
        sdni, f1, f2, beta, minAtten, exp,
        useOuter, useCross, fluxCorrection, preFocal);

    CUDA_OK(cudaDeviceSynchronize());

    irradsOut.assign(fieldHost.nVoxels, 0.0f);
    CUDA_OK(cudaMemcpy(irradsOut.data(), d_irrads, sizeof(float) * fieldHost.nVoxels,
                       cudaMemcpyDeviceToHost));

    cudaFree(d_helios);
    cudaFree(d_irrads);
    return true;
}

}  // namespace


TEST(CudaIrradiance, SingleHelio_HitsVoxels)
{
    field_layout field = MakeSmallField();
    std::vector<heliostat> helios{MakeTestHelio()};
    heliostat_design design = MakeTestDesign();

    std::vector<float> out;
    EXPECT_TRUE(RunCudaKernel(out, field, helios, design,
                              0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                              0, 1, 0.0f, 1.0f));
    EXPECT_TRUE(CountHitVoxels(out) > 0);
    return true;
}

TEST(CudaIrradiance, NearTower_Skipped)
{
    field_layout field = MakeSmallField();
    std::vector<heliostat> helios{MakeTestHelio()};
    helios[0].loc = {0.0f, 0.0f, 0.0f};
    heliostat_design design = MakeTestDesign();

    std::vector<float> out;
    EXPECT_TRUE(RunCudaKernel(out, field, helios, design,
                              0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                              1, 0, 0.0f, 1.0f));
    EXPECT_EQ(0, CountHitVoxels(out));
    return true;
}

// Bit-exact parity isn't realistic — FP ordering and atomic accumulation
// differ. We compare hit-voxel count (1% tolerance) and peak irradiance (1%).
TEST(CudaIrradiance, ParityWithCpu_SingleHelio)
{
    field_layout field = MakeSmallField();
    std::vector<heliostat> helios{MakeTestHelio()};
    heliostat_design design = MakeTestDesign();

    std::vector<float> gpu;
    EXPECT_TRUE(RunCudaKernel(gpu, field, helios, design,
                              0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                              0, 1, 0.0f, 1.0f));

    field_layout cpuField = field;
    cpuField.nHelios = static_cast<int>(helios.size());
    cpuField.helios = helios.data();
    std::vector<float> cpu(field.nVoxels, 0.0f);
    CPU_ComputeHeliostatIrradiance(cpu.data(), cpuField, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   0, 1, 0.0f, 1.0f);

    const int nGpu = CountHitVoxels(gpu);
    const int nCpu = CountHitVoxels(cpu);
    EXPECT_TRUE(nGpu > 0 && nCpu > 0);

    const int diff = std::abs(nGpu - nCpu);
    const int maxDiff = std::max(1, nCpu / 100);
    EXPECT_TRUE(diff <= maxDiff);

    const float peakGpu = MaxOf(gpu);
    const float peakCpu = MaxOf(cpu);
    EXPECT_TRUE(peakGpu > 0.0f && peakCpu > 0.0f);
    EXPECT_NEAR(peakCpu, peakGpu, peakCpu * 0.01f);
    return true;
}

TEST(CudaIrradiance, ParityWithCpu_OuterRays)
{
    field_layout field = MakeSmallField();
    std::vector<heliostat> helios{MakeTestHelio()};
    heliostat_design design = MakeTestDesign(9, 3, 3);

    std::vector<float> gpu;
    EXPECT_TRUE(RunCudaKernel(gpu, field, helios, design,
                              0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                              1, 0, 0.0f, 1.0f));

    field_layout cpuField = field;
    cpuField.nHelios = static_cast<int>(helios.size());
    cpuField.helios = helios.data();
    std::vector<float> cpu(field.nVoxels, 0.0f);
    CPU_ComputeHeliostatIrradiance(cpu.data(), cpuField, design, nullptr,
                                   0, 1.0f, 0.001f, 0.018f, 1.0f, 2.0f,
                                   1, 0, 0.0f, 1.0f);

    const int nGpu = CountHitVoxels(gpu);
    const int nCpu = CountHitVoxels(cpu);
    EXPECT_TRUE(nGpu > 0 && nCpu > 0);
    const int diff = std::abs(nGpu - nCpu);
    EXPECT_TRUE(diff <= std::max(1, nCpu / 100));

    EXPECT_NEAR(MaxOf(cpu), MaxOf(gpu), MaxOf(cpu) * 0.01f);
    return true;
}
