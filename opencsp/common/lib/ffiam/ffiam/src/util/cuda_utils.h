// Copyright Sandia National Laboratories. All rights reserved.

/*
 *  This file loads CUDA-related header files, math, etc.
 *  Do not include any project-specific files here.
 */

#pragma once

#define __STDC_VERSION__ 0
#include "cuda_runtime.h"

#include <iostream>


inline void CheckStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}


