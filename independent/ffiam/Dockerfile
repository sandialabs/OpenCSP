FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (ffmpeg needed by moviepy for GIF export)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Allow pip install without venv (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Ensure CUDA libraries are discoverable by ctypes.util.find_library
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
RUN ldconfig

# C++ locale support (fmt library requires a valid locale)
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Copy and build C++ library
# Use explicit CUDA_ARCHITECTURES since no GPU is available during docker build
WORKDIR /opt/ffiam
COPY ffiam/ ./ffiam/
# Convert Windows line endings in CSV data files (C++ parser doesn't handle \r)
RUN find ./ffiam/data -name "*.csv" -exec sed -i 's/\r$//' {} +
WORKDIR /opt/ffiam/ffiam
# Build both the CUDA library and the CPU-only fallback so the image runs with
# or without a GPU (the one-click quickstart picks the backend at run time).
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120" \
    && cmake --build build --target ffiam_lib ffiam_lib_cpu

# Copy and install Python wrapper
WORKDIR /opt/ffiam
COPY pyffiam/ ./pyffiam/
RUN find ./pyffiam/src/pyffiam/data -name "*.csv" -exec sed -i 's/\r$//' {} +
RUN pip install --no-cache-dir -e ./pyffiam

# Verify both shared libraries are in place
RUN ls -la ./pyffiam/src/pyffiam/external/libffiam_lib.so \
           ./pyffiam/src/pyffiam/external/libffiam_lib_cpu.so

WORKDIR /opt/ffiam/pyffiam
ENTRYPOINT ["python3"]
