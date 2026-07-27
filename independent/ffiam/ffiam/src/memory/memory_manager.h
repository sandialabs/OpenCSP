#pragma once

#include <memory>
#include <cstdint>
#include <iostream>
#include "memory/pool.h"
#include "cuda_runtime.h"

namespace ffiam {

// RAII wrapper for the pool allocator. Owns the backing buffer.
class MemoryPool {
public:
    // @param totalSize total buffer size in bytes
    // @param chunkSize size of each allocation chunk in bytes
    MemoryPool(size_t totalSize, size_t chunkSize, size_t alignment = 8)
        : totalSize_(totalSize), chunkSize_(chunkSize) {

        buffer_ = std::make_unique<uint8_t[]>(totalSize);
        if (buffer_) {
            pool_init(&pool_, buffer_.get(), totalSize, chunkSize, alignment);
            initialized_ = true;
        }
    }

    // Non-copyable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // Movable
    MemoryPool(MemoryPool&& other) noexcept
        : buffer_(std::move(other.buffer_)),
          pool_(other.pool_),
          totalSize_(other.totalSize_),
          chunkSize_(other.chunkSize_),
          initialized_(other.initialized_) {
        other.initialized_ = false;
    }

    MemoryPool& operator=(MemoryPool&& other) noexcept {
        if (this != &other) {
            buffer_ = std::move(other.buffer_);
            pool_ = other.pool_;
            totalSize_ = other.totalSize_;
            chunkSize_ = other.chunkSize_;
            initialized_ = other.initialized_;
            other.initialized_ = false;
        }
        return *this;
    }

    ~MemoryPool() = default;

    // Returns nullptr if pool is exhausted.
    void* allocate() {
        if (!initialized_) return nullptr;
        return pool_alloc(&pool_);
    }

    void deallocate(void* ptr) {
        if (!initialized_ || !ptr) return;
        pool_free(&pool_, ptr);
    }

    // Frees all allocations without releasing the backing buffer.
    void reset() {
        if (!initialized_) return;
        pool_free_all(&pool_);
    }

    void* data() { return buffer_.get(); }
    const void* data() const { return buffer_.get(); }

    size_t size() const { return totalSize_; }
    size_t chunkSize() const { return chunkSize_; }
    bool valid() const { return initialized_; }

    Pool* getPool() { return &pool_; }

private:
    std::unique_ptr<uint8_t[]> buffer_;
    Pool pool_{};
    size_t totalSize_ = 0;
    size_t chunkSize_ = 0;
    bool initialized_ = false;
};


// RAII wrapper for CUDA device memory.
class CudaBuffer {
public:
    explicit CudaBuffer(size_t size) : size_(size) {
        cudaError_t err = cudaMalloc(&ptr_, size);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] CudaBuffer: cudaMalloc failed: "
                      << cudaGetErrorString(err) << std::endl;
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    // Non-copyable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Movable
    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~CudaBuffer() {
        free();
    }

    // @param bytes must be <= size()
    cudaError_t copyToDevice(const void* src, size_t bytes) {
        if (!ptr_ || bytes > size_) return cudaErrorInvalidValue;
        return cudaMemcpy(ptr_, src, bytes, cudaMemcpyHostToDevice);
    }

    // @param bytes must be <= size()
    cudaError_t copyToHost(void* dst, size_t bytes) {
        if (!ptr_ || bytes > size_) return cudaErrorInvalidValue;
        return cudaMemcpy(dst, ptr_, bytes, cudaMemcpyDeviceToHost);
    }

    void* get() { return ptr_; }
    const void* get() const { return ptr_; }

    template<typename T>
    T* as() { return static_cast<T*>(ptr_); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }

    size_t size() const { return size_; }
    bool valid() const { return ptr_ != nullptr; }

    // Releases ownership; caller must call cudaFree.
    void* release() {
        void* p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return p;
    }

private:
    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void* ptr_ = nullptr;
    size_t size_ = 0;
};


// RAII wrapper for CUDA unified memory (accessible from host and device).
class CudaUnifiedBuffer {
public:
    explicit CudaUnifiedBuffer(size_t size) : size_(size) {
        cudaError_t err = cudaMallocManaged(&ptr_, size);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] CudaUnifiedBuffer: cudaMallocManaged failed: "
                      << cudaGetErrorString(err) << std::endl;
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    // Non-copyable
    CudaUnifiedBuffer(const CudaUnifiedBuffer&) = delete;
    CudaUnifiedBuffer& operator=(const CudaUnifiedBuffer&) = delete;

    // Movable
    CudaUnifiedBuffer(CudaUnifiedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaUnifiedBuffer& operator=(CudaUnifiedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~CudaUnifiedBuffer() {
        free();
    }

    void* get() { return ptr_; }
    const void* get() const { return ptr_; }

    template<typename T>
    T* as() { return static_cast<T*>(ptr_); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }

    size_t size() const { return size_; }
    bool valid() const { return ptr_ != nullptr; }

    cudaError_t prefetchToDevice(int device = 0) {
        if (!ptr_) return cudaErrorInvalidValue;
#if CUDART_VERSION >= 13000
        // CUDA 13 replaced the int-device overload with a cudaMemLocation.
        cudaMemLocation loc{};
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
        return cudaMemPrefetchAsync(ptr_, size_, loc, 0, 0);
#else
        return cudaMemPrefetchAsync(ptr_, size_, device);
#endif
    }

    cudaError_t prefetchToHost() {
        if (!ptr_) return cudaErrorInvalidValue;
#if CUDART_VERSION >= 13000
        cudaMemLocation loc{};
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
        return cudaMemPrefetchAsync(ptr_, size_, loc, 0, 0);
#else
        return cudaMemPrefetchAsync(ptr_, size_, cudaCpuDeviceId);
#endif
    }

private:
    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void* ptr_ = nullptr;
    size_t size_ = 0;
};


// Convenience size literals
constexpr size_t operator""_KB(unsigned long long x) { return x * 1024; }
constexpr size_t operator""_MB(unsigned long long x) { return x * 1024 * 1024; }
constexpr size_t operator""_GB(unsigned long long x) { return x * 1024 * 1024 * 1024; }

}  // namespace ffiam
