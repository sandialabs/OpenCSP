#pragma once

#include <string>
#include <iostream>

namespace ffiam {

enum class ErrorCode {
    Success = 0,
    FileNotFound,
    ParseError,
    InvalidFormat,
    MemoryAllocationFailed,
    InvalidParameter,
    CudaError,
    Unknown
};

inline const char* ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success: return "Success";
        case ErrorCode::FileNotFound: return "File not found";
        case ErrorCode::ParseError: return "Parse error";
        case ErrorCode::InvalidFormat: return "Invalid format";
        case ErrorCode::MemoryAllocationFailed: return "Memory allocation failed";
        case ErrorCode::InvalidParameter: return "Invalid parameter";
        case ErrorCode::CudaError: return "CUDA error";
        case ErrorCode::Unknown: return "Unknown error";
        default: return "Unrecognized error";
    }
}

// Result type carrying an error code, optional message, and value.
template<typename T>
struct Result {
    ErrorCode error = ErrorCode::Success;
    std::string message;
    T value;

    bool ok() const { return error == ErrorCode::Success; }

    static Result Ok(T val) {
        return Result{ErrorCode::Success, "", val};
    }

    static Result Fail(ErrorCode err, const std::string& msg = "") {
        return Result{err, msg, T{}};
    }
};

// void specialization (no value).
template<>
struct Result<void> {
    ErrorCode error = ErrorCode::Success;
    std::string message;

    bool ok() const { return error == ErrorCode::Success; }

    static Result Ok() {
        return Result{ErrorCode::Success, ""};
    }

    static Result Fail(ErrorCode err, const std::string& msg = "") {
        return Result{err, msg};
    }
};

// Result for CSV import operations; includes row count.
struct ImportResult {
    ErrorCode error = ErrorCode::Success;
    std::string message;
    int rowCount = 0;

    bool ok() const { return error == ErrorCode::Success; }
};

inline void LogError(const char* func, const std::string& msg) {
    std::cerr << "[ERROR] " << func << ": " << msg << std::endl;
}

inline void LogWarning(const char* func, const std::string& msg) {
    std::cerr << "[WARNING] " << func << ": " << msg << std::endl;
}

template<typename T>
inline bool ValidatePointer(T* ptr, const char* name) {
    if (ptr == nullptr) {
        LogError(__func__, std::string("Null pointer: ") + name);
        return false;
    }
    return true;
}

// Returns false and logs an error if value is outside [min, max].
template<typename T>
inline bool ValidateRange(T value, T min, T max, const char* name) {
    if (value < min || value > max) {
        LogError(__func__, std::string(name) + " out of range");
        return false;
    }
    return true;
}

}  // namespace ffiam
