// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include <filesystem>
#include <string>

namespace ffiam {
namespace paths {

namespace fs = std::filesystem;

// Returns cwd/data/<filename>.
inline fs::path GetDataPath(const std::string& filename) {
    return fs::current_path() / "data" / filename;
}

// Resolves a path: returns absolute paths unchanged; prepends data dir otherwise.
inline fs::path ResolvePath(const std::string& filename) {
    if (filename.empty()) {
        return fs::path();
    }

    fs::path inputPath(filename);
    if (inputPath.is_absolute()) {
        return inputPath;
    }

    // Check if it already starts with "data/" or "data\\"
    std::string normalized = filename;
    if (normalized.find("data/") == 0 || normalized.find("data\\") == 0) {
        return fs::current_path() / inputPath;
    }

    return GetDataPath(filename);
}

inline bool FileExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec);
}

inline std::string GetWorkingDirectory() {
    return fs::current_path().string();
}

}  // namespace paths
}  // namespace ffiam
