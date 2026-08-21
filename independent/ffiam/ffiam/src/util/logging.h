// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "fmt/core.h"
#include <iostream>
#include <string>

// @param nDecimals decimal places; units label appended if provided
inline void PrintParam(const std::string& name, float value, int nDecimals = 2, const std::string& units = "")
{
    std::cout << fmt::format("{} \t{:.{}f} {}\n", name, value, nDecimals, units);
}

inline void Log(const std::string& msg)
{
    std::cout << msg << std::endl;
}
