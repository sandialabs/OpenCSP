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
