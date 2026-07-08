#pragma once

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "ffiam/ffiam.h"
#include "util/error.h"
#include "util/paths.h"


class CsvRow;
class CsvIterator;
class CsvRange;

inline float CellToFloat(std::string_view cell);


// Loads heliostat positions and design properties from CSV.
// @param helioLocs output array of xyz positions
// @param design output common heliostat properties (read from first data row)
// @return number of heliostats imported
int ImportHeliostatData(const std::string& filename, float3* helioLocs, heliostat_design* design);


// Loads per-heliostat aim vectors from CSV (columns: id, azimuth [deg CW from N], elevation [deg]).
// @return number of rows imported
int ImportAimData(const std::string& filename, float3* aimVectors);


// Loads facet origin positions from CSV (columns: id, x, y, z).
// @return number of rows imported
int ImportFacetCoordinates(std::string filename, float3* facetOrigins);


// ============================================================================
// Error-returning versions (Phase 4)
// ============================================================================

ffiam::ImportResult ImportHeliostatDataSafe(const std::string& filename,
                                            float3* helioLocs,
                                            heliostat_design* design);

ffiam::ImportResult ImportAimDataSafe(const std::string& filename,
                                      float3* aimVectors);

ffiam::ImportResult ImportFacetCoordinatesSafe(const std::string& filename,
                                               float3* facetOrigins);
