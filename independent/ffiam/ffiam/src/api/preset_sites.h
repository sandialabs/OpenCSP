#pragma once

#include "ffiam/ffiam.h"

// preset_site_config is defined in ffiam/types.h

// Populates config and aimFile for the given site/aim strategy.
// Returns false if siteId is unrecognised.
bool get_preset_site_config(int siteId, const int aimId, preset_site_config* config, std::string* aimFile);

// Returns airspace parameters (radius, zMin, zMax, voxelSize) for analysis version 1, 2, or 3.
void get_version_params(const int version, int* radius, int* zMin, int* zMax, int* voxelSize);
