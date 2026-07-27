#pragma once

#include "ffiam/ffiam.h"

struct SolarData {
    float azimuth;      // Sun azimuth angle [deg]
    float elevation;    // Sun elevation angle [deg]
    float3 sunVector;   // Unit vector pointing toward sun
    float scaledDni;    // DNI scaled by time-of-day factor
    float sunrise;      // Sunrise time [minutes from midnight]
    float sunset;       // Sunset time [minutes from midnight]
    float trueSolarTime; // True solar time [minutes from midnight]
};

// Computes solar position and scaled DNI via SOLPOS.
// @param hour fractional hour, e.g. 10.5 = 10:30 AM
// @return false if sun is below horizon or DNI is zero
bool ComputeSolarPosition(const float lat, const float lng, const float timezone,
                          const int year, const int month, const int day, const float hour,
                          const float baseDni, SolarData* outData);
