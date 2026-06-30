#include "solar/solar_position.h"
#include "util/math.h"
#include <cmath>
#include <cstdio>

extern "C" {
#include "solpos.c"
}

// Runs SOLPOS for the given site and time, scales DNI by time-from-noon factor, and fills outData.
// Returns false if the sun is below the horizon or the scaled DNI is zero.
bool ComputeSolarPosition(const float lat, const float lng, const float timezone,
                          const int year, const int month, const int day, const float hour,
                          const float baseDni, SolarData* outData)
{
    posdata pd;
    posdata* pdat = &pd;

    S_init(pdat);

    // Use month, day input instead of day-of-year
    pdat->function = ((S_REFRAC | S_SOLAZM | S_SRSS | S_TST) & ~S_DOY);

    pdat->latitude = lat;
    pdat->longitude = lng;
    pdat->timezone = timezone;

    pdat->year = year;
    pdat->month = month;
    pdat->day = day;

    // Convert fractional hour to integer hour, minute, second for SOLPOS
    // Example: 10.5 -> 10:30:00, 14.75 -> 14:45:00
    int hours = static_cast<int>(std::floor(hour));
    float remainder = (hour - static_cast<float>(hours)) * 60.0f;
    int minutes = static_cast<int>(std::floor(remainder));
    int seconds = static_cast<int>((remainder - static_cast<float>(minutes)) * 60.0f);

    pdat->hour = hours;
    pdat->minute = minutes;
    pdat->second = seconds;
    pdat->temp = 27.0;
    pdat->press = 1006.0;

    pdat->tilt = pdat->latitude;
    pdat->aspect = 135.0;

    long retval = S_solpos(pdat);
    S_decode(retval, pdat);

    outData->azimuth = pdat->azim;
    outData->elevation = pdat->elevref;
    outData->sunrise = pdat->sretr;
    outData->sunset = pdat->ssetr;
    outData->trueSolarTime = pdat->tst;

    // Check if sun is below horizon
    if (outData->elevation <= 0) {
        printf("ERROR: sun below horizon! Aborting analysis.\n");
        return false;
    }

    // Scale DNI based on time from sunrise vs. solar noon
    // Assumes peak DNI at solar noon
    const float tSunrise = pdat->sretr;
    const float tSunset = pdat->ssetr;
    const float tst = pdat->tst;
    const float noonMin = 60 * 12;

    const float nowToSolnoon = std::abs(noonMin - tst);
    const bool beforeSolNoon = tst < noonMin;
    float solFactor = 1.0f;

    if (beforeSolNoon) {
        const float sunriseToSolnoon = noonMin - tSunrise;
        solFactor = 1.0f - nowToSolnoon / sunriseToSolnoon;
        printf("Sunrise to solnoon: %.0f, nowToSolnoon %.0f\n", sunriseToSolnoon, nowToSolnoon);
    } else {
        const float solnoonToSunset = tSunset - noonMin;
        solFactor = 1.0f - nowToSolnoon / solnoonToSunset;
        printf("Sunset to solnoon: %.0f, nowToSolnoon %.0f\n", solnoonToSunset, nowToSolnoon);
    }

    const float normDni = std::cos(1.0f - std::abs(solFactor));
    outData->scaledDni = baseDni * normDni;

    if (outData->scaledDni <= 0) {
        printf("ERROR: DNI is 0. Sun down? Aborting analysis.\n");
        return false;
    }

    // Compute sun unit vector
    outData->sunVector = ConvertAzElToCartesian(outData->azimuth, outData->elevation);
    Norm(&outData->sunVector);

    printf("Sun position: Azim %.1f, Elev %.1f, (%.2f, %.2f, %.2f)\n",
           outData->azimuth, outData->elevation,
           outData->sunVector.x, outData->sunVector.y, outData->sunVector.z);

    return true;
}
