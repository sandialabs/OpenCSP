#include "api/preset_sites.h"
#include "fmt/core.h"
#include <iostream>

// Maps analysis version (1/2/3) to field radius, altitude bounds, and voxel size.
void get_version_params(const int version, int* radius, int* zMin, int* zMax, int* voxelSize)
{
    if (version == 1) {
        *radius = 600;
        *zMin = 4;
        *zMax = 100;
        *voxelSize = 2;
    }
    else if (version == 2) {
        *radius = 1000;
        *zMin = 4;
        *zMax = 200;
        *voxelSize = 2;
    }
    else {
        *radius = 1600;
        *zMin = 4;
        *zMax = 300;
        *voxelSize = 2;
    }
}

// Fills config with hardcoded geometry and file paths for the requested site; returns false for unknown IDs.
bool get_preset_site_config(int siteId, const int aimId, preset_site_config* config, std::string* aimFile)
{
    // Default location is NSTTF
    config->coords = {34.96348f, -106.50964f};
    config->timezone = -7.0f;
    config->towerHeight = 61.0f;
    config->helioFile = "";
    config->nHelios = 0;
    config->facetFile = "";
    config->nFacets = 25;
    config->nFacetCols = 5;
    config->facetDims = {1.2f, 1.2f};
    config->radius = 600;
    config->layout = layout_grid;
    *aimFile = "";

    auto site = static_cast<csp_site>(siteId);
    if (site == site_null) site = site_nsttf;

    if (site == site_nsttf)
    {
        std::cout << fmt::format("Using SNL NSTTF field layout\n");
        config->nFacets = 25;
        config->nFacetCols = 5;
        config->facetFile = "data/NSTTF_Facet_Centroids.csv";
        config->helioFile = "data/NSTTF_Heliostats_origin_at_torque_tube.csv";
        config->nHelios = 218;
        config->towerHeight = 61.0f;

        // aim_data_csv uses the bundled NSTTF aim file; other strategies are computed.
        if (aimId == aim_data_csv) {
            *aimFile = "data/NSTTF_Aim_6on_East20.csv";
        }
    }
    else if (site == site_radialSmall)
    {
        // Generated ~1 km radial field (no CSVs; geometry built at run time).
        std::cout << fmt::format("Using generic RADIAL (1 km) field layout\n");
        config->coords.x = 35.0f;       // generic location
        config->coords.y = -115.0f;
        config->timezone = -8;
        config->towerHeight = 100.0f;

        config->nFacets = 25;
        config->nFacetCols = 5;
        config->facetDims.x = 1.8f;
        config->facetDims.y = 1.8f;
        config->facetFile = "";
        config->helioFile = "";
        config->nHelios = 6400;
        config->radius = 1000;
        config->layout = layout_radial;
    }
    else if (site == site_radial)
    {
        // Generated ~1.6 km radial field.
        std::cout << fmt::format("Using generic RADIAL field layout\n");
        config->coords.x = 35.0f;
        config->coords.y = -115.0f;
        config->timezone = -8;
        config->towerHeight = 100.0f;

        config->nFacets = 25;
        config->nFacetCols = 5;
        config->facetDims.x = 1.8f;
        config->facetDims.y = 1.8f;
        config->facetFile = "";
        config->helioFile = "";
        config->nHelios = 10000;
        config->radius = 1600;
        config->layout = layout_radial;
    }
    else if (site == site_sampleV1)
    {
        std::cout << fmt::format("Using Sample V1 field layout\n");
        config->nHelios = 44 * 44; // V1 requires 2k
        config->nFacets = 25;
        config->nFacetCols = 5;
    }
    else if (site == site_sampleV2)
    {
        std::cout << fmt::format("Using Sample V2 field layout\n");
        config->nHelios = 80 * 80; // > 6k
    }
    else if (site == site_sampleV3)
    {
        std::cout << fmt::format("Using Sample V3 field layout\n");
        config->nHelios = 105 * 105; // > 11k
    }
    else
    {
        return false;
    }

    return true;
}
