# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Dataclasses for computed FFIAM analysis results, populated after the C++ analysis completes."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pyffiam import utils


@dataclass
class HeliostatResults:
    """Results related to heliostat positions and orientations."""
    locations: np.ndarray           # (num_heliostats, 3) positions
    aim_vectors: np.ndarray         # (num_heliostats, 3) aim directions
    movement_angles: np.ndarray     # (num_heliostats,) rotation angles
    facet_origins: np.ndarray       # (num_heliostats, num_facets, 3) facet positions
    total_movement: float           # Total heliostat movement (degrees)


@dataclass
class IrradianceResults:
    """Results related to irradiance calculations."""
    values: np.ndarray              # (num_voxels,) irradiance per voxel
    total: int                      # Sum of all irradiance values
    peak: float                     # Maximum irradiance value
    num_impacted: int               # Number of voxels with non-zero irradiance

    @classmethod
    def from_array(cls, irrads: np.ndarray) -> 'IrradianceResults':
        """Build IrradianceResults by computing summary stats from a raw irradiance array."""
        return cls(
            values=irrads,
            total=int(np.nansum(irrads)),
            peak=float(np.nanmax(irrads)),
            num_impacted=int(np.nansum(irrads > 0)),
        )


@dataclass
class ThresholdResults:
    """Results filtered by irradiance threshold."""
    mask: np.ndarray                # Boolean mask where irrad > threshold
    irradiances: np.ndarray         # Irradiance values above threshold
    voxel_ids: np.ndarray           # Indices of threshold-exceeding voxels
    voxel_locations: np.ndarray     # (N, 3) Cartesian positions of threshold voxels
    total_irradiance: int           # Sum of above-threshold irradiance
    num_voxels: int                 # Count of threshold-exceeding voxels
    has_glare: bool                 # True if any voxel exceeds threshold

    @classmethod
    def from_irradiance(
        cls,
        irrads: np.ndarray,
        threshold: float,
        voxel_size: int,
        field_radius: int,
        min_height: int,
    ) -> 'ThresholdResults':
        """Filter irradiance array to above-threshold voxels and compute their positions."""
        mask = irrads > threshold
        threshold_irrads = irrads[mask]
        voxel_ids = np.where(mask)[0]
        voxel_locs = utils.get_voxel_locs_from_indexes(
            voxel_ids, voxel_size, field_radius, min_height
        )

        return cls(
            mask=mask,
            irradiances=threshold_irrads,
            voxel_ids=voxel_ids,
            voxel_locations=voxel_locs,
            total_irradiance=int(np.nansum(threshold_irrads)),
            num_voxels=int(np.nansum(mask)),
            has_glare=int(np.nansum(mask)) > 0,
        )


@dataclass
class AnalysisResults:
    """All computed results from a single FFIAM analysis.

    Coordinates (x, y, z) = (east, north, up) with tower at origin. Units: meters for distances,
    kW/m² for irradiance values.
    """
    heliostats: HeliostatResults    # Heliostat positions, aim vectors, and tracking angles
    irradiance: IrradianceResults   # Full volumetric irradiance array and summary stats (total, peak, impacted voxel count)
    threshold: ThresholdResults     # Subset of irradiance data filtered to voxels exceeding the threshold

    # Voxel grid (derived from AnalysisConfig, stored here for convenience)
    voxel_locations: np.ndarray     # (num_voxels, 3) center position of every voxel (m)
    voxel_layout: tuple             # (nx, ny, nz) grid dimensions along east/north/up axes

    @property
    def has_results(self) -> bool:
        return self.irradiance is not None

    @property
    def has_glare(self) -> bool:
        return self.threshold.has_glare if self.threshold else False

    @property
    def total_irrad(self) -> int:
        return self.irradiance.total

    @property
    def peak_irrad(self) -> float:
        return self.irradiance.peak

    @property
    def n_glaring_voxels(self) -> int:
        return self.threshold.num_voxels

    @classmethod
    def from_raw_data(
        cls,
        helio_locs: np.ndarray,
        helio_aim_vs: np.ndarray,
        helio_angles: np.ndarray,
        facet_origins: np.ndarray,
        irrads: np.ndarray,
        movement: float,
        voxel_locs: np.ndarray,
        voxel_layout: tuple,
        threshold: float,
        voxel_size: int,
        field_radius: int,
        min_height: int,
    ) -> 'AnalysisResults':
        """Build from raw C++ arrays, computing all derived metrics."""
        heliostats = HeliostatResults(
            locations=helio_locs,
            aim_vectors=helio_aim_vs,
            movement_angles=helio_angles,
            facet_origins=facet_origins,
            total_movement=movement,
        )

        irradiance = IrradianceResults.from_array(irrads)

        threshold_results = ThresholdResults.from_irradiance(
            irrads=irrads,
            threshold=threshold,
            voxel_size=voxel_size,
            field_radius=field_radius,
            min_height=min_height,
        )

        return cls(
            heliostats=heliostats,
            irradiance=irradiance,
            threshold=threshold_results,
            voxel_locations=voxel_locs,
            voxel_layout=voxel_layout,
        )
