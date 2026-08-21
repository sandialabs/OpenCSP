# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Immutable configuration dataclasses for FFIAM analysis inputs."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np

from pyffiam.ffiam_types import CspSite, AimType


@dataclass(frozen=True)
class TimeConfig:
    """Date and time configuration for analysis."""

    year: int
    month: int
    day: int
    hour: int

    @property
    def datetime(self) -> datetime:
        return datetime(self.year, self.month, self.day, self.hour)

    @property
    def datetime_str(self) -> str:
        return self.datetime.strftime('%Y%m%d%H%M')


@dataclass(frozen=True)
class LocationConfig:
    """Geographic location configuration."""

    latitude: float
    longitude: float
    timezone: float  # UTC offset


@dataclass(frozen=True)
class FieldConfig:
    """CSP field configuration."""

    radius: int  # Field radius in meters
    num_heliostats: int  # Number of heliostats
    min_height: int  # Minimum analysis height (m)
    max_height: int  # Maximum analysis height (m)
    tower_height: float  # Receiver tower height (m)


@dataclass(frozen=True)
class HeliostatConfig:
    """Heliostat design parameters."""

    num_facets: int  # Total facets per heliostat
    num_facet_cols: int  # Facet columns per heliostat
    facet_width: float  # Individual facet width (m)
    facet_height: float  # Individual facet height (m)
    heliostat_file: str = ""  # CSV file with heliostat positions
    facet_file: str = ""  # CSV file with facet positions

    @property
    def heliostat_width(self) -> float:
        return self.facet_width * self.num_facet_cols


@dataclass(frozen=True)
class AimConfig:
    """Aiming strategy configuration."""

    strategy: AimType
    parameters: np.ndarray  # Strategy-specific parameters [3]
    aim_file: str = ""  # Optional CSV file with per-heliostat aim data

    class Config:
        # Allow numpy arrays in frozen dataclass
        arbitrary_types_allowed = True


@dataclass(frozen=True)
class VoxelConfig:
    """Voxel grid configuration."""

    size: int = 2  # Voxel edge length in meters

    def compute_layout(self, field_radius: int, min_height: int, max_height: int) -> tuple:
        """Return (nx, ny, nz) voxel grid dimensions for the given field and height range."""
        num_x = int(2 * field_radius / self.size)
        num_y = int(2 * field_radius / self.size)
        num_z = int((max_height - min_height) / self.size + 1)
        return (num_x, num_y, num_z)

    def compute_num_voxels(self, field_radius: int, min_height: int, max_height: int) -> int:
        """Return total voxel count for the given field and height range."""
        layout = self.compute_layout(field_radius, min_height, max_height)
        return layout[0] * layout[1] * layout[2]

    @property
    def area(self) -> float:
        return self.size**2

    @property
    def volume(self) -> float:
        return self.size**3


@dataclass(frozen=True)
class OpticalConfig:
    """Optical and environmental parameters."""

    reflectivity: float = 0.9  # Heliostat reflectivity
    peak_dni: float = 0.1  # Peak direct normal irradiance (W/cm²)
    sun_angle: float = 0.0093  # Solar disk angle (radians)
    slope_error: float = 0.0012  # Mirror slope error (radians)
    beta: float = 0.0094  # Beam spread coefficient


@dataclass
class AnalysisConfig:
    """Aggregates all configuration sub-classes for a single analysis run.

    Acts as a structured alternative to the flat kwargs accepted by analysis().
    Build directly or use the from_params() convenience constructor.
    """

    # Identification
    name: str  # Display name for this analysis run
    site: CspSite  # Preset site enum; CspSite.Custom requires all sub-configs to be filled manually

    # Sub-configurations (each groups a logical set of parameters)
    time: TimeConfig  # Date and decimal hour
    location: LocationConfig  # Lat/lng/timezone
    field: FieldConfig  # Field radius, heliostat count, height bounds, tower height
    heliostat: HeliostatConfig  # Facet geometry and optional position CSV files
    aim: AimConfig  # Aiming strategy enum, parameter array, and optional aim CSV
    voxel: VoxelConfig  # Voxel edge length (m)
    optical: OpticalConfig  # Reflectivity, DNI, beam spread coefficient

    # Analysis parameters
    threshold: float  # Irradiance threshold for glare reporting (kW/m²); voxels below this are excluded in outputs

    # Path analysis (optional UAS exposure analysis)
    paths: Optional[List] = None  # List of polyline paths, each a sequence of (x, y, z) waypoints (m)
    path_speeds: Optional[List] = None  # Travel speed (m/s) for each path in `paths`

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def has_paths(self) -> bool:
        return self.paths is not None and len(self.paths) > 0

    @property
    def slug(self) -> str:
        from pyffiam.utils import slug

        return slug(self.name)

    @property
    def created_at_str(self) -> str:
        return self.created_at.strftime("%Y%m%d%H%M")

    @classmethod
    def from_params(
        cls,
        name: str,
        site: CspSite,
        year: int,
        month: int,
        day: int,
        hour: int,
        latitude: float,
        longitude: float,
        timezone: float,
        field_radius: int,
        num_heliostats: int,
        min_height: int,
        max_height: int,
        tower_height: float,
        num_facets: int,
        num_facet_cols: int,
        facet_width: float,
        facet_height: float,
        aim_strategy: AimType,
        aim_parameters: np.ndarray,
        threshold: float,
        heliostat_file: str = "",
        facet_file: str = "",
        aim_file: str = "",
        voxel_size: int = 2,
        reflectivity: float = 0.9,
        peak_dni: float = 0.1,
        paths: Optional[List] = None,
        path_speeds: Optional[List] = None,
    ) -> 'AnalysisConfig':
        """Convenience constructor. Builds sub-configs from flat keyword args."""
        return cls(
            name=name,
            site=site,
            time=TimeConfig(year=year, month=month, day=day, hour=hour),
            location=LocationConfig(latitude=latitude, longitude=longitude, timezone=timezone),
            field=FieldConfig(
                radius=field_radius,
                num_heliostats=num_heliostats,
                min_height=min_height,
                max_height=max_height,
                tower_height=tower_height,
            ),
            heliostat=HeliostatConfig(
                num_facets=num_facets,
                num_facet_cols=num_facet_cols,
                facet_width=facet_width,
                facet_height=facet_height,
                heliostat_file=heliostat_file,
                facet_file=facet_file,
            ),
            aim=AimConfig(strategy=aim_strategy, parameters=aim_parameters, aim_file=aim_file),
            voxel=VoxelConfig(size=voxel_size),
            optical=OpticalConfig(reflectivity=reflectivity, peak_dni=peak_dni),
            threshold=threshold,
            paths=paths,
            path_speeds=path_speeds,
        )
