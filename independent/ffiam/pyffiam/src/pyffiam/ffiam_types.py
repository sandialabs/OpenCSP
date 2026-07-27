"""Core enums: CSP sites, aim strategies, and coordinate directions."""

import logging
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class CspSite(Enum):
    """Concentrated Solar Power site identifiers.

    Values must match the C++ csp_site enum and UE ECspSite.
    """

    Custom = 0
    NSTTF = 1
    RadialSmall = 2  # generated ~1 km radial field
    Radial = 3  # generated ~1.6 km radial field
    SampleV1 = 4
    SampleV2 = 5
    SampleV3 = 6


class FieldLayout(Enum):
    """Field generation type; mirrors C++ field_layout_type."""

    Grid = 0  # square grid
    Radial = 1  # concentric staggered rings

    @staticmethod
    def from_str(s: str) -> "FieldLayout":
        return {"grid": FieldLayout.Grid, "radial": FieldLayout.Radial}.get(
            (s or "grid").strip().lower(), FieldLayout.Grid
        )


class AimType(Enum):
    """Heliostat aiming strategy types.

    For tracking heliostats (Point, Ring, SplitRing, Vector, CsvData):
        Heliostats tilt to redirect sunlight toward the specified aim direction.

    For fixed-orientation heliostats (FixedNormal):
        Heliostat surface normal is fixed (e.g., STOW/face-up position).
        The reflected beam direction depends on sun position via law of reflection.
    """

    Null = 0
    Point = 1
    Ring = 2
    SplitRing = 3
    Vector = 4
    CsvData = 5
    FixedNormal = 6


class Direction(Enum):
    """Cartesian coordinate directions."""

    East = 0
    North = 1
    Up = 2


# Site name mappings for string parsing
_SITE_MAPPINGS: dict[str, CspSite] = {
    'nsttf': CspSite.NSTTF,
    'radialsmall': CspSite.RadialSmall,
    'radialsm': CspSite.RadialSmall,
    'radial': CspSite.Radial,
    'samplev1': CspSite.SampleV1,
    'sample v1': CspSite.SampleV1,
    'samplev2': CspSite.SampleV2,
    'sample v2': CspSite.SampleV2,
    'samplev3': CspSite.SampleV3,
    'sample v3': CspSite.SampleV3,
    'custom': CspSite.Custom,
}

# Aim type name mappings for string parsing
_AIM_MAPPINGS: dict[str, AimType] = {
    'point': AimType.Point,
    'ring': AimType.Ring,
    'splitring': AimType.SplitRing,
    'split ring': AimType.SplitRing,
    'vector': AimType.Vector,
    'vec': AimType.Vector,
    'csv': AimType.CsvData,
    'csvdata': AimType.CsvData,
    'null': AimType.Null,
    'fixednormal': AimType.FixedNormal,
    'fixed_normal': AimType.FixedNormal,
    'fixed': AimType.FixedNormal,
    'stow': AimType.FixedNormal,
}


def get_csp_site_from_string(site_str: str, strict: bool = False) -> CspSite:
    """Parse a site name string to CspSite. Returns Custom if unrecognised (strict=False)."""
    site_lower = site_str.lower().strip()

    if site_lower in _SITE_MAPPINGS:
        return _SITE_MAPPINGS[site_lower]

    # substring fallback for backwards compatibility
    if 'nsttf' in site_lower:
        return CspSite.NSTTF
    elif 'radial' in site_lower and 'sm' in site_lower:
        return CspSite.RadialSmall
    elif 'radial' in site_lower:
        return CspSite.Radial
    elif 'v1' in site_lower:
        return CspSite.SampleV1
    elif 'v2' in site_lower:
        return CspSite.SampleV2
    elif 'v3' in site_lower:
        return CspSite.SampleV3

    if strict:
        valid_sites = ', '.join(sorted(_SITE_MAPPINGS.keys()))
        raise ValueError(f"Unknown site: '{site_str}'. Valid sites: {valid_sites}")

    log.warning(f"Unknown site string '{site_str}', defaulting to Custom")
    return CspSite.Custom


def get_aim_type_from_string(aim_type_str: str, strict: bool = False) -> AimType:
    """Parse an aim type string to AimType. Returns Point if unrecognised (strict=False)."""
    aim_lower = aim_type_str.lower().strip()

    if aim_lower in _AIM_MAPPINGS:
        return _AIM_MAPPINGS[aim_lower]

    # substring fallback for backwards compatibility
    if 'point' in aim_lower:
        return AimType.Point
    elif 'splitring' in aim_lower:
        return AimType.SplitRing
    elif 'ring' in aim_lower:
        return AimType.Ring
    elif 'vec' in aim_lower:
        return AimType.Vector
    elif 'csv' in aim_lower:
        return AimType.CsvData
    elif 'fixed' in aim_lower or 'stow' in aim_lower:
        return AimType.FixedNormal

    if strict:
        valid_types = ', '.join(sorted(_AIM_MAPPINGS.keys()))
        raise ValueError(f"Unknown aim type: '{aim_type_str}'. Valid types: {valid_types}")

    log.warning(f"Unknown aim type string '{aim_type_str}', defaulting to Point")
    return AimType.Point
