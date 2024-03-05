from dataclasses import dataclass


@dataclass
class GeometryProcessingParams:
    """Parameter dataclass for processing optic geometry"""

    perimeter_refine_axial_search_dist: float = 50.0
    perimeter_refine_perpendicular_search_dist: float = 50.0
    facet_corns_refine_step_length: float = 10.0
    facet_corns_refine_perpendicular_search_dist: float = 10.0
    facet_corns_refine_frac_keep: float = 0.5
