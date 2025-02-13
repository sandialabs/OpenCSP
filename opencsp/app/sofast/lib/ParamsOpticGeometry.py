from dataclasses import dataclass

from opencsp.common.lib.tool import hdf5_tools


@dataclass
class ParamsOpticGeometry(hdf5_tools.HDF5_IO_Abstract):
    """Parameter dataclass for processing optic geometry"""

    perimeter_refine_axial_search_dist: float = 50.0
    """The length of the search box (along the search direction) to use when finding optic
    perimeter. Units pixels. (Default 50.0)"""
    perimeter_refine_perpendicular_search_dist: float = 50.0
    """The half-width of the search box (perpendicular to the search direction) to use when finding
    optic perimeter. Units pixels. (Default 50.0)"""
    facet_corns_refine_step_length: float = 10.0
    """The length of the search box (along the search direction) to use when refining facet corner
    locations (when processing a facet ensemble). Units pixels. (Default 10.0)"""
    facet_corns_refine_perpendicular_search_dist: float = 10.0
    """The half-width of the search box (perpendicular to the search direction) to use when
    refining facet corner locations (when processing a facet ensemble). Units pixels. (Default 10.0)"""
    facet_corns_refine_frac_keep: float = 0.5
    """The fraction of pixels to consider within search box when finding optic edges. (Default 0.5)"""

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + ParamsOpticGeometry/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.perimeter_refine_axial_search_dist,
            self.perimeter_refine_perpendicular_search_dist,
            self.facet_corns_refine_step_length,
            self.facet_corns_refine_perpendicular_search_dist,
            self.facet_corns_refine_frac_keep,
        ]
        datasets = [
            prefix + "ParamsOpticGeometry/perimeter_refine_axial_search_dist",
            prefix + "ParamsOpticGeometry/perimeter_refine_perpendicular_search_dist",
            prefix + "ParamsOpticGeometry/facet_corns_refine_step_length",
            prefix + "ParamsOpticGeometry/facet_corns_refine_perpendicular_search_dist",
            prefix + "ParamsOpticGeometry/facet_corns_refine_frac_keep",
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ""):
        """Loads data from given file. Assumes data is stored as: PREFIX + ParamsOpticGeometry/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        datasets = [
            prefix + "ParamsOpticGeometry/perimeter_refine_axial_search_dist",
            prefix + "ParamsOpticGeometry/perimeter_refine_perpendicular_search_dist",
            prefix + "ParamsOpticGeometry/facet_corns_refine_step_length",
            prefix + "ParamsOpticGeometry/facet_corns_refine_perpendicular_search_dist",
            prefix + "ParamsOpticGeometry/facet_corns_refine_frac_keep",
        ]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)
        return cls(**data)
