from dataclasses import dataclass

import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


@dataclass
class ParamsOpticGeometry:
    """Parameter dataclass for processing optic geometry"""

    perimeter_refine_axial_search_dist: float = 50.0
    perimeter_refine_perpendicular_search_dist: float = 50.0
    facet_corns_refine_step_length: float = 10.0
    facet_corns_refine_perpendicular_search_dist: float = 10.0
    facet_corns_refine_frac_keep: float = 0.5

    def save_to_hdf(self, file: str, prefix: str = ''):
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
            prefix + 'ParamsOpticGeometry/perimeter_refine_axial_search_dist',
            prefix + 'ParamsOpticGeometry/perimeter_refine_perpendicular_search_dist',
            prefix + 'ParamsOpticGeometry/facet_corns_refine_step_length',
            prefix + 'ParamsOpticGeometry/facet_corns_refine_perpendicular_search_dist',
            prefix + 'ParamsOpticGeometry/facet_corns_refine_frac_keep',
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
