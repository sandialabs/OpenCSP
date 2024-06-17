from dataclasses import dataclass

from opencsp.common.lib.tool import hdf5_tools


@dataclass
class ParamsMaskCalculation(hdf5_tools.HDF5_IO_Abstract):
    """Parameters for calculating mask"""

    mask_hist_thresh: float = 0.5
    """Defines threshold to use when calculating optic mask. Uses a histogram of pixel values
    of the mask difference image (light image - dark image). This is the fraction of the way
    from the first histogram peak (most common dark pixel value) to the the last histogram peak
    (most common light pixel value)."""
    mask_filt_width: int = 9
    """Side length of square kernel used to filter mask image"""
    mask_filt_thresh: int = 4
    """Threshold (minimum number of active pixels) to use when removing small active mask areas."""
    mask_thresh_active_pixels: float = 0.05
    """If number of active mask pixels is below this fraction of total image pixels, thow error."""
    mask_keep_largest_area: bool = False
    """Flag to apply processing step that keeps only the largest mask area"""

    def save_to_hdf(self, file: str, prefix: str = ''):
        """Saves data to given HDF5 file. Data is stored in PREFIX + ParamsMaskCalculation/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.mask_hist_thresh,
            self.mask_filt_width,
            self.mask_filt_thresh,
            self.mask_thresh_active_pixels,
            self.mask_keep_largest_area,
        ]
        datasets = [
            prefix + 'ParamsMaskCalculation/mask_hist_thresh',
            prefix + 'ParamsMaskCalculation/mask_filt_width',
            prefix + 'ParamsMaskCalculation/mask_filt_thresh',
            prefix + 'ParamsMaskCalculation/mask_thresh_active_pixels',
            prefix + 'ParamsMaskCalculation/mask_keep_largest_area',
        ]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ''):
        """Loads data from given file. Assumes data is stored as: PREFIX + ParamsMaskCalculation/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """

        # Load sofast parameters
        datasets = [
            prefix + 'ParamsMaskCalculation/mask_hist_thresh',
            prefix + 'ParamsMaskCalculation/mask_filt_width',
            prefix + 'ParamsMaskCalculation/mask_filt_thresh',
            prefix + 'ParamsMaskCalculation/mask_thresh_active_pixels',
            prefix + 'ParamsMaskCalculation/mask_keep_largest_area',
        ]
        data = hdf5_tools.load_hdf5_datasets(datasets, file)

        return cls(**data)
