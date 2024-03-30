import datetime as dt

import numpy as np

import opencsp.app.sofast.lib.AbstractMeasurementSofast as ams
import opencsp.app.sofast.lib.OpticScreenDistance as osd
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


class MeasurementSofastFixed(ams.AbstractMeasurementSofast):
    """Fixed pattern measuremnt class. Stores measurement data. Can save
    and load to HDF file format.
    """

    def __init__(
        self,
        image: np.ndarray,
        optic_screen_dist_measure: osd.OpticScreenDistance,
        origin: Vxy,
        date: dt.datetime = None,
        name: str = '',
    ):
        """Saves measurement data in class.

        Parameters
        ----------
        image : np.ndarray
            (M, N) ndarray, measurement image
        v_measure_point_facet : Vxyz
            Location of measurem point on facet, meters
        dist_optic_screen : float
            Optic to screen distance, meters
        origin : Vxy
            The centroid of the origin dot, pixels
        date : datetime, optional
            Collection date/time. Default is dt.datetime.now()
        name : str, optional
            Name or serial number of measurement. Default is empty string ''
        """
        super().__init__(optic_screen_dist_measure, date, name)
        self.image = image
        self.origin = origin
        self.date = date
        self.name = name

    @classmethod
    def load_from_hdf(cls, file: str, prefix='') -> 'MeasurementSofastFixed':
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF file to load

        """
        # Load grid data
        datasets = [prefix + 'MeasurementSofastFixed/image', prefix + 'MeasurementSofastFixed/origin']
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)
        kwargs.update(super()._load_from_hdf(file, prefix + 'MeasurementSofastFixed'))

        kwargs['origin'] = Vxy(kwargs['origin'])

        return cls(**kwargs)

    def save_to_hdf(self, file: str, prefix='') -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF file to save
        """
        datasets = [prefix + 'MeasurementSofastFixed/image', prefix + 'MeasurementSofastFixed/origin']
        data = [self.image, self.origin.data.squeeze()]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
        super()._save_to_hdf(file, prefix + 'MeasurementSofastFixed')
