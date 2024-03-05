import datetime as dt

import numpy as np

from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


class FixedPatternMeasurement:
    """Fixed pattern measuremnt class. Stores measurement data. Can save
    and load to HDF file format.
    """

    def __init__(
        self,
        image: np.ndarray,
        v_measure_point_facet: Vxyz,
        dist_optic_screen: float,
        origin: Vxy,
        date: dt.datetime = dt.datetime.now(),
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
        date : datetime
            Collection date/time.
        name : str
            Name or serial number of measurement.
        """
        self.image = image
        self.v_measure_point_facet = v_measure_point_facet
        self.dist_optic_screen = dist_optic_screen
        self.origin = origin
        self.date = date
        self.name = name

    @classmethod
    def load_from_hdf(cls, file) -> 'FixedPatternMeasurement':
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF file to load

        """
        # Load grid data
        datasets = [
            'FixedPatternMeasurement/image',
            'FixedPatternMeasurement/v_measure_point_facet',
            'FixedPatternMeasurement/dist_optic_screen',
            'FixedPatternMeasurement/origin',
            'FixedPatternMeasurement/date',
            'FixedPatternMeasurement/name',
        ]
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)

        kwargs['v_measure_point_facet'] = Vxyz(kwargs['v_measure_point_facet'])
        kwargs['origin'] = Vxy(kwargs['origin'])
        kwargs['date'] = dt.datetime.fromisoformat(kwargs['date'])

        return cls(**kwargs)

    def save_to_hdf(self, file) -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF file to save

        NOTE: Collection date is saved as string in iso-format.
        """
        datasets = [
            'FixedPatternMeasurement/image',
            'FixedPatternMeasurement/v_measure_point_facet',
            'FixedPatternMeasurement/dist_optic_screen',
            'FixedPatternMeasurement/origin',
            'FixedPatternMeasurement/date',
            'FixedPatternMeasurement/name',
        ]
        data = [
            self.image,
            self.v_measure_point_facet.data.squeeze(),
            self.dist_optic_screen,
            self.origin.data.squeeze(),
            self.date.isoformat(),
            self.name,
        ]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
