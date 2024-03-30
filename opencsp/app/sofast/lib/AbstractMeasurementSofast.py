"""Measurement class for SofastFringe
"""

from abc import ABC, abstractmethod
import datetime as dt

import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
import opencsp.app.sofast.lib.OpticScreenDistance as sod
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as h5


class AbstractMeasurementSofast(h5.HDF5_IO_Abstract, ABC):
    """Sofast measurement (fringe, fixed, etc) data class that contains captured images
    and metadata about the measurement.

    By abstracting out the common parts of Fringe and Fixed measurements, we allow the OpticScreenDistance to be
    determined once and used for many measurements. Having a common interface also reduces the burden on the programmer
    to learn how to use each sofast mode.
    """

    def __init__(
        self, optic_screen_dist_measure: sod.OpticScreenDistance, date: dt.datetime, name: str
    ) -> 'AbstractMeasurementSofast':
        """
        Parameters
        ----------
        optic_screen_dist_measure : MeasurementDistance
            Optic-screen distance measurement.
        date : datetime
            Collection date/time.
        name : str
            Name or serial number of measurement.

        """
        # Get default values
        if date is None:
            date = dt.datetime.now()

        # Save input measurement data
        self.optic_screen_dist_measure = optic_screen_dist_measure
        self.date = date
        self.name = name

    def __repr__(self) -> str:
        # "AbstractMeasurementSofast: { name }"
        cls_name = type(self).__name__
        return cls_name + ': { ' + self.name + ' }'

    @property
    def measure_point(self):
        """Convenience method for accessing optic_screen_dist_measure.measure_point"""
        # Added as a property (1) for convenience, and (2) so that we don't need to go update everything
        return self.optic_screen_dist_measure.measure_point

    @property
    def optic_screen_dist(self):
        """Convenience method for accessing optic_screen_dist_measure.optic_screen_dist"""
        # Added as a property (1) for convenience, and (2) so that we don't need to go update everything
        return self.optic_screen_dist_measure.optic_screen_dist

    @classmethod
    def _load_from_hdf(cls, file: str, prefix: str) -> dict[str, any]:
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF file to load

        """
        # Load grid data
        datasets = [prefix + '/date', prefix + '/name']
        kwargs = h5.load_hdf5_datasets(datasets, file)

        kwargs['dist_optic_screen_measure'] = sod.DistanceOpticScreen.load_from_hdf(file, prefix)
        kwargs['date'] = dt.datetime.fromisoformat(kwargs['date'])

        return kwargs

    def _save_to_hdf(self, file: str, prefix: str) -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF file to save

        NOTE: Collection date is saved as string in iso-format.
        """
        datasets = [prefix + '/date', prefix + '/name']
        data = [self.date.isoformat(), self.name]

        # Save data
        h5.save_hdf5_datasets(data, datasets, file)
        self.optic_screen_dist_measure.save_to_hdf(file, prefix)
