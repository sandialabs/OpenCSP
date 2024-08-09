"""Measurement class for SofastFringe
"""

from abc import ABC
import datetime as dt

import opencsp.app.sofast.lib.DistanceOpticScreen as sod
import opencsp.common.lib.tool.hdf5_tools as h5


class AbstractMeasurementSofast(h5.HDF5_IO_Abstract, ABC):
    """Sofast measurement (fringe, fixed, etc) data class that contains captured images
    and metadata about the measurement.

    By abstracting out the common parts of Fringe and Fixed measurements, we allow the OpticScreenDistance to be
    determined once and used for many measurements. Having a common interface also reduces the burden on the programmer
    to learn how to use each sofast mode.
    """

    def __init__(
        self, dist_optic_screen_measure: sod.DistanceOpticScreen, date: dt.datetime, name: str
    ) -> 'AbstractMeasurementSofast':
        """
        Parameters
        ----------
        dist_optic_screen_measure : MeasurementDistance
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
        self.dist_optic_screen_measure = dist_optic_screen_measure
        self.date = date
        self.name = name

    def __repr__(self) -> str:
        # "AbstractMeasurementSofast: { name }"
        cls_name = type(self).__name__
        return cls_name + ': { ' + self.name + ' }'

    @property
    def v_measure_point_facet(self):
        """Convenience method for accessing dist_optic_screen_measure.v_measure_point_facet"""
        return self.dist_optic_screen_measure.v_measure_point_facet

    @property
    def dist_optic_screen(self):
        """Convenience method for accessing dist_optic_screen_measure.dist_optic_screen"""
        return self.dist_optic_screen_measure.dist_optic_screen

    @classmethod
    def _load_from_hdf(cls, file: str, prefix: str = '') -> dict[str, any]:
        # Load grid data
        datasets = [prefix + 'date', prefix + 'name']
        kwargs = h5.load_hdf5_datasets(datasets, file)

        kwargs['dist_optic_screen_measure'] = sod.DistanceOpticScreen.load_from_hdf(file, prefix)
        kwargs['date'] = dt.datetime.fromisoformat(kwargs['date'])

        return kwargs

    def _save_to_hdf(self, file: str, prefix: str = '') -> None:
        # NOTE: Collection date is saved as string in iso-format.
        datasets = [prefix + 'date', prefix + 'name']
        data = [self.date.isoformat(), self.name]

        # Save data
        h5.save_hdf5_datasets(data, datasets, file)
        self.dist_optic_screen_measure.save_to_hdf(file, prefix)
