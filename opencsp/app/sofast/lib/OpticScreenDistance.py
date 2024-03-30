from dataclasses import dataclass, field

from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as h5


@dataclass
class OpticScreenDistance(h5.HDF5_IO_Abstract):
    """Represents a distance measurement between the optic and the center of the screen. This measurement is typically
    achieved by displaying the crosshairs from the SofastGUI and measuring from the center of the optic
    (measurement_point 0,0,0) to the center of the crosshairs."""

    measure_point: Vxyz = field(default_factory=lambda: Vxyz(0.0, 0.0, 0.0))
    """ Location of measure point, meters. """
    optic_screen_dist: float = 0
    """ Optic-screen distance, meters. """

    def save_to_hdf(self, file: str, prefix: str) -> None:
        """
        Saves to HDF file
        """
        datasets = [prefix + '/measure_point', prefix + '/optic_screen_dist']
        data = [self.measure_point.data.squeeze(), self.optic_screen_dist]

        # Save data
        h5.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str):
        """
        Loads from HDF file

        """
        # Load grid data
        datasets = [prefix + '/measure_point', prefix + '/optic_screen_dist']
        kwargs = h5.load_hdf5_datasets(datasets, file)

        kwargs['measure_point'] = Vxyz(kwargs['measure_point'])

        return cls(**kwargs)
