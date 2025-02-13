from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as h5


class SpatialOrientation(h5.HDF5_IO_Abstract):
    """Holds relative orientations of camera, screen, and optic for deflectometry systems"""

    def __init__(self, r_cam_screen: Rotation, v_cam_screen_cam: Vxyz) -> "SpatialOrientation":
        """Instantiates Spatial Orienatation object

        Parameters
        ----------
        r_cam_screen : Rotation
            Rotation
        v_cam_screen_cam : Vxyz
            Vector
        """
        # Orient screen to camera
        self._orient_screen_cam(r_cam_screen, v_cam_screen_cam)

        # Camera-optic
        self.optic_oriented = False

        self.r_cam_optic: Rotation
        self.v_cam_optic_cam: Vxyz
        self.v_cam_optic_optic: Vxyz

        self.r_optic_cam: Rotation
        self.v_optic_cam_cam: Vxyz
        self.v_optic_cam_optic: Vxyz

        self.trans_cam_optic: TransformXYZ

        # Screen-optic
        self.r_optic_screen: Rotation
        self.v_optic_screen_optic: Vxyz
        self.v_optic_screen_screen: Vxyz

        self.r_screen_optic: Rotation
        self.v_screen_optic_optic: Vxyz
        self.v_screen_optic_screen: Vxyz

        self.trans_screen_optic: TransformXYZ

        # Camera-screen
        self.r_cam_screen: Rotation
        self.v_cam_screen_cam: Vxyz
        self.v_cam_screen_screen: Vxyz

        self.r_screen_cam: Rotation
        self.v_screen_cam_cam: Vxyz
        self.v_screen_cam_screen: Vxyz

        self.trans_screen_cam: TransformXYZ

    def __copy__(self) -> "SpatialOrientation":
        """Returns a copy of spatial orientation"""
        r_cam_screen = Rotation.from_rotvec(self.r_cam_screen.as_rotvec().copy())
        v_cam_screen_cam = self.v_cam_screen_cam.copy()
        ori = SpatialOrientation(r_cam_screen, v_cam_screen_cam)

        if self.optic_oriented:
            r_cam_optic = Rotation.from_rotve(self.r_cam_optic.as_rotvec().copy())
            v_cam_optic_cam = self.v_cam_optic_cam.copy()
            ori.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

        return ori

    def _orient_screen_cam(self, r_cam_screen: Rotation, v_cam_screen_cam: Vxyz) -> None:
        """Orients the screen and camera

        Parameters
        ----------
        r_cam_screen : Rotation
            Rotation
        v_cam_screen_cam : Vxyz
            Vector
        """
        # Display-camera orientation
        self.r_cam_screen = r_cam_screen
        self.r_screen_cam = r_cam_screen.inv()

        self.v_cam_screen_cam = v_cam_screen_cam
        self.v_screen_cam_cam = -v_cam_screen_cam

        self.v_cam_screen_screen = v_cam_screen_cam.rotate(r_cam_screen)
        self.v_screen_cam_screen = -self.v_cam_screen_screen

        self.trans_screen_cam = TransformXYZ.from_R_V(self.r_screen_cam, self.v_cam_screen_cam)

    def orient_optic_cam(self, r_cam_optic: Rotation, v_cam_optic_cam: Vxyz) -> None:
        """Orients the optic and camera, and thus the optic and screen

        Parameters
        ----------
        r_cam_optic : Rotation
            Rotation
        v_cam_optic_cam : Vxyz
            Vector
        """
        self.optic_oriented = True

        self.r_cam_optic = r_cam_optic
        self.r_optic_cam = r_cam_optic.inv()

        self.v_cam_optic_cam = v_cam_optic_cam
        self.v_optic_cam_cam = -v_cam_optic_cam

        self.v_cam_optic_optic = v_cam_optic_cam.rotate(r_cam_optic)
        self.v_optic_cam_optic = -self.v_cam_optic_optic

        self.trans_cam_optic = TransformXYZ.from_R_V(self.r_cam_optic, self.v_optic_cam_optic)

        self._orient_optic_screen()

    def _orient_optic_screen(self) -> None:
        """Orients the optic and screen (must be called last)"""
        # Optic-screen orientation
        self.r_optic_screen = self.r_cam_screen * self.r_optic_cam
        self.r_screen_optic = self.r_optic_screen.inv()

        self.v_optic_screen_optic = self.v_optic_cam_optic + self.v_cam_screen_cam.rotate(self.r_cam_optic)
        self.v_screen_optic_optic = -self.v_optic_screen_optic

        self.v_optic_screen_screen = self.v_optic_screen_optic.rotate(self.r_optic_screen)
        self.v_screen_optic_screen = -self.v_optic_screen_screen

        self.trans_screen_optic = TransformXYZ.from_R_V(self.r_screen_optic, self.v_optic_screen_optic)

    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        """Saves only camera-screen orientation data to HDF file. Data is stored as prefix + SpatialOrientation/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        if self.optic_oriented:
            datasets = [
                prefix + "SpatialOrientation/r_cam_screen",
                prefix + "SpatialOrientation/v_cam_screen_cam",
                prefix + "SpatialOrientation/optic_oriented",
                prefix + "SpatialOrientation/r_cam_optic",
                prefix + "SpatialOrientation/v_cam_optic_cam",
            ]
            data = [
                self.r_cam_screen.as_rotvec(),
                self.v_cam_screen_cam.data,
                self.optic_oriented,
                self.r_cam_optic.as_rotvec(),
                self.v_cam_optic_cam.data,
            ]
        else:
            datasets = [
                prefix + "SpatialOrientation/r_cam_screen",
                prefix + "SpatialOrientation/v_cam_screen_cam",
                prefix + "SpatialOrientation/optic_oriented",
            ]
            data = [self.r_cam_screen.as_rotvec(), self.v_cam_screen_cam.data, self.optic_oriented]

        h5.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = "") -> "SpatialOrientation":
        """Loads data from given file. Assumes data is stored as: PREFIX + SpatialOrientation/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        # Load screen-camera orientation information
        datasets = [
            prefix + "SpatialOrientation/r_cam_screen",
            prefix + "SpatialOrientation/v_cam_screen_cam",
            prefix + "SpatialOrientation/optic_oriented",
        ]
        data = h5.load_hdf5_datasets(datasets, file)
        r_cam_screen = Rotation.from_rotvec(data["r_cam_screen"])
        v_cam_screen_cam = Vxyz(data["v_cam_screen_cam"])
        ori = cls(r_cam_screen, v_cam_screen_cam)

        # If optic is oriented, load optic orientation information
        if data["optic_oriented"]:
            datasets = [prefix + "SpatialOrientation/r_cam_optic", prefix + "SpatialOrientation/v_cam_optic_cam"]
            data = h5.load_hdf5_datasets(datasets, file)
            r_cam_optic = Rotation.from_rotvec(data["r_cam_optic"])
            v_cam_optic_cam = Vxyz(data["v_cam_optic_cam"])
            ori.orient_optic_cam(r_cam_optic, v_cam_optic_cam)

        return ori
