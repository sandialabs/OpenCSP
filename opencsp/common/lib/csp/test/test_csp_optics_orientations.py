import copy
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.RegionXY import RegionXY, Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestCSPOpticsOrientation:
    """Tests that roations applied to CSP Optics add as expected. The optic heirarchy
    is defined as Mirror -> Facet -> FacetEnsemble -> Heliostat -> SolarField.
    Any of these could be placed into a scene instead of a larger object. Optics are created with spatial
    transformations at different levels in their heirarchy. Orientation of the mirror
    are checked at different levels in the heirarchy.

    These tests all set the relative rotations between some of these objects and then look at how
    they affect the overall transformations. Rotation `r5` is not used anymore.
    """

    def _generate_optics_rotation(self, r1: Rotation, r2: Rotation, r3: Rotation, r4: Rotation, r5: Rotation):
        # Define delta movement
        dv = Vxyz((0, 0, 0))

        # Define pointing function
        def child_to_parent(r):
            return TransformXYZ.from_R_V(r, dv)

        t1 = child_to_parent(r1)
        t2 = child_to_parent(r2)
        t3 = child_to_parent(r3)
        t4 = child_to_parent(r4)
        t5 = child_to_parent(r5)

        # define optics
        shape = RegionXY.rectangle((2, 4))
        mirror = MirrorParametric.generate_flat(shape)

        facet = Facet(mirror)

        ensemble = FacetEnsemble([facet])

        heliostat = HeliostatAzEl(ensemble)

        scene = Scene()
        scene.add_object(heliostat)

        # Position optics
        mirror._self_to_parent_transform = t1
        facet._self_to_parent_transform = t2
        ensemble._self_to_parent_transform = t3
        heliostat._self_to_parent_transform = t4

        # return object references
        return mirror, facet, ensemble, heliostat

    def _check_rotation(
        self,
        mirror: MirrorAbstract,
        facet: Facet,
        ensemble: FacetEnsemble,
        heliostat: HeliostatAzEl,
        a1: float,
        a2: float,
        a3: float,
        a4: float,
    ):
        # Test
        UP = Vxyz([0, 0, 1])
        resolution = Resolution.pixelX(1)
        norm_0 = mirror.self_to_global_tranformation.apply(UP)  # facet
        norm_1 = facet.self_to_global_tranformation.apply(UP)  # ensemble
        norm_2 = ensemble.self_to_global_tranformation.apply(UP)  # heliostat
        norm_3 = heliostat.self_to_global_tranformation.apply(UP)  # scene

        np.testing.assert_almost_equal(norm_0.x[0], a1, 4)
        np.testing.assert_almost_equal(norm_0.y[0], a1, 4)

        np.testing.assert_almost_equal(norm_1.x[0], a2, 4)
        np.testing.assert_almost_equal(norm_1.y[0], a2, 4)

        np.testing.assert_almost_equal(norm_2.x[0], a3, 4)
        np.testing.assert_almost_equal(norm_2.y[0], a3, 4)

        np.testing.assert_almost_equal(norm_3.x[0], a4, 4)
        np.testing.assert_almost_equal(norm_3.y[0], a4, 4)

    def test_identity_rotation_0(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.identity()
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_1(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.identity()
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.001
        a2 = 0
        a3 = 0
        a4 = 0

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_2(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.002
        a2 = 0.001
        a3 = 0.000
        a4 = 0.000

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_rotation_2(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.001
        a2 = 0.001
        a3 = 0.000
        a4 = 0.000

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_3(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.003
        a2 = 0.002
        a3 = 0.001
        a4 = 0.000

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_rotation_3(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.identity()
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.001
        a2 = 0.001
        a3 = 0.001
        a4 = 0.000

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_4(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.004
        a2 = 0.003
        a3 = 0.002
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)

    def test_rotation_4(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.identity()
        r3 = Rotation.identity()
        r4 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r5 = Rotation.identity()
        mirror, facet, ensemble, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0.001
        a2 = 0.001
        a3 = 0.001
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, ensemble, heliostat, a1, a2, a3, a4)


if __name__ == "__main__":
    test = TestCSPOpticsOrientation()

    test.test_rotation_2()
