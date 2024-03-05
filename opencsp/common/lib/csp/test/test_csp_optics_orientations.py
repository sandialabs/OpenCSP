import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestCSPOpticsOrientation:
    """Tests that roations applied to CSP Optics add as expected. The optic heirarchy
    is defined as mirror -> facet -> Ensemble -> world. Optics are created with spatial
    transformations at different levels in their heirarchy. Orientation of the mirror
    are checked at different levels in the heirarchy.
    """

    def _generate_optics_rotation(
        self, r1: Rotation, r2: Rotation, r3: Rotation, r4: Rotation, r5: Rotation
    ):
        # Define delta movement
        dv = Vxyz((0, 0, 0))

        # Define pointing function
        def child_to_parent(r):
            return TransformXYZ.from_R_V(r, dv)

        # Define mirror
        shape = RegionXY.rectangle((2, 4))
        mirror = MirrorParametric.generate_flat(shape)
        mirror.set_position_in_space(dv, r1)

        # Define facet
        facet = Facet(mirror)
        facet.define_pointing_function(child_to_parent)
        facet.set_pointing(r2)
        facet.set_position_in_space(dv, r3)

        # Define facet ensemble
        heliostat = FacetEnsemble([facet])
        heliostat.define_pointing_function(child_to_parent)
        heliostat.set_pointing(r4)
        heliostat.set_position_in_space(dv, r5)

        # Save objects
        return mirror, facet, heliostat

    def _check_rotation(
        self, mirror, facet, heliostat, a1: float, a2: float, a3: float, a4: float
    ):
        # Test
        norm_0 = mirror.surface_norm_at(Vxy((0, 0)))  # mirror base
        norm_1 = mirror.survey_of_points(1)[1][0]  # mirror parent
        norm_2 = facet.survey_of_points(1)[1][0]  # facet parent
        norm_3 = heliostat.survey_of_points(1)[1][0]  # ensemble parent

        np.testing.assert_almost_equal(norm_0.x, a1, 6)
        np.testing.assert_almost_equal(norm_0.y, a1, 6)

        np.testing.assert_almost_equal(norm_1.x, a2, 6)
        np.testing.assert_almost_equal(norm_1.y, a2, 6)

        np.testing.assert_almost_equal(norm_2.x, a3, 6)
        np.testing.assert_almost_equal(norm_2.y, a3, 6)

        np.testing.assert_almost_equal(norm_3.x, a4, 6)
        np.testing.assert_almost_equal(norm_3.y, a4, 6)

    def test_culul_rotation_1(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.identity()
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0.001
        a3 = 0.001
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_2(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0.001
        a3 = 0.002
        a4 = 0.002

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_rotation_2(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0
        a3 = 0.001
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_3(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0.001
        a3 = 0.003
        a4 = 0.003

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_rotation_3(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.identity()
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.identity()
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0
        a3 = 0.001
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_4(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0.001
        a3 = 0.003
        a4 = 0.004

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_rotation_4(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.identity()
        r3 = Rotation.identity()
        r4 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r5 = Rotation.identity()
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_culul_rotation_5(self):
        # Generate optics with defined rotations
        r1 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r2 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r3 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r4 = Rotation.from_rotvec([-0.001, 0.001, 0])
        r5 = Rotation.from_rotvec([-0.001, 0.001, 0])
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0.001
        a3 = 0.003
        a4 = 0.005

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)

    def test_rotation_5(self):
        # Generate optics with defined rotations
        r1 = Rotation.identity()
        r2 = Rotation.identity()
        r3 = Rotation.identity()
        r4 = Rotation.identity()
        r5 = Rotation.from_rotvec([-0.001, 0.001, 0])
        mirror, facet, heliostat = self._generate_optics_rotation(r1, r2, r3, r4, r5)

        # Define expected xy angles
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0.001

        # Check surface normal angles
        self._check_rotation(mirror, facet, heliostat, a1, a2, a3, a4)


if __name__ == '__main__':
    test = TestCSPOpticsOrientation()

    test.test_rotation_2()
