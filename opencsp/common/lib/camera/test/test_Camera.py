import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestVxyz:
    @classmethod
    def setup_class(cls):
        # Create camera parameters: f=1000 pixels, 1000 x 1000 pixels
        intrinsic_mat = np.array([[1000, 0, 500.5], [0, 1000, 250.5], [0, 0, 1]])
        distortion_coef_zeros = np.array([0.0, 0.0, 0.0, 0.0])
        distortion_coef_real = np.array([0.01, 0.02, 0.001, 0.002])
        image_shape_xy = (1000, 500)

        cls.camera_ideal = Camera(intrinsic_mat, distortion_coef_zeros, image_shape_xy, "Test Ideal Camera")
        cls.camera_real = Camera(intrinsic_mat, distortion_coef_real, image_shape_xy, "Test Real Camera")

        # Define upper left 3D point and image location
        cls.Vxyz_ul = Vxyz((-1, -0.5, 2))
        cls.Puv_real_ul = Pxy((-0.1640625, 0.1679687))
        cls.Puv_ideal_ul = Pxy((0.5, 0.5))

        # Define lower right 3D point and image location
        cls.Vxyz_lr = Vxyz((1, 0.5, 2))
        cls.Puv_real_lr = Pxy((1004.9140625, 502.7070312))
        cls.Puv_ideal_lr = Pxy((1000.5, 500.5))

    def test_center_ray(self):
        # Create point location
        V = Vxyz((0, 0, 10))
        # Create point rotation
        R = Rotation.from_rotvec([0, 0, 0])
        # Create point
        P = Pxyz((0, 0, 0))
        Puv_exp = Pxy((500.5, 250.5))

        # Project point to ideal camera
        Puv = self.camera_ideal.project(P, R, V)
        np.testing.assert_almost_equal(Puv.data, Puv_exp.data)

        # Project point to real camera
        Puv = self.camera_real.project(P, R, V)
        np.testing.assert_almost_equal(Puv.data, Puv_exp.data)

    def test_edge_ray_ideal(self):
        # Create lower right ray
        V = self.Vxyz_lr
        R = Rotation.from_rotvec([0, 0, 0])
        P = Pxyz((0, 0, 0))
        Puv = self.camera_ideal.project(P, R, V)
        Puv_exp = self.Puv_ideal_lr
        np.testing.assert_almost_equal(Puv.data, Puv_exp.data)

        # Create upper left ray
        V = self.Vxyz_ul
        R = Rotation.from_rotvec([0, 0, 0])
        P = Pxyz((0, 0, 0))
        Puv = self.camera_ideal.project(P, R, V)
        Puv_exp = self.Puv_ideal_ul
        np.testing.assert_almost_equal(Puv.data, Puv_exp.data)

    def test_edge_ray_real(self):
        # Create lower right ray
        V = self.Vxyz_lr
        R = Rotation.from_rotvec([0, 0, 0])
        P = Pxyz((0, 0, 0))
        Puv = self.camera_real.project(P, R, V)
        Puv_exp = self.Puv_real_lr
        np.testing.assert_almost_equal(Puv.data, Puv_exp.data)

        # Create upper left ray
        V = self.Vxyz_ul
        R = Rotation.from_rotvec([0, 0, 0])
        P = Pxyz((0, 0, 0))
        Puv = self.camera_real.project(P, R, V)
        Puv_exp = self.Puv_real_ul
        np.testing.assert_almost_equal(Puv.data, Puv_exp.data)

    def test_ray_direction_ideal(self):
        # Create lower right ray
        Uxyz_point = self.camera_ideal.vector_from_pixel(self.Puv_ideal_lr)
        np.testing.assert_(type(Uxyz_point) is Uxyz)
        np.testing.assert_almost_equal(Uxyz_point.data, self.Vxyz_lr.normalize().data)

        # Create upper left ray
        Uxyz_point = self.camera_ideal.vector_from_pixel(self.Puv_ideal_ul)
        np.testing.assert_(type(Uxyz_point) is Uxyz)
        np.testing.assert_almost_equal(Uxyz_point.data, self.Vxyz_ul.normalize().data)

    def test_ray_direction_real(self):
        # Create lower right ray
        Uxyz_point = self.camera_real.vector_from_pixel(self.Puv_real_lr)
        np.testing.assert_(type(Uxyz_point) is Uxyz)
        np.testing.assert_almost_equal(Uxyz_point.data, self.Vxyz_lr.normalize().data)

        # Create upper left ray
        Uxyz_point = self.camera_real.vector_from_pixel(self.Puv_real_ul)
        np.testing.assert_(type(Uxyz_point) is Uxyz)
        np.testing.assert_almost_equal(Uxyz_point.data, self.Vxyz_ul.normalize().data)

    def test_image_shape(self):
        im_shape_xy = self.camera_real.image_shape_xy
        im_shape_yx = self.camera_real.image_shape_yx
        np.testing.assert_(type(im_shape_xy) is tuple)
        np.testing.assert_(type(im_shape_yx) is tuple)
        np.testing.assert_(im_shape_xy == im_shape_yx[::-1])


if __name__ == "__main__":
    Test = TestVxyz()
    Test.setup_class()

    Test.test_center_ray()
    Test.test_edge_ray_ideal()
    Test.test_edge_ray_real()
    Test.test_ray_direction_ideal()
    Test.test_image_shape()
