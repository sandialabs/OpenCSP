"""


Model of machine vision camera pose in six degrees of freedom.

"""

import cv2 as cv
import numpy as np

import opencsp.common.lib.geometry.geometry_3d as g3d
import opencsp.common.lib.geometry.transform_3d as t3d


class CameraTransform:
    """
    Model of a camera positon in space, including rotation.
    """

    def __init__(
        self,
        x,  # m.   (x,y,z) location of coordinate system origin.
        y,  # m.
        z,  # m.
        az,  # rad.  Azimuth.    Compass heading measured clockwise from north.
        el,  # rad.  Elevation.  Tilt measured from xy plane, positive is up.
    ):
        super(CameraTransform, self).__init__()

        # Input parameters.
        self._x = x  # Do not access directly.  Fetch via tvec.
        self._y = y  # Do not access directly.  Fetch via tvec.
        self._z = z  # Do not access directly.  Fetch via tvec.
        self._az = az  # Do not access directly.  Fetch via rvec.
        self._el = el  # Do not access directly.  Fetch via rvec.
        # Dependent parameters.
        # The "inverse transform" transforms the camera face up at the (0,0,0) origin to the camera poition with proper view direction.
        self.inverse_rotation_matrix = self.construct_inverse_rotation_matrix(az, el)
        self.inverse_translation = self.construct_inverse_translation(x, y, z)
        self.inverse_transform = self.construct_inverse_transform()
        # The "transform" transforms points in the world coordinate system to camera coordinates.
        # That is, this is the tranform that moves the cameras for it position and view direction back to face up at (0,0,0).
        self.transform = self.construct_transform()
        self.rotation_matrix = self.construct_rotation_matrix()
        self.rvec = self.construct_rvec()  # These are the transforms required by OpenCV.
        self.tvec = self.construct_tvec()  #
        # View direction and origin plane.
        self.view_dir = self.construct_view_direction()
        self.origin_plane = self.construct_origin_plane()

    def construct_inverse_rotation_matrix(self, az, el):
        # This rotation rotates the camera z axis to the camera view direction.
        # Constants.
        x_axis = [1, 0, 0]
        z_axis = [0, 0, 1]
        # Component rotations.
        rot_z_to_y = t3d.axisrotation(x_axis, np.deg2rad(-90.0))
        rot_y_to_el = t3d.axisrotation(x_axis, el)
        rot_az_about_z = t3d.axisrotation(z_axis, -az)  # Azimuth is a compass heading, measured clockwise from north.
        # Combined rotation.
        rot_z_to_el = rot_y_to_el.dot(rot_z_to_y)
        rot_z_to_azel = rot_az_about_z.dot(rot_z_to_el)
        # Return.
        return rot_z_to_azel

    def construct_inverse_translation(self, x, y, z):
        # This vector translates the (0,0,0) origin to the camera position.
        return np.array([x, y, z])

    def construct_inverse_transform(self):
        # This transform moves the camera from the origin on its back, to its position with the proper view direction.
        r = self.inverse_rotation_matrix
        t = self.inverse_translation
        return np.array(
            [
                [r[0][0], r[0][1], r[0][2], t[0]],
                [r[1][0], r[1][1], r[1][2], t[1]],
                [r[2][0], r[2][1], r[2][2], t[2]],
                [0, 0, 0, 1],
            ]
        )

    def construct_transform(self):
        # This transform moves the camera from its position back to the origin on its back.
        # It also transforms points in world coordinates to camera coordinates.
        xf = self.inverse_transform
        return np.linalg.inv(xf)

    def construct_rotation_matrix(self):
        # This rotation moves the camera from its pointing direction to on its back.
        xf = self.transform
        return np.array(
            [[xf[0][0], xf[0][1], xf[0][2]], [xf[1][0], xf[1][1], xf[1][2]], [xf[2][0], xf[2][1], xf[2][2]]]
        )

    def construct_rvec(self):
        # Construct rvec, the Rodrigues vector representation of the camera rotation.
        # This is the form required by OpenCV.
        rvec, jacobian = cv.Rodrigues(self.rotation_matrix)
        # Return.
        return rvec

    def construct_tvec(self):
        # This vector moves the camera from its position to the origin, in the rotated coordinate system.
        xf = self.transform
        return np.array([xf[0][3], xf[1][3], xf[2][3]])

    def construct_view_direction(self):
        # Constant.
        z_axis = [0, 0, 1]
        return self.inverse_rotation_matrix.dot(z_axis)

    def construct_origin_plane(self):
        # Plane perpendicular to view direction containing the camera origin.
        # Orientted so the plane surface normal points in the diection of the camera view.
        normal = self.view_dir
        point_on_plane = self.inverse_translation
        distance_origin_to_plane = normal.dot(point_on_plane)
        A = normal[0]
        B = normal[1]
        C = normal[2]
        D = -distance_origin_to_plane  # Negate so that points on +z side of plane have positive distance values.
        # Return.
        return [A, B, C, D]

    # ACCESS

    def image_plane_front(self, camera):
        # Plane perpendicular to view direction at a distance equal to the focal length in front of the camera origin.
        # Orientted so the plane surface normal points in the diection of the camera view.
        normal = self.view_dir
        focal_length = camera.max_focal_length()
        point_on_plane = self.inverse_translation + (focal_length * normal)
        distance_origin_to_plane = normal.dot(point_on_plane)
        A = normal[0]
        B = normal[1]
        C = normal[2]
        D = -distance_origin_to_plane  # Negate so that points on +z side of plane have positive distance values.
        # Return.
        return [A, B, C, D]

    def pq_or_none(self, camera, xyz):
        # Computes the projection of the (x,y,z) point onto the image plane, unless the point
        # is on or behind the origin plane.
        # (Points behind the origin plane are behind the camera, and thus not in the field of view.
        # Points on the origin plane have numerically ill-defined projections.)
        # If the point is on or behind the origin plane, returns None.

        # Compute distance to origin plane.
        distance = g3d.homogeneous_plane_signed_distance_to_xyz(xyz, self.origin_plane)

        if distance < 1e-6:  # Fuzzy tolerance for on or behind plane.
            # Point is on or behind the origin plane.
            return None
        else:
            # Convert xyz into form required by OpenCV.
            obj_points = np.float64(
                [xyz]
            )  # float 64 required by OpenCV.  Can also "obj_points = obj_points.astype('float64')"
            open_cv_img_points, jacobian_project = cv.projectPoints(
                obj_points, self.rvec, self.tvec, camera.camera_matrix, camera.distortion_coeffs
            )
            # The OpenCV function returns a nested list of points.  We want just a list of points.
            if len(open_cv_img_points) != 1:
                print(
                    'ERROR: In CameraTransform.pq_or_none(), open_cv_img_pts = "'
                    + str(open_cv_img_points)
                    + '" was not of length 1.'
                )
                assert False
            nested_img_pt = open_cv_img_points[0]
            if len(nested_img_pt) != 1:
                print(
                    'ERROR: In CameraTransform.pq_or_none(), nested_img_pt = "'
                    + str(nested_img_pt)
                    + '" was not of length 1.'
                )
                assert False
            img_pt = nested_img_pt[0]
            if len(img_pt) != 2:
                print('ERROR: In CameraTransform.pq_or_none(), img_pt = "' + str(img_pt) + '" was not of length 2.')
                assert False
            # Set (p,q) coordinates.
            p = img_pt[0]
            q = -img_pt[1]  # Negate q so image will appear right-side up.
            # Return.
            return [p, q]
