import cv2 as cv
import numpy as np
from PIL import Image
import unittest

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from contrib.common.lib.cv.spot_analysis.image_processor.MomentsImageProcessor import MomentsImageProcessor
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
import opencsp.common.lib.geometry.Pxy as p2


class TestImportAnnotations(unittest.TestCase):
    def setUp(self):
        self._build_ellipse_img()

        return super().setUp()

    def _build_ellipse_img(self):
        dimensions = (140, 140, 3)
        base_image = np.zeros(dimensions, dtype="uint8")
        center = (70, 70)
        axes_lengths = (50, 25)
        angle = 45
        s, e = 0, 360  # start, end angles
        color = (255, 255, 255)
        thickness = -1  # fill
        self.ellipse_img = cv.ellipse(base_image, center, axes_lengths, angle, s, e, color, thickness)

    def test_centroid(self):
        """Test that the moments annotation produces the correct value when asked for the centroid"""
        # import here to avoid import cycles
        import contrib.common.lib.cv.annotations.MomentsAnnotation as manno

        # calculate the moments
        processor = MomentsImageProcessor()
        sa = SpotAnalysis(self._testMethodName, [processor])
        sa.set_primary_images([self.ellipse_img])
        for result in sa:
            pass

        # verify that the centroid is in the center
        moments: manno.MomentsAnnotation = result.get_fiducials_by_type(manno.MomentsAnnotation)[0]
        self.assertAlmostEqual(moments.cX, 70, delta=0.1)
        self.assertAlmostEqual(moments.cY, 70, delta=0.1)

    def test_rotation(self):
        """Test that the moments annotation produces the correct value when asked for the rotation"""
        # import here to avoid import cycles
        import contrib.common.lib.cv.annotations.MomentsAnnotation as manno

        # calculate the moments
        processor = MomentsImageProcessor()
        sa = SpotAnalysis(self._testMethodName, [processor])
        sa.set_primary_images([self.ellipse_img])
        for result in sa:
            pass

        # verify that the rotation is at 45 degrees
        moments: manno.MomentsAnnotation = result.get_fiducials_by_type(manno.MomentsAnnotation)[0]
        self.assertAlmostEqual(
            np.rad2deg(moments.rotation_angle_2d), 135, delta=0.1
        )  # 135 degrees is an acceptable answer

    def test_eccentricity(self):
        """Regression test to check that the eccentricity value hasn't changed since the last time this test was run."""
        # import here to avoid import cycles
        import contrib.common.lib.cv.annotations.MomentsAnnotation as manno

        # calculate the moments
        processor = MomentsImageProcessor()
        sa = SpotAnalysis(self._testMethodName, [processor])
        sa.set_primary_images([self.ellipse_img])
        for result in sa:
            pass

        # verify that the eccentricity matches the mathematical definition
        # e = sqrt(1 - b**2/a**2) = 0.8660254037844386
        moments: manno.MomentsAnnotation = result.get_fiducials_by_type(manno.MomentsAnnotation)[0]
        self.assertAlmostEqual(moments.eccentricity_untested, 0.8660254037844386, delta=0.01)

    def test_translate(self):
        # import here to avoid import cycles
        import contrib.common.lib.cv.annotations.MomentsAnnotation as manno

        # calculate the moments for the untranslated image
        processor = MomentsImageProcessor()
        sa = SpotAnalysis(self._testMethodName, [processor])
        sa.set_primary_images([self.ellipse_img])
        for result in sa:
            act_moments: manno.MomentsAnnotation = result.get_fiducials_by_type(manno.MomentsAnnotation)[0]
            act_moments = act_moments.translate(p2.Pxy([10, 20]))

        # translate the image and recalculate
        t_ellipse_img = np.zeros_like(self.ellipse_img)
        t_ellipse_img[20:, 10:, :] = self.ellipse_img[:-20, :-10, :]
        t_processor = MomentsImageProcessor()
        t_sa = SpotAnalysis(self._testMethodName, [t_processor])
        t_sa.set_primary_images([t_ellipse_img])
        for t_result in t_sa:
            t_moments: manno.MomentsAnnotation = t_result.get_fiducials_by_type(manno.MomentsAnnotation)[0]

        # verify that the moments have changed as expected
        self.assertAlmostEqual(t_moments.moments["m00"], act_moments.moments["m00"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m10"], act_moments.moments["m10"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m01"], act_moments.moments["m01"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m11"], act_moments.moments["m11"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m20"], act_moments.moments["m20"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m02"], act_moments.moments["m02"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m12"], act_moments.moments["m12"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m21"], act_moments.moments["m21"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m30"], act_moments.moments["m30"], delta=0.1)
        self.assertAlmostEqual(t_moments.moments["m03"], act_moments.moments["m03"], delta=0.1)

        # verify that the centroid is the only changed value
        self.assertAlmostEqual(act_moments.cX, 80, delta=0.1)
        self.assertAlmostEqual(act_moments.cY, 90, delta=0.1)
        self.assertAlmostEqual(np.rad2deg(act_moments.rotation_angle_2d), 135, delta=0.1)
        self.assertAlmostEqual(act_moments.eccentricity_untested, 0.86, delta=0.01)


if __name__ == "__main__":
    unittest.main()
