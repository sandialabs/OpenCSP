# Copyright Sandia National Laboratories. All rights reserved.

import os
from pathlib import Path
import unittest
import numpy as np

from pyffiam.analysis import analysis
from pyffiam.ffiam_types import CspSite, AimType


class OutputTestCase(unittest.TestCase):
    def setUp(self):
        cwd = Path(os.getcwd())
        if 'tests' in cwd.as_posix():
            os.chdir('../src/pyffiam/')

    def test_sample_nsttf_point_analysis_succeeds(self):
        """ Tests predefined NSTTF site. """
        results = analysis(site=CspSite.NSTTF,
                           aim_strat=AimType.Point,
                           aim_params=np.array([0, 0, 90]),
                           threshold=4,
                           create_plots=True,
                           create_gifs=False,
                           create_xls=True,
                           open_output_dir=False,
                           )

        self.assertTrue(results.has_results)
        self.assertTrue(results.has_plots)
        # Total irradiance varies slightly between runs due to floating point;
        # places=-4 gives tolerance of 10,000
        self.assertAlmostEqual(results.total_irrad, 1_239_000, places=-4)

        self.assertIsNone(results.nu_heatmap_gif)

        self.assertIsNotNone(results.en_heatmap)

        self.assertTrue(results.xls_file.exists())

    def test_sample_v2_ring_analysis_succeeds(self):
        """Tests predefined generic site. """
        results = analysis(site=CspSite.SampleV2,
                           aim_strat=AimType.Ring,
                           aim_params=np.array([80, 120, 0]),
                           threshold=4,
                           create_plots=False,
                           create_gifs=False,
                           create_xls=True,
                           open_output_dir=False,
                           )

        self.assertTrue(results.has_results)
        # Total irradiance varies slightly between runs; places=-5 gives tolerance of 100,000
        self.assertAlmostEqual(results.total_irrad, 32_300_000, places=-5)

    def test_plots_created(self):
        results = analysis(site=CspSite.SampleV2,
                           aim_strat=AimType.Ring,
                           aim_params=np.array([80, 120, 0]),
                           threshold=3,
                           create_plots=True,
                           create_gifs=False,
                           create_xls=False,
                           open_output_dir=False,
                           )

        self.assertTrue(Path(results.en_heatmap).exists())
        self.assertTrue(Path(results.eu_heatmap).exists())
        self.assertTrue(Path(results.nu_heatmap).exists())
        self.assertTrue(Path(results.en_heatmap_zoomed).exists())
        self.assertTrue(Path(results.nu_heatmap_zoomed).exists())
        self.assertTrue(Path(results.eu_heatmap_zoomed).exists())

    def test_datafile_created(self):
        results = analysis(site=CspSite.SampleV2,
                           aim_strat=AimType.Ring,
                           aim_params=np.array([80, 120, 0]),
                           threshold=3,
                           create_plots=False,
                           create_gifs=False,
                           create_xls=True,
                           open_output_dir=False,
                           )

        self.assertTrue(Path(results.xls_file).exists())

    def test_gifs_created(self):
        # Check if ffmpeg is available (required for GIF creation)
        import shutil
        if shutil.which('ffmpeg') is None:
            self.skipTest("ffmpeg not found - required for GIF creation")

        try:
            results = analysis(site=CspSite.NSTTF,
                               aim_strat=AimType.Point,
                               aim_params=np.array([0, 0, 120]),
                               field_r=50,
                               max_height=40,
                               n_helios=10*10,
                               threshold=3,
                               create_plots=True,
                               create_gifs=True,
                               create_xls=False,
                               open_output_dir=False,
                               )
        except Exception as e:
            if 'codec' in str(e).lower() or 'ffmpeg' in str(e).lower():
                self.skipTest(f"GIF creation failed (ffmpeg/codec issue): {e}")
            raise

        self.assertTrue(Path(results.en_heatmap_gif).exists())
        self.assertTrue(Path(results.nu_heatmap_gif).exists())
        self.assertTrue(Path(results.eu_heatmap_gif).exists())

    def test_irrad_below_threshold_ignored(self):
        """Verifies that irradiance below threshold is ignored; i.e. plots are skipped. """
        results = analysis(site=CspSite.SampleV2,
                           aim_strat=AimType.Ring,
                           aim_params=np.array([80, 120, 0]),
                           n_helios=10*10,
                           threshold=50000,
                           create_plots=True,
                           create_gifs=False,
                           create_xls=False,
                           open_output_dir=False,
                           )

        self.assertIsNone(results.en_heatmap)
        self.assertTrue(results.total_irrad > 0)

        self.assertFalse(results.has_glare)
        self.assertEqual(results.n_glaring_voxels, 0)
        self.assertTrue(results.total_irrad_threshold == 0)
