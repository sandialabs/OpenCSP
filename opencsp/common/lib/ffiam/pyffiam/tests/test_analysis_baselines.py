"""
Regression tests that verify analysis results against known baselines.

These tests ensure that changes to the analysis code don't unexpectedly
alter the computed irradiance values. The baselines were captured from
verified runs and stored in ffiam/tests/test_baselines.json.
"""

import json
import os
import pathlib
import unittest

import numpy as np


class AnalysisBaselineTests(unittest.TestCase):
    """Tests that verify analysis results match expected baselines."""

    @classmethod
    def setUpClass(cls):
        """Load baseline data and import analysis module."""
        # Find the baselines JSON file
        test_dir = pathlib.Path(__file__).parent
        repo_root = test_dir.parent.parent
        cls.baselines_path = repo_root / 'ffiam' / 'tests' / 'test_baselines.json'

        if not cls.baselines_path.exists():
            raise FileNotFoundError(f"Baselines file not found: {cls.baselines_path}")

        with open(cls.baselines_path, 'r') as f:
            cls.baselines = json.load(f)

        # Change to pyffiam source directory for imports
        cls.original_cwd = os.getcwd()
        pyffiam_src = test_dir.parent / 'src'
        os.chdir(pyffiam_src)

        # Import after changing directory
        from pyffiam.analysis import analysis
        from pyffiam.ffiam_types import CspSite, AimType
        cls.analysis = analysis
        cls.CspSite = CspSite
        cls.AimType = AimType

    @classmethod
    def tearDownClass(cls):
        """Restore original working directory."""
        os.chdir(cls.original_cwd)

    def _run_analysis(self, site, aim_type, aim_params, year, month, day, hour):
        """Helper to run analysis with common settings."""
        return self.analysis(
            site=site,
            year=year,
            month=month,
            day=day,
            hour=hour,
            threshold=4,
            aim_strat=aim_type,
            aim_params=np.array(aim_params),
            create_plots=False,
            create_gifs=False,
            create_xls=False,
            open_output_dir=False,
        )

    def _assert_within_tolerance(self, expected, actual, tolerance_pct, metric_name):
        """Assert that actual value is within tolerance percentage of expected."""
        if expected == 0:
            self.assertEqual(actual, 0, f"{metric_name}: expected 0, got {actual}")
            return

        diff_pct = abs(actual - expected) / expected * 100
        self.assertLessEqual(
            diff_pct, tolerance_pct,
            f"{metric_name}: {actual} differs from baseline {expected} by {diff_pct:.1f}% "
            f"(tolerance: {tolerance_pct}%)"
        )

    def test_nsttf_point_aim_total_irradiance(self):
        """Test NSTTF Point aim total irradiance matches baseline."""
        baseline = self.baselines["nsttf_point"]
        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        # Total irradiance should match within 1%
        self._assert_within_tolerance(
            baseline["total_irrad"], result.total_irrad, 1.0, "total_irrad"
        )

    def test_nsttf_point_aim_peak_irradiance(self):
        """Test NSTTF Point aim peak irradiance matches baseline."""
        baseline = self.baselines["nsttf_point"]
        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        # Peak irradiance should match within 2%
        self._assert_within_tolerance(
            baseline["peak_irrad"], result.peak_irrad, 2.0, "peak_irrad"
        )

    def test_nsttf_point_aim_glaring_voxels(self):
        """Test NSTTF Point aim glaring voxel count matches baseline."""
        baseline = self.baselines["nsttf_point"]
        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        # Glaring voxels should match within 5%
        self._assert_within_tolerance(
            baseline["n_glaring_voxels"], result.n_glaring_voxels, 5.0, "n_glaring_voxels"
        )

    def test_nsttf_ring_aim_total_irradiance(self):
        """Test NSTTF Ring aim total irradiance matches baseline."""
        baseline = self.baselines["nsttf_ring"]
        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Ring,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self._assert_within_tolerance(
            baseline["total_irrad"], result.total_irrad, 1.0, "total_irrad"
        )

    def test_nsttf_ring_aim_peak_irradiance(self):
        """Test NSTTF Ring aim produces lower peak than Point aim."""
        baseline_ring = self.baselines["nsttf_ring"]
        baseline_point = self.baselines["nsttf_point"]

        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Ring,
            aim_params=baseline_ring["aim_params"],
            year=baseline_ring["date"]["year"],
            month=baseline_ring["date"]["month"],
            day=baseline_ring["date"]["day"],
            hour=baseline_ring["date"]["hour"],
        )

        # Ring aim should distribute irradiance, resulting in lower peak
        self.assertLess(
            result.peak_irrad, baseline_point["peak_irrad"],
            "Ring aim should have lower peak irradiance than Point aim"
        )

        # Should still match its own baseline
        self._assert_within_tolerance(
            baseline_ring["peak_irrad"], result.peak_irrad, 2.0, "peak_irrad"
        )

    def test_nsttf_morning_reduced_irradiance(self):
        """Test that morning analysis produces less irradiance than noon."""
        baseline_morning = self.baselines["nsttf_morning"]
        baseline_noon = self.baselines["nsttf_point"]

        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Point,
            aim_params=baseline_morning["aim_params"],
            year=baseline_morning["date"]["year"],
            month=baseline_morning["date"]["month"],
            day=baseline_morning["date"]["day"],
            hour=baseline_morning["date"]["hour"],
        )

        # Morning should have less total irradiance than noon
        self.assertLess(
            result.total_irrad, baseline_noon["total_irrad"],
            "Morning irradiance should be less than noon"
        )

        # Should match baseline within tolerance
        self._assert_within_tolerance(
            baseline_morning["total_irrad"], result.total_irrad, 1.0, "total_irrad"
        )

    def test_samplev1_point_aim_total_irradiance(self):
        """Test SampleV1 Point aim total irradiance matches baseline."""
        baseline = self.baselines["samplev1_point"]
        result = self._run_analysis(
            site=self.CspSite.SampleV1,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self._assert_within_tolerance(
            baseline["total_irrad"], result.total_irrad, 1.0, "total_irrad"
        )

    def test_samplev1_more_heliostats_more_irradiance(self):
        """Test that SampleV1 (more heliostats) produces more total irradiance."""
        baseline_sample = self.baselines["samplev1_point"]
        baseline_nsttf = self.baselines["nsttf_point"]

        result = self._run_analysis(
            site=self.CspSite.SampleV1,
            aim_type=self.AimType.Point,
            aim_params=baseline_sample["aim_params"],
            year=baseline_sample["date"]["year"],
            month=baseline_sample["date"]["month"],
            day=baseline_sample["date"]["day"],
            hour=baseline_sample["date"]["hour"],
        )

        # SampleV1 has ~1936 heliostats vs NSTTF's 218
        # Should produce significantly more total irradiance
        self.assertGreater(
            result.total_irrad, baseline_nsttf["total_irrad"],
            "SampleV1 should produce more irradiance than NSTTF"
        )

    def test_heliostat_count_matches_baseline(self):
        """Test that heliostat counts match expected values."""
        baseline = self.baselines["nsttf_point"]
        result = self._run_analysis(
            site=self.CspSite.NSTTF,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self.assertEqual(
            result.num_heliostats, baseline["num_heliostats"],
            f"Heliostat count mismatch: got {result.num_heliostats}, "
            f"expected {baseline['num_heliostats']}"
        )

    # =========================================================================
    # V3 Field Size Tests (largest field configuration)
    # =========================================================================

    def test_samplev3_point_aim_total_irradiance(self):
        """Test SampleV3 (V3 field size) total irradiance matches baseline."""
        baseline = self.baselines["samplev3_point"]
        result = self._run_analysis(
            site=self.CspSite.SampleV3,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self._assert_within_tolerance(
            baseline["total_irrad"], result.total_irrad, 1.0, "total_irrad"
        )

    def test_samplev3_point_aim_peak_irradiance(self):
        """Test SampleV3 peak irradiance matches baseline.

        Note: Large fields like V3 have more variability in peak values
        due to the larger voxel grid, so we use a higher tolerance (5%).
        """
        baseline = self.baselines["samplev3_point"]
        result = self._run_analysis(
            site=self.CspSite.SampleV3,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        # Higher tolerance for large fields due to variability in peak detection
        self._assert_within_tolerance(
            baseline["peak_irrad"], result.peak_irrad, 5.0, "peak_irrad"
        )

    def test_samplev3_heliostat_count(self):
        """Test SampleV3 has expected heliostat count (~11,000)."""
        baseline = self.baselines["samplev3_point"]
        result = self._run_analysis(
            site=self.CspSite.SampleV3,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self.assertEqual(
            result.num_heliostats, baseline["num_heliostats"],
            f"SampleV3 heliostat count mismatch: got {result.num_heliostats}, "
            f"expected {baseline['num_heliostats']}"
        )
        # V3 should have ~11,000 heliostats
        self.assertGreater(result.num_heliostats, 10000)

    def test_samplev3_more_irradiance_than_v1(self):
        """Test that V3 field produces more irradiance than V1."""
        baseline_v3 = self.baselines["samplev3_point"]
        baseline_v1 = self.baselines["samplev1_point"]

        result = self._run_analysis(
            site=self.CspSite.SampleV3,
            aim_type=self.AimType.Point,
            aim_params=baseline_v3["aim_params"],
            year=baseline_v3["date"]["year"],
            month=baseline_v3["date"]["month"],
            day=baseline_v3["date"]["day"],
            hour=baseline_v3["date"]["hour"],
        )

        # V3 has ~11,000 heliostats vs V1's ~1,936
        self.assertGreater(
            result.total_irrad, baseline_v1["total_irrad"],
            "V3 field should produce more irradiance than V1"
        )

    # =========================================================================
    # Crescent Dunes Tests (real-world site with actual heliostat positions)
    # =========================================================================

    def test_crescent_dunes_total_irradiance(self):
        """Test Crescent Dunes total irradiance matches baseline."""
        baseline = self.baselines["crescent_dunes_point"]
        result = self._run_analysis(
            site=self.CspSite.CrescentDunes,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self._assert_within_tolerance(
            baseline["total_irrad"], result.total_irrad, 1.0, "total_irrad"
        )

    def test_crescent_dunes_peak_irradiance(self):
        """Test Crescent Dunes peak irradiance matches baseline."""
        baseline = self.baselines["crescent_dunes_point"]
        result = self._run_analysis(
            site=self.CspSite.CrescentDunes,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self._assert_within_tolerance(
            baseline["peak_irrad"], result.peak_irrad, 2.0, "peak_irrad"
        )

    def test_crescent_dunes_heliostat_count(self):
        """Test Crescent Dunes has expected heliostat count (10,348)."""
        baseline = self.baselines["crescent_dunes_point"]
        result = self._run_analysis(
            site=self.CspSite.CrescentDunes,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self.assertEqual(
            result.num_heliostats, baseline["num_heliostats"],
            f"Crescent Dunes heliostat count mismatch: got {result.num_heliostats}, "
            f"expected {baseline['num_heliostats']}"
        )
        # Crescent Dunes should have ~10,348 heliostats
        self.assertGreater(result.num_heliostats, 10000)

    def test_crescent_dunes_glaring_voxels(self):
        """Test Crescent Dunes glaring voxel count matches baseline."""
        baseline = self.baselines["crescent_dunes_point"]
        result = self._run_analysis(
            site=self.CspSite.CrescentDunes,
            aim_type=self.AimType.Point,
            aim_params=baseline["aim_params"],
            year=baseline["date"]["year"],
            month=baseline["date"]["month"],
            day=baseline["date"]["day"],
            hour=baseline["date"]["hour"],
        )

        self._assert_within_tolerance(
            baseline["n_glaring_voxels"], result.n_glaring_voxels, 5.0, "n_glaring_voxels"
        )


class BaselineFileTests(unittest.TestCase):
    """Tests for the baseline file itself."""

    def test_baselines_file_exists(self):
        """Verify the baselines file exists."""
        test_dir = pathlib.Path(__file__).parent
        repo_root = test_dir.parent.parent
        baselines_path = repo_root / 'ffiam' / 'tests' / 'test_baselines.json'

        self.assertTrue(
            baselines_path.exists(),
            f"Baselines file not found at {baselines_path}"
        )

    def test_baselines_file_valid_json(self):
        """Verify the baselines file contains valid JSON."""
        test_dir = pathlib.Path(__file__).parent
        repo_root = test_dir.parent.parent
        baselines_path = repo_root / 'ffiam' / 'tests' / 'test_baselines.json'

        with open(baselines_path, 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, dict)
        self.assertIn("nsttf_point", data)
        self.assertIn("nsttf_ring", data)
        self.assertIn("samplev1_point", data)
        self.assertIn("nsttf_morning", data)
        self.assertIn("samplev3_point", data)
        self.assertIn("crescent_dunes_point", data)

    def test_baselines_have_required_fields(self):
        """Verify each baseline has required fields."""
        test_dir = pathlib.Path(__file__).parent
        repo_root = test_dir.parent.parent
        baselines_path = repo_root / 'ffiam' / 'tests' / 'test_baselines.json'

        with open(baselines_path, 'r') as f:
            data = json.load(f)

        required_fields = ["total_irrad", "peak_irrad", "n_glaring_voxels", "aim_params", "date"]

        for name, baseline in data.items():
            for field in required_fields:
                self.assertIn(
                    field, baseline,
                    f"Baseline '{name}' missing required field '{field}'"
                )


if __name__ == '__main__':
    unittest.main()
