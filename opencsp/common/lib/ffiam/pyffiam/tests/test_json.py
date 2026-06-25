# Copyright Sandia National Laboratories. All rights reserved.

import os
from pathlib import Path
import unittest
import numpy as np

from pyffiam.analysis import utils
from pyffiam.ffiam_types import CspSite


class PresetImportTestCase(unittest.TestCase):
    def setUp(self):
        cwd = Path(os.getcwd())
        if 'tests' in cwd.as_posix():
            os.chdir('../src/pyffiam/')

    def test_successfully_imports_all_preset_sites(self):
        presets = utils.load_preset_sites_from_json()
        self.assertTrue(CspSite.NSTTF in presets)
        self.assertTrue(CspSite.CrescentDunesSmall in presets)
        self.assertTrue(CspSite.CrescentDunes in presets)
        self.assertTrue(CspSite.SampleV1 in presets)
        self.assertTrue(CspSite.SampleV2 in presets)
        self.assertTrue(CspSite.SampleV3 in presets)

        nsttf = presets[CspSite.NSTTF]

        self.assertTrue("RunId" in nsttf)
        self.assertTrue("SiteType" in nsttf)
        self.assertTrue("HelioDesign" in nsttf)