import unittest


class TestImportAnnotations(unittest.TestCase):
    def test_import(self):
        """Tests that we can import the annotations without encountering a cyclic import error."""
        import contrib.common.lib.cv.annotations.MomentsAnnotation
        import contrib.common.lib.cv.annotations.RectangleAnnotations
        import contrib.common.lib.cv.annotations.SpotWidthAnnotation


if __name__ == "__main__":
    unittest.main()
