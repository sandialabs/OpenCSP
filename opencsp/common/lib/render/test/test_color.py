import unittest

import matplotlib

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.render.color as cl


class test_Color(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path, name, _ = ft.path_components(__file__)
        cls.out_dir = ft.join(path, 'data/output', name.split('test_')[-1])
        ft.create_directories_if_necessary(cls.out_dir)
        ft.delete_files_in_directory(cls.out_dir, '*')
        return super().setUpClass()

    def setUp(self) -> None:
        self.test_name = self.id().split('.')[-1]

    def test_init_Color(self):
        """Simple class initialization."""
        instance = cl.Color(0, 0, 0, 'black', 'k')
        self.assertIsNotNone(instance)

    def test_init_range(self):
        """Verify that the RGB channels are enforced to a range 0-1."""
        color = cl.Color(-1, 1, 2, 'black', 'k')
        self.assertAlmostEqual(color.rgb()[0], 0)
        self.assertAlmostEqual(color.rgb()[1], 1)
        self.assertAlmostEqual(color.rgb()[2], 1)

    def test_rgb(self):
        color = cl.Color(0.16471, 0.32942, 0.49412, "The Answer", "DA")
        self.assertAlmostEqual(color.rgb()[0], 0.16471)
        self.assertAlmostEqual(color.rgb()[1], 0.32942)
        self.assertAlmostEqual(color.rgb()[2], 0.49412)

    def test_rgb_255(self):
        color = cl.Color(0.16471, 0.32942, 0.49412, "The Answer", "DA")
        self.assertEqual(color.rgb_255()[0], 42)
        self.assertEqual(color.rgb_255()[1], 84)
        self.assertEqual(color.rgb_255()[2], 126)

    def test_to_hex(self):
        color = cl.Color(0.16471, 0.32942, 0.49412, "The Answer", "DA")
        self.assertEqual(color.to_hex(), "#2A547E")

    def test_from_hex(self):
        color = cl.Color.from_hex("#2A547E", "The Answer", "DA")
        self.assertEqual(color.rgb_255()[0], 42)
        self.assertEqual(color.rgb_255()[1], 84)
        self.assertEqual(color.rgb_255()[2], 126)

    def test_to_hsv(self):
        hsv = cl.Color(0.16470588, 0.32941176, 0.49411764, "The Answer", "DA").to_hsv()
        self.assertAlmostEqual(hsv[0], 210 / 360.0, places=5)
        self.assertAlmostEqual(hsv[1], 66.667 / 100.0, places=5)
        self.assertAlmostEqual(hsv[2], 126 / 255.0, places=5)

    def test_from_hsv(self):
        color = cl.Color.from_hsv(210 / 360.0, 66.667 / 100.0, 126 / 255.0, "The Answer", "DA")
        self.assertAlmostEqual(color.rgb()[0], 0.16470588, places=5)
        self.assertAlmostEqual(color.rgb()[1], 0.32941176, places=5)
        self.assertAlmostEqual(color.rgb()[2], 0.49411764, places=5)

    def test_colormap_2(self):
        """Fade from red to blue. Verify there is a smooth transition between colors."""
        color_map = cl.red().build_colormap(cl.blue())

        # red
        self.assertAlmostEqual(color_map(0)[0], 1.0, places=2)
        self.assertAlmostEqual(color_map(0)[1], 0.0, places=2)
        self.assertAlmostEqual(color_map(0)[2], 0.0, places=2)

        self.assertAlmostEqual(color_map(0.5)[0], 0.5, places=2)
        self.assertAlmostEqual(color_map(0.5)[1], 0.0, places=2)
        self.assertAlmostEqual(color_map(0.5)[2], 0.5, places=2)

        # blue
        self.assertAlmostEqual(color_map(1.0)[0], 0.0, places=2)
        self.assertAlmostEqual(color_map(1.0)[1], 0.0, places=2)
        self.assertAlmostEqual(color_map(1.0)[2], 1.0, places=2)

    def test_colormap_3(self):
        """Fade from red to green to blue. Verify there is a smooth transition
        between colors. It is assumed that if colormaps work for 2 colors and 3
        colors, then they should work for any number of colors >= 2."""
        color_map = cl.red().build_colormap(cl.green(), cl.blue())

        # red
        self.assertAlmostEqual(color_map(0)[0], 1.0, places=2)
        self.assertAlmostEqual(color_map(0)[1], 0.0, places=2)
        self.assertAlmostEqual(color_map(0)[2], 0.0, places=2)

        self.assertAlmostEqual(color_map(0.25)[0], 0.5, places=2)
        self.assertAlmostEqual(color_map(0.25)[1], 0.5, places=2)
        self.assertAlmostEqual(color_map(0.25)[2], 0.0, places=2)

        # green
        self.assertAlmostEqual(color_map(0.5)[0], 0.0, places=2)
        self.assertAlmostEqual(color_map(0.5)[1], 1.0, places=2)
        self.assertAlmostEqual(color_map(0.5)[2], 0.0, places=2)

        self.assertAlmostEqual(color_map(0.75)[0], 0.0, places=2)
        self.assertAlmostEqual(color_map(0.75)[1], 0.5, places=2)
        self.assertAlmostEqual(color_map(0.75)[2], 0.5, places=2)

        # blue
        self.assertAlmostEqual(color_map(1.0)[0], 0.0, places=2)
        self.assertAlmostEqual(color_map(1.0)[1], 0.0, places=2)
        self.assertAlmostEqual(color_map(1.0)[2], 1.0, places=2)

    def test_matplotlibcolors_match(self):
        """
        The matplotlib tab10 colors are encoded in color.py. This test is here
        to catch any change in the matplotlib colors, if there ever is any.
        """
        mpl_colors = matplotlib.color_sequences['tab10']

        for i, color in cl.plot_colorsi.items():
            r1, g1, b1 = color.rgb()
            r2, g2, b2 = mpl_colors[i]
            self.assertAlmostEqual(r1, r2)
            self.assertAlmostEqual(g1, g2)
            self.assertAlmostEqual(b1, b2)


if __name__ == '__main__':
    unittest.main()
