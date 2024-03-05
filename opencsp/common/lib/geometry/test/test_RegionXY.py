import matplotlib.pyplot as plt
import numpy as np

from   opencsp.common.lib.geometry.LoopXY import LoopXY
from   opencsp.common.lib.geometry.RegionXY import RegionXY
from   opencsp.common.lib.geometry.Vxy import Vxy


class TestRegionXY:
    def test_Region_instantiation(self):
        # Region with one square loop
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)
        region = RegionXY(loop)

        assert type(region.loops) is list

    def test_add_loop(self):
        # Region with one square loop
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)
        region = RegionXY(loop)

        with np.testing.assert_raises(NotImplementedError):
            region.add_loop(loop)

    def test_draw(self):
        # Region with one square loop
        verts = Vxy(([1, 0, 0, 1], [1, 1, 0, 0]))
        loop = LoopXY.from_vertices(verts)
        region = RegionXY(loop)

        fig = plt.figure()
        region.draw(fig.gca())
        plt.close(fig)
