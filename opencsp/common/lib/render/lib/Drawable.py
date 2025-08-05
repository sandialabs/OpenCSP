from abc import ABC, abstractmethod
from typing import Any

import opencsp.common.lib.render.figure_management  # for docstring comments
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class Drawable(ABC):
    @abstractmethod
    def draw(self, view: v3d.View3d, style: rcps.RenderControlPointSeq | Any) -> None:
        """Draw this instance on the given view.

        Parameters
        ----------
        view : v3d.View3d
            The view to draw this instance on. See :py:meth:`figure_management.setup_figure` or
            :py:meth:`figure_management.setup_figure_for_3d_data` for examples of view
            instantiation.
        style : rcps.RenderControlPointSeq | Any
            The style used to render this instance. The implementing class will
            typically override this to use its own style class type.
        """
