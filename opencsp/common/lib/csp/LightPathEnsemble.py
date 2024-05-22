import numpy as np
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.tool.typing_tools import strict_types


class LightPathEnsemble:

    def __init__(self, lps: list[LightPath]) -> None:
        self.current_directions = Vxyz.merge([lp.current_direction for lp in lps])
        self.init_directions = Vxyz.merge([lp.init_direction for lp in lps])
        self.points_lists = [lp.points_list for lp in lps]
        self.colors = [lp.color for lp in lps]

    def __getitem__(self, key) -> LightPath:
        if type(key) == slice:
            sargs = (key.start, key.stop, key.step)
            return [self[i] for i in range(*sargs)]
        points_list = self.points_lists[key]
        init_direction = self.init_directions[key]
        current_direction = self.current_directions[key]
        return LightPath(points_list, init_direction, current_direction)

    def __len__(self):
        return len(self.points_lists)

    def __iadd__(self, lpe: 'LightPathEnsemble'):
        """Alias for concatenate-in_place."""
        return self.concatenate_in_place(lpe)

    @classmethod
    def from_parts(cls, init_directions: Uxyz, points: list[Pxyz], curr_directions: Uxyz, colors=[]):
        lpe = LightPathEnsemble([])
        lpe.current_directions = curr_directions
        lpe.init_directions = init_directions
        lpe.points_lists = points
        lpe.colors = colors
        return lpe

    # @strict_types
    def add_steps(self, points: Pxyz, new_current_directions: Uxyz):
        if len(points) != len(new_current_directions):
            raise ValueError(
                f"The number of points but be the same as the number of new directions when appending to a LightPathEnsemble.\n \
                               There are {len(points)} points and {len(new_current_directions)} new directions."
            )
        if len(points) != len(self.points_lists):
            raise ValueError(
                f"The number of new steps is not equal to the number of light paths in the light path ensemble. \n \
                               There are {len(points)} new points and {len(self.points_lists)} light paths in the ensemble."
            )

        # for each new point we will concatenate it to the repsctive old point list
        for i, (old_points, new_point) in enumerate(zip(self.points_lists, points)):
            old_points: Pxyz
            new_point: Pxyz
            if new_point == None:
                new_current_directions = self.current_directions[i]
            else:
                self.points_lists[i] = old_points.concatenate(new_point)

        self.current_directions = new_current_directions  # update the current directions

    def concatenate_in_place(self: 'LightPathEnsemble', lpe1: 'LightPathEnsemble'):
        self.current_directions = self.current_directions.concatenate(lpe1.current_directions)
        self.init_directions = self.init_directions.concatenate(lpe1.init_directions)
        self.points_lists += lpe1.points_lists
        self.colors += lpe1.colors
        return self

    def asLightPathList(self) -> list[LightPath]:
        lps: list[LightPath] = []
        for cd, id, pl in zip(self.current_directions, self.init_directions, self.points_lists):
            lp = LightPath(pl, id, cd)
            lps.append(lp)
        return lps

    def __add__(self, lpe: 'LightPathEnsemble'):
        """Alias for concatenate-in_place."""
        return self.concatenate(lpe)

    def concatenate(self: 'LightPathEnsemble', lpe1: 'LightPathEnsemble'):
        new_lpe = LightPathEnsemble([])
        new_lpe.current_directions = self.current_directions.concatenate(lpe1.current_directions)
        new_lpe.init_directions = self.init_directions.concatenate(lpe1.init_directions)
        new_lpe.points_lists = self.points_lists + (lpe1.points_lists)
        new_lpe.colors = self.colors + (lpe1.colors)
        return new_lpe

    def asLightPathList(self) -> list[LightPath]:
        lps: list[LightPath] = []
        for cd, id, pl in zip(self.current_directions, self.init_directions, self.points_lists):
            lp = LightPath(pl, id, cd)
            lps.append(lp)
        return lps
