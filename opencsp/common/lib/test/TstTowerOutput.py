"""
Demonstrate Tower Plotting Routines

Copyright (c) 2021 Sandia National Laboratories.

"""

from   datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
from   opencsp.common.lib.csp.SolarField import SolarField
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
from   opencsp.common.lib.csp.Tower import Tower
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
from   opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
from   opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from   opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
import opencsp.common.lib.render_control.RenderControlTower as rct
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

 
class TestTowerOutput(to.TestOutput):

    @classmethod
    def setup_class(self, 
                    source_file_body:str='TestTowerOutput',  # Set these here, because pytest calls
                    figure_prefix_root:str='tto',             # setup_class() with no arguments.
                     interactive:bool=False,
                     verify:bool=True):
        
        
        # Save input.
        # Interactive mode flag.
        # This has two effects:
        #    1. If interactive mode, plots stay displayed.
        #    2. If interactive mode, after several plots have been displayed, 
        #       plot contents might change due to Matplotlib memory issues.
        #       This may in turn cause misleading failures in comparing actual
        #       output to expected output.
        # Thus:
        #   - Run pytest in non-interactive mode.
        #   - If viewing run output interactively, be advised that figure content might 
        #     change (e.g., garbled plot titles, etc), and actual-vs-expected comparison 
        #     might erroneously fail (meaning they might report an error which is not 
        #     actually a code error).
        #



        super(TestTowerOutput, self).setup_class(source_file_body=source_file_body,
                                                  figure_prefix_root=figure_prefix_root,
                                                  interactive=interactive,
                                                  verify=verify)

        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant 
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.





    def test_single_tower(self) -> None:
        """
        Draws one tower.
        """
        # Initialize test.
        self.start_test()

        #View setup
        title = 'Single Tower'
        caption = 'A single Sandia NSTTF tower.'
        comments = []

        # Configuration setup
        tower= Tower(name='Sandia NSTTF', origin=np.array([0,0,0]), parts = ["whole tower", "target"])
        
        # Setup render control.
        # Style setup
        tower_control = rct.normal_tower()

        #comments\
        comments.append("Demonstration of single 3d tower drawing.")
        
        # Draw.
        fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),number_in_name=False,
                                                input_prefix=self.figure_prefix(1), # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                                                title=title, caption=caption, comments=comments, code_tag=self.code_tag)
        tower.draw(fig_record.view, tower_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

  


    def test_multiple_towers(self) -> None:
        """
        Draws one tower.
        """
        # Initialize test.
        self.start_test()

        #View setup
        title = 'Multiple Towers'
        caption = 'Sandia NSTTF reciever and control tower.'
        comments = []

        # Configuration setup
        tower_receiver= Tower(name='Sandia NSTTF', origin=np.array([0,0,0]), parts = ["whole tower", "target"])
        tower_control= Tower(name='Sandia NSTTF Control Tower', 
                            origin = np.array([0,0,0]),
                            height=50, 
                            east = 8.8,
                            west = -8.8,
                            south = 332.4,
                            north = 350)
        
        # Setup render control.
        # Style setup
        tower_control_rec = rct.normal_tower()
        tower_control_con = rct.no_target()

        #comments\
        comments.append("Demonstration of single 3d tower drawing.")
        comments.append("Black tower is NSTTF receiver tower.")
        comments.append("Green tower is NSTTF control tower.")        

        
        # Draw.
        fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),number_in_name=False,
                                                input_prefix=self.figure_prefix(2), # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                                                title=title, caption=caption, comments=comments, code_tag=self.code_tag)
        tower_receiver.draw(fig_record.view, tower_control_rec)
        tower_control.draw(fig_record.view, tower_control_con)

        # Output.
        self.show_save_and_check_figure(fig_record)


   

    def test_heliostat_vector_field(self) -> None:
        """
        Draws heliostat vector field.
        """
        # Initialize test.
        self.start_test()

        # View setup
        title = 'Heliostat Vector Field'
        caption = 'Rendering of the normal vector at the heliostat origin, for each heliostat in a field of tracking heliostats.'
        comments = []
        
        # Tracking setup
        # Define tracking time.
        solar_field = self.solar_field
        aimpoint_xyz = [60.0, 8.8, 28.9]
        when_ymdhmsz = [2021,   5,   13,   13,    2,       0,    -6]  # NSTTF solar noon
        #[year, month, day, hour, minute, second, zone]
        solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
        
        # Style setup
        solar_field_style = rcsf.heliostat_vector_field(color='b')
        
        # Comment
        comments.append("Each heliostat's surface normal, which can be viewed as a vector field.")
        
        # Draw and produce output for 3d
        fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(), number_in_name=False,
                                                input_prefix=self.figure_prefix(13), # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                                                title=title, caption=caption, comments=comments, code_tag=self.code_tag)
        fig_record.view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz')
        solar_field.draw(fig_record.view, solar_field_style)
        self.show_save_and_check_figure(fig_record)




# MAIN EXECUTION

if __name__ == "__main__":
    # Control flags.
    interactive = True
    # Set verify to False when you want to generate all figures and then copy 
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False #False
    # Setup.
    test_object = TestTowerOutput()
    test_object.setup_class(interactive=interactive, verify=verify)
    # Tests.
    lt.info('Beginning tests...')
    test_object.test_single_tower()
    test_object.test_multiple_towers()

    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.teardown_method()
