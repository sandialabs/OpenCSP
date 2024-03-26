"""
Example generation of color target images.
"""

import matplotlib
import numpy as np

import opencsp.common.lib.render.color as Color  # ?? SCAFFOLDING RCB - FIX FILENAME TO CAPITALIZED
import opencsp.common.lib.target.TargetColor as tc
import opencsp.common.lib.target.target_color_2d_rgb as tc2r
import opencsp.common.lib.target.target_color_convert as tcc
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.unit_conversion as uc


class ExampleTargetColor(to.TestOutput):
    @classmethod
    def setup_class(
        self,
        source_file_body: str = 'ExampleTargetColor',  # Set these here, because pytest calls
        figure_prefix_root: str = 'ttc',  # setup_class() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):
        # Generic setup.
        super(ExampleTargetColor, self).setup_class(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
            output_path='targetcolor',
        )
        # Setup matplotlib backend
        # matplotlib.use('TkAgg')

        # Define image size and resolution for all tests.
        # self.image_width_in  = 3.0 # 48.0 # inch
        # self.image_height_in = 5.0 # 96.0 # inch
        # self.image_width_in  = 9.5 # 48.0 # inch
        # self.image_height_in = 15.0 # 96.0 # inch
        # self.image_width_in  = 47.0 # 48.0 # inch
        # self.image_height_in = 47.0 # 96.0 # inch

        # For debugging.
        self.image_width_in = 20.0  # 48.0 # inch
        self.image_height_in = 20.0  # 96.0 # inch
        self.dpi = 25  # 100 # 100

        # # For debugging and briefing.
        # self.image_width_in  = 40.0 # 48.0 # inch
        # self.image_height_in = 40.0 # 96.0 # inch
        # self.dpi = 25 #100 # 100

        # # Full print size at Creative Services.
        # self.image_width_in  = 48.0 # inch
        # self.image_height_in = 48.0 # inch
        # self.dpi = 100

        # # Full print size at ARI.
        # self.image_width_in  = 118.1102362 # inch.  Note (39.37007874 inch = 1.0 m), (118.1102362 inch = 3 m)
        # self.image_height_in = 118.1102362 # inch.  Note (39.37007874 inch = 1.0 m), (118.1102362 inch = 3 m)
        # self.dpi = 100

        # # For debugging tall linear.
        # self.image_width_in  = 3.937007874 # inch.  Note (39.37007874 inch = 1.0 m), (3.937007874 inch = 0.1 m)
        # self.image_height_in = 118.1102362 # inch.  Note (39.37007874 inch = 1.0 m), (118.1102362 inch = 3 m)
        # self.dpi = 25 #100 # 100

        # Convert to standard internal units.
        self.image_height = uc.inch_to_meter(self.image_height_in)
        self.image_width = uc.inch_to_meter(self.image_width_in)
        self.dpm = uc.dpi_to_dpm(self.dpi)

    # TARGET CONSTRUCTION TESTS

    # ?? SCAFFOLDING RCB -- ADD COLOR_BAR TYPE TIP BELOW
    def execute_example_linear_color_bar(
        self, color_below_min: Color, color_bar, color_bar_name: str, color_above_max: Color, generate_all: bool
    ) -> None:
        if generate_all:
            # Linear color bar in x, discrete.
            target = tc.construct_target_linear_color_bar(
                self.image_width,
                self.image_height,
                self.dpm,
                color_below_min,
                color_bar,
                color_bar_name,
                color_above_max,
                'x',
                'discrete',
            )
            self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

        if generate_all:
            # Linear color bar in x, continuous.
            target = tc.construct_target_linear_color_bar(
                self.image_width,
                self.image_height,
                self.dpm,
                color_below_min,
                color_bar,
                color_bar_name,
                color_above_max,
                'x',
                'continuous',
            )
            self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

        # Linear color bar in y, discrete.
        target = tc.construct_target_linear_color_bar(
            self.image_width,
            self.image_height,
            self.dpm,
            color_below_min,
            color_bar,
            color_bar_name,
            color_above_max,
            'y',
            'discrete',
        )
        self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

        if generate_all:
            # Linear color bar in y, continuous.
            target = tc.construct_target_linear_color_bar(
                self.image_width,
                self.image_height,
                self.dpm,
                color_below_min,
                color_bar,
                color_bar_name,
                color_above_max,
                'y',
                'continuous',
            )
            self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

    # ?? SCAFFOLDING RCB -- ADD COLOR_BAR TYPE TIP BELOW
    def execute_example_polar_color_bar(
        self, color_below_min: Color, color_bar, color_bar_name: str, color_above_max: Color
    ) -> None:
        # Default.
        target = tc.construct_target_polar_color_bar(self.image_width, self.image_height, self.dpm)
        self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

        # Selected for first 3m x 3m print.
        target = tc.construct_target_polar_color_bar(
            self.image_width,
            self.image_height,
            self.dpm,
            discrete_or_continuous='continuous',
            pattern_boundary='image_boundary',
            radial_gradient_type='light_center_to_saturated',
            radial_gradient_name='l2s',
            light_center_to_saturated_saturation_min=0.2,
        )
        self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

    def execute_example_blue_under_red_cross_green(self) -> None:
        # Construct target.
        target = tc.construct_target_blue_under_red_cross_green(self.image_width, self.image_height, self.dpm)
        self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

    def execute_example_rgb_cube_inscribed_square(self, project_to_cube: bool) -> None:
        # Construct target.
        target = tc.construct_target_rgb_cube_inscribed_square(
            self.image_width, self.image_height, self.dpm, project_to_cube
        )
        self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

    # TARGET MODIFICATION TESTS

    def execute_example_adjust_color_saturation(self, saturation_fraction: float) -> None:
        # Construct target.
        target = tc.construct_target_blue_under_red_cross_green(self.image_width, self.image_height, self.dpm)
        # Adjust color saturation.
        target.adjust_color_saturation(saturation_fraction)
        print('WARNING:  In execute_example_adjust_color_saturation(), saturation adjustment not implemented yet.')
        # Save and check.
        self.save_and_check_image(target.image, self.dpm, target.description_inch(), '.png')

    # TARGET EXTENSION TESTS

    def execute_example_extend_target(self) -> None:
        # Target.
        target = tc.construct_target_blue_under_red_cross_green(self.image_width, self.image_height, self.dpm)
        # Extend left.
        left_pixels = 10  # Pixels
        extended_target_left = tc.extend_target_left(target, left_pixels, Color.white())
        # self.save_and_check_image(extended_target_left.image, self.dpm, extended_target_left.description_inch(), '.png')
        # Extend right.
        right_pixels = 20  # Pixels
        extended_target_left_right = tc.extend_target_right(extended_target_left, right_pixels, Color.grey())
        # self.save_and_check_image(extended_target_left_right.image, self.dpm, extended_target_left_right.description_inch(), '.png')
        # Extend top.
        top_pixels = 30  # Pixels
        extended_target_left_right_top = tc.extend_target_top(
            extended_target_left_right, top_pixels, Color.light_grey()
        )
        # self.save_and_check_image(extended_target_left_right_top.image, self.dpm, extended_target_left_right_top.description_inch(), '.png')
        # Extend bottom.
        bottom_pixels = 40  # Pixels
        extended_target_left_right_top_bottom = tc.extend_target_bottom(
            extended_target_left_right_top, bottom_pixels, Color.dark_grey()
        )
        # self.save_and_check_image(extended_target_left_right_top_bottom.image, self.dpm, extended_target_left_right_top_bottom.description_inch(), '.png')
        # Border all around.
        border_pixels = 5  # Pixels
        extended_target_left_right_top_bottom_border = tc.extend_target_all(
            extended_target_left_right_top_bottom, border_pixels, Color.magenta()
        )
        self.save_and_check_image(
            extended_target_left_right_top_bottom_border.image,
            self.dpm,
            extended_target_left_right_top_bottom_border.description_inch(),
            '.png',
        )

    # TARGET COMBINATION TESTS

    def execute_example_splice_targets_above_below(self) -> None:
        # ?? SCAFFOLDING RCB -- FIXUP PARAMETER PASSING, ETC.
        # Target #1.
        target_1 = tc.construct_target_blue_under_red_cross_green(self.image_width, self.image_height, self.dpm)
        # Target #2.
        # Selected for first 3m x 3m print.
        # project_to_cube = True
        # target_2 = tc.construct_target_rgb_cube_inscribed_square(self.image_width, self.image_height, self.dpm, project_to_cube)
        target_2 = tc.construct_target_polar_color_bar(
            self.image_width,
            self.image_height,
            self.dpm,
            discrete_or_continuous='continuous',
            pattern_boundary='image_boundary',
            radial_gradient_type='light_center_to_saturated',
            radial_gradient_name='l2s',
            light_center_to_saturated_saturation_min=0.2,
        )
        # Combine.
        # ?? SCAFFOLDING RCB -- SHOULD THIS BE IN INCHES?
        gap = 0  # Pixels  # ?? SCAFFOLDING RCB -- SHOULD THIS BE IN INCHES?
        spliced_target = tc.splice_targets_above_below(target_1, target_2, gap, Color.white())
        self.save_and_check_image(spliced_target.image, self.dpm, spliced_target.description_inch(), '.png')

    def execute_example_cascade_target_A(self) -> None:
        # For tall linear target elements.
        # Quick test run.
        color_bar_width_in = 2.952755906  # inch.  Note 2.952755906 inch = 0.075 m
        grey_bar_width_in = 2.755905512  # inch.  Note 2.755905512 inch = 0.070 m
        gap_between_bars_in = 1.771653543  # inch.  Note 1.771653543 inch = 0.045 m
        ref_gap_in = 0.196850393  # inch.  Note 0.196850393 inch = 0.005 m
        color_total_height_in = 94.48818898  # inch.  Note 94.48818898 inch = 2.4 m
        composite_dpi = 5
        # # For 2.4m targets on 8' stock.
        # color_bar_width_in    = 2.952755906  # inch.  Note 2.952755906 inch = 0.075 m
        # grey_bar_width_in     = 2.755905512  # inch.  Note 2.755905512 inch = 0.070 m
        # gap_between_bars_in   = 1.771653543  # inch.  Note 1.771653543 inch = 0.045 m
        # ref_gap_in            = 0.196850393  # inch.  Note 0.196850393 inch = 0.005 m
        # color_total_height_in = 94.48818898  # inch.  Note 94.48818898 inch = 2.4 m
        # composite_dpi         = 80 #100 # 100
        # # For 3m targets on 10' stock.
        # # color_total_height_in = 118.1102362 # inch.  Note 118.1102362 inch = 3 m
        # color_bar_width_in    = 3.690944883  # inch.  Note 2.952755906 inch = 0.075 m
        # grey_bar_width_in     = 3.444881890  # inch.  Note 2.755905512 inch = 0.070 m
        # gap_between_bars_in   = 2.214566929  # inch.  Note 1.771653543 inch = 0.045 m
        # ref_gap_in            = 0.246062991  # inch.  Note 0.196850393 inch = 0.005 m
        # color_total_height_in = 118.1102362  # inch.  Note 118.1102362 inch = 3 m
        # composite_dpi         = 80 #100 # 100
        # Convert to standard internal units.
        color_total_height = uc.inch_to_meter(color_total_height_in)
        color_bar_width = uc.inch_to_meter(color_bar_width_in)
        grey_bar_width = uc.inch_to_meter(grey_bar_width_in)
        gap_between_bars = uc.inch_to_meter(gap_between_bars_in)
        ref_gap = uc.inch_to_meter(ref_gap_in)
        composite_dpm = uc.dpi_to_dpm(composite_dpi)

        # Linear color bar definition.
        # Main:  Color bar corrected for Nikon D3300 response.
        color_below_min = Color.black()  # Black below bottom of color bar.
        color_bar = tcc.nikon_D3300_monitor_equal_step_color_bar()
        color_bar_name = 'D3300_monitor'  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS MEMBER
        color_above_max = Color.white()  # White background for "saturated data."
        # Closed color wheel linear color bar.
        ref_color_below_min = Color.black()  # Black below bottom of color bar.
        ref_color_bar = tcc.O_color_bar()
        ref_color_bar_name = 'O'  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS MEMBER
        ref_color_above_max = Color.white()  # White background for "saturated data."

        cascade_target = tc.construct_linear_color_bar_cascade(  # Dimensions.
            color_bar_width,
            color_total_height,
            composite_dpm,
            # Main color bar.
            color_below_min,
            color_bar,
            color_bar_name,
            color_above_max,
            # Reference color bar.
            ref_color_below_min,
            ref_color_bar,
            ref_color_bar_name,
            ref_color_above_max,
            # Direction.
            x_or_y='y',
            # Color stack specification.
            stack_sequence=[1, 2, 5, 10],
            list_of_discrete_or_continuous_lists=[
                ['continuous'],
                # 2-stack
                ['continuous', 'continuous'],
                # 5-stack
                ['discrete', 'continuous', 'continuous', 'continuous', 'continuous'],
                # 10-stack
                [
                    'discrete',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                    'continuous',
                ],
            ],
            list_of_saturation_spec_lists=[
                [[None, None, None, None]],
                # 2-stack
                [['light_to_saturated', None, 0.4, 1.0], ['saturated_to_white', 1.25, None, None]],
                # 5-stack
                [
                    [None, None, None, None],
                    ['saturated_to_white', 0.75, None, None],
                    ['saturated_to_white', 1.75, None, None],
                    ['light_to_saturated', None, 0.33, 1.0],
                    ['light_to_saturated', None, 0.67, 1.0],
                ],
                # 10-stack
                [
                    ['saturated_to_white', 1.5, None, None],
                    [None, None, None, None],
                    ['light_to_saturated', None, 0.00, 1.0],
                    ['light_to_saturated', None, 0.25, 1.0],
                    ['light_to_saturated', None, 0.50, 1.0],
                    ['light_to_saturated', None, 0.75, 1.0],
                    ['saturated_to_white', 0.5, None, None],
                    ['saturated_to_white', 1.0, None, None],
                    ['saturated_to_white', 1.5, None, None],
                    ['saturated_to_white', 2.0, None, None],
                ],
            ],
            # Grey context bar specification.
            include_grey_neighbors=True,
            grey_bar_width=grey_bar_width,
            # Space between bars.
            gap_between_bars_pix=round(
                gap_between_bars * composite_dpm
            ),  # Pixels  # ?? SCAFFOLDING RCB -- SHOULD THIS BE IN INCHES?
            ref_gap_pix=round(ref_gap * composite_dpm),  # Pixels  # ?? SCAFFOLDING RCB -- SHOULD THIS BE IN INCHES?
            gap_color=Color.white(),
        )

        # Fiducial marks.
        n_ticks_x = 13  # No units.  Number of tick marks to draw along top/bottom horizontal target edges.
        n_ticks_y = 25  # No units.  Number of tick marks to draw along left/right vertical target edges.
        tick_length = 0.010  # Meters.    Length to draw edge tick marks.
        tick_width_pix = 3  # Pixels.    Width to draw edge tick marks; should be odd number.
        tick_color: Color = Color.black()  # Color.     Color of edge tick marks.
        cascade_target.set_ticks_along_top_and_bottom_edges(n_ticks_x, tick_length, tick_width_pix, tick_color)
        cascade_target.set_ticks_along_left_and_right_edges(n_ticks_y, tick_length, tick_width_pix, tick_color)

        # Save result.
        # self.save_and_check_image(cascade_target.image, composite_dpm, cascade_target.description_inch(), '.tiff')  #'.png')
        self.save_and_check_image(cascade_target.image, composite_dpm, cascade_target.description_inch(), '.png')

    def example_matlab(self) -> None:
        # Initialize test.
        self.start_test()
        # MATLAB color bar.
        color_below_min = Color.black()  # Black below bottom of color bar.
        color_bar = tcc.matlab_color_bar()
        color_bar_name = 'matlab'
        color_above_max = Color.white()  # White background for "saturated data."
        self.execute_example_linear_color_bar(color_below_min, color_bar, color_bar_name, color_above_max, False)

    def example_matlab_equal_angle(self) -> None:
        # Initialize test.
        self.start_test()
        # Normalized MATLAB color bar.
        color_below_min = Color.black()  # Black below bottom of color bar.
        color_bar = tcc.normalize_color_bar_to_equal_angles(tcc.matlab_color_bar())
        color_bar_name = 'matlab_equal_angle'
        color_above_max = Color.white()  # White background for "saturated data."
        self.execute_example_linear_color_bar(color_below_min, color_bar, color_bar_name, color_above_max, False)

    #         # Closed corner tour color bar.
    # # ?? SCAFFOLDING RCB -- USE THIS TO CLARIFY AND THEN FIX PROBLEMS WITH COLOR INTERPOLATION.
    #         color_below_min = Color.black()  # Black below bottom of color bar.
    #         color_bar = tcc.O_color_bar()
    #         color_bar_name = 'O'  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS MEMBER
    #         # color_bar = tcc.corner_tour_closed_color_bar()
    #         # color_bar_name = 'corner_tour_closed'
    #         color_above_max = Color.white()  # White background for "saturated data."
    #         self.execute_example_linear_color_bar(color_below_min, color_bar, color_bar_name, color_above_max, True)

    def example_corner_tour_closed_equal_angle(self) -> None:
        # Initialize test.
        self.start_test()
        # Normalized closed corner tour color bar.
        color_below_min = Color.black()  # Black below bottom of color bar.
        color_bar = tcc.normalize_color_bar_to_equal_angles(tcc.corner_tour_closed_color_bar())
        color_bar_name = 'corner_tour_closed_equal_angle'
        color_above_max = Color.white()  # White background for "saturated data."
        self.execute_example_linear_color_bar(color_below_min, color_bar, color_bar_name, color_above_max, True)

    # # Closed color wheel linear color bar.
    # color_below_min = Color.black()  # Black below bottom of color bar.
    # color_bar = tcc.O_color_bar()
    # color_bar_name = 'O'  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS MEMBER
    # color_above_max = Color.white()  # White background for "saturated data."
    # self.execute_example_linear_color_bar(color_below_min, color_bar, color_bar_name, color_above_max)

    def example_polar_color_bar(self) -> None:
        # Initialize test.
        self.start_test()
        # Closed color wheel polar image.
        color_below_min = Color.black()  # Black below bottom of color bar.
        color_bar = tcc.O_color_bar()
        color_bar_name = 'O'  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS MEMBER
        color_above_max = Color.white()  # White background for "saturated data."
        self.execute_example_polar_color_bar(color_below_min, color_bar, color_bar_name, color_above_max)

    def example_blue_under_red_cross_green(self) -> None:
        # Initialize test.
        self.start_test()
        # Blue underlying red cross green.
        self.execute_example_blue_under_red_cross_green()

    def example_rgb_cube_inscribed_square_False(self) -> None:
        # Initialize test.
        self.start_test()
        # Square inscribed in the [R,G,B] space basis vector hexagon.
        self.execute_example_rgb_cube_inscribed_square(False)

    def example_rgb_cube_inscribed_square_True(self) -> None:
        # Initialize test.
        self.start_test()
        # Square inscribed in the [R,G,B] space basis vector hexagon, with rays projected to cube surface.
        self.execute_example_rgb_cube_inscribed_square(True)

        # # Adjust color saturation.
        # self.execute_example_adjust_color_saturation(0.0)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.1)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.2)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.3)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.4)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.5)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.6)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.7)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.8)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(0.9)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.0)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.1)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.2)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.3)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.4)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.5)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.6)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.7)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.8)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(1.9)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(2.0)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(2.1)  # saturation_fraction
        # self.execute_example_adjust_color_saturation(2.2)  # saturation_fraction

    def example_extend_target(self) -> None:
        # Initialize test.
        self.start_test()
        # Add border pixels to a target.
        self.execute_example_extend_target()

    def example_splice_targets_above_below(self) -> None:
        # Initialize test.
        self.start_test()
        # Combine squares into a single image.
        self.execute_example_splice_targets_above_below()

    def example_cascade_target_A(self) -> None:
        # Initialize test.
        self.start_test()
        # Composite image #1.
        self.execute_example_cascade_target_A()


def example_driver():
    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False  # True
    # Setup.
    example_object = ExampleTargetColor()
    example_object.setup_class(interactive=interactive, verify=verify)
    # Examples.
    lt.info('Beginning tests...')
    example_object.example_matlab()
    example_object.example_matlab_equal_angle()
    example_object.example_corner_tour_closed_equal_angle()
    example_object.example_polar_color_bar()
    example_object.example_blue_under_red_cross_green()
    example_object.example_rgb_cube_inscribed_square_False()
    example_object.example_rgb_cube_inscribed_square_True()
    example_object.example_extend_target()
    example_object.example_splice_targets_above_below()
    example_object.example_cascade_target_A()
    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    example_object.teardown_method()


if __name__ == "__main__":
    example_driver()
