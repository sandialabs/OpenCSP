"""
Routines supporting OpenCSP automatic tests.
"""

import os
import matplotlib.testing.compare as mplt

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
import opencsp.common.lib.target.target_image as ti
import opencsp.common.lib.csp.SolarField as sf


def lines_share_common_string(line_1: str, line_2: str, ignore: str) -> bool:
    if (line_1.find(ignore) != -1) and (line_2.find(ignore) != -1):
        # Then both lines contain the ignore string.
        return True
    else:
        return False


def lines_share_an_ignore_string(line_1: str, line_2: str, ignore_string_list: list[str]) -> bool:
    for ignore_string in ignore_string_list:
        if lines_share_common_string(line_1, line_2, ignore_string):
            return True
    # We fell through the loop, and no common ignore string was found.
    return False


def svg_lines_are_equal(line_1: str, line_2: str) -> bool:
    """
    Compares two strings and determines whether they correspond to equivalent lines in an svg file.
    Two lines are considered equivalent if they are compatible with an equivalent image.
    Differences due to generation at a different time are ignored.
    """
    if line_1 == line_2:
        return True
    elif len(line_1) != len(line_2):
        return False
    # svg string analysis.
    if lines_share_an_ignore_string(line_1, line_2, ['<dc:date>', 'clip-path', 'clipPath', 'path id', '<dc:title>']):
        return True
    else:
        # Walk the lines comparing characters, ignoring hashed addresses.
        ignore = False
        for c1, c2 in zip(line_1, line_2):
            if ignore == True:
                if (c1 == c2) and ((c1 == '"') or (c1 == ')') or (c1 == ';')):
                    ignore = False
            else:
                if (c1 == c2) and (c1 == '#'):
                    ignore = True
                else:
                    # We are not ignoring and the current character is not an escape character.
                    if c1 != c2:
                        # We found a counterexample showing the images are not equal.
                        return False
                    else:
                        # The characters are the same, so continue down the line.
                        pass
        # We fell through the loop with no counterexample found.
        return True


def compare_svg_files(svg_dir_body_ext_1: str, svg_dir_body_ext_2: str) -> bool:
    """
    Compares two svg files, and determines whether they correspond to the same image.
    This requires ignoring differences that result from two files being created with
    the same content but at different times.

    Returns True if the files are image-equivalent.

    Assumes the inputfiles are svg files, without checking.
    """
    # Verify both files have the same size.
    size_1 = ft.file_size(svg_dir_body_ext_1)
    size_2 = ft.file_size(svg_dir_body_ext_2)
    if size_1 != size_2:
        return False
    # Verify both files have the same number of lines.
    with open(svg_dir_body_ext_1, newline='') as input_stream_1:
        lines_1 = input_stream_1.readlines()
    with open(svg_dir_body_ext_2, newline='') as input_stream_2:
        lines_2 = input_stream_2.readlines()
    if len(lines_1) != len(lines_2):
        return False
    # Verify both files have the same contents.
    for line_1, line_2 in zip(lines_1, lines_2):
        lines_are_equal = svg_lines_are_equal(line_1, line_2)
        if not lines_are_equal:
            return False
    # We fell through the loop with no unequal lines found.
    return True


def compare_txt_files(expected_file: str, actual_file: str) -> bool:
    assert_msg = '\nexpected_file: ' + expected_file + '\nactual_file: ' + actual_file

    # Read file contents
    with open(expected_file, newline='') as expected_file_stream:
        lines_expected = expected_file_stream.readlines()
    with open(actual_file, newline='') as actual_file_stream:
        lines_actual = actual_file_stream.readlines()

    assert len(lines_expected) == len(lines_actual), assert_msg

    # Compare lines of the file ignoring trailing whitespace such as carriage returns
    for line_expected, line_actual in zip(lines_expected, lines_actual):
        assert line_expected.rstrip() == line_actual.rstrip(), assert_msg

    # We fell through the loop with no unequal lines found.
    return True


def verify_output_file_matches_expected(file_created: str, actual_output_dir: str, expected_output_dir: str) -> None:
    """
    Verifies that the actual output file matches what's expected.
    """
    # Construct actual and created fully qualified path names.
    created_dir, created_body, created_ext = ft.path_components(file_created)
    created_body_ext = created_body + created_ext
    actual_dir_body_ext = os.path.join(actual_output_dir, created_body_ext)
    expected_dir_body_ext = os.path.join(expected_output_dir, created_body_ext)
    # Verify both files exist.
    if not ft.file_exists(actual_dir_body_ext):
        lt.error_and_raise(
            FileNotFoundError,
            'In verify_output_file_matches_expected(), actual file does not exist.\n'
            '    actual_dir_body_ext = ' + str(actual_dir_body_ext),
        )
    if not ft.file_exists(expected_dir_body_ext):
        lt.error_and_raise(
            FileNotFoundError,
            'In verify_output_file_matches_expected(), expected file does not exist.\n'
            '    expected_dir_body_ext = ' + str(expected_dir_body_ext),
        )
    # Verify both files are equal.
    if created_ext == '.svg':
        svg_files_are_equal = compare_svg_files(actual_dir_body_ext, expected_dir_body_ext)
        if not svg_files_are_equal:
            lt.error_and_raise(
                ValueError,
                'In verify_output_file_matches_expected(), actual and expected .svg files are not equal, after ignoring allowable differences.\n'
                '    actual_dir_body_ext   = ' + str(actual_dir_body_ext) + '\n'
                '    expected_dir_body_ext = ' + str(expected_dir_body_ext),
            )
    elif created_ext == '.png':
        png_files_are_equal = compare_actual_expected_images(actual_dir_body_ext, expected_dir_body_ext)
        if png_files_are_equal is not None:
            lt.error_and_raise(ValueError, 'In verify_output_file_matches_expected(), ' + png_files_are_equal)

    else:
        files_are_equal = compare_txt_files(actual_dir_body_ext, expected_dir_body_ext)
        if not files_are_equal:
            lt.error_and_raise(
                ValueError,
                'In verify_output_file_matches_expected(), actual and expected files are not exactly equal.\n'
                '    actual_dir_body_ext   = ' + str(actual_dir_body_ext) + '\n'
                '    expected_dir_body_ext = ' + str(expected_dir_body_ext),
            )


def verify_output_files_match_expected(
    files_created: list[str], actual_output_dir: str, expected_output_dir: str
) -> None:
    """
    Verifies that all of the output files match what's expected.
    """
    for file_created in files_created:
        verify_output_file_matches_expected(file_created, actual_output_dir, expected_output_dir)


def show_save_and_check_figure(
    fig_record: RenderControlFigureRecord,
    actual_output_dir: str,
    expected_output_dir: str,
    verify: bool,
    show_figs: bool = True,
    dpi=600,
) -> None:
    """
    Once a figure is drawn, this routine does what's needed to wrap up the test.
    """
    # Show.
    if show_figs:
        fig_record.view.show(
            equal=fig_record.equal,
            x_limits=fig_record.x_limits,
            y_limits=fig_record.y_limits,
            z_limits=fig_record.z_limits,
        )
    # Save.
    files_created = fig_record.save(actual_output_dir, format='png', dpi=dpi)  # Filename inferred from figure title.
    # Check.
    if verify:
        verify_output_files_match_expected(files_created, actual_output_dir, expected_output_dir)


def save_and_check_image(
    image,  # ?? SCAFFOLDING RCB -- ADD TYPE CUE
    output_dpm,  # ?? SCAFFOLDING RCB -- ADD TYPE CUE
    actual_output_dir: str,
    expected_output_dir: str,
    output_file_body: str,
    output_ext: str,
    verify: bool,
) -> None:
    """
    Once an image is generated, this routine does what's needed to wrap up the test.
    """
    # Save.
    file_created = ti.save_image(
        image, output_dpm, actual_output_dir, output_file_body, output_ext
    )  # ?? SCAFFOLDING RCB -- REPLACE WITH GENERAL IMAGE SAVE ROUTINE
    files_created = [file_created]
    # Check.
    if verify:
        verify_output_files_match_expected(files_created, actual_output_dir, expected_output_dir)


def compare_actual_expected_images(actual_location: str, expected_location: str, tolerance=0.2):
    return mplt.compare_images(expected_location, actual_location, tolerance)


def load_solar_field_partition(heliostat_names: list, partitioned_csv_file_name: str) -> sf.SolarField:
    import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
    import opencsp.common.lib.geo.lon_lat_nsttf as lln
    import csv

    # Load the CSV into a dictionary
    with open(dpft.sandia_nsttf_test_heliostats_origin_file(), 'r', newline='') as infile:
        reader = csv.reader(infile)
        dict = {rows[0]: rows[0:-1] for rows in reader}

    # Create a partition of the csv
    with open(partitioned_csv_file_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict['Name'])
        for heliostat_name in heliostat_names:
            writer.writerow(dict[heliostat_name])

    return sf.sf_from_csv_files(
        name='Sandia NSTTF Partition',
        short_name='NSTTF',
        origin_lon_lat=lln.NSTTF_ORIGIN,
        heliostat_file=partitioned_csv_file_name,
        facet_centroids_file=dpft.sandia_nsttf_test_facet_centroidsfile(),
    )
