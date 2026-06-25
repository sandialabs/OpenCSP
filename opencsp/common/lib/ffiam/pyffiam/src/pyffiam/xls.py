# Copyright Sandia National Laboratories. All rights reserved.

"""Excel report generation for FFIAM analysis results."""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
import xlsxwriter
from xlsxwriter.workbook import Workbook
from xlsxwriter.worksheet import Worksheet
from xlsxwriter.format import Format

from pyffiam.ffiam_types import AimType
from pyffiam.analysis_data import AnalysisData, PathData


# Column indices for data layout
TITLE_COL: int = 1
UNIT_COL: int = 2
VALUE_COL: int = 3

# Units that require superscript formatting
SUPERSCRIPT_UNITS: list[str] = ['m^2', 'm^3', 'kW/m^2']


class ExcelWriter:
    """Writes FFIAM analysis results to an xlsxwriter Workbook."""

    def __init__(self, workbook: Workbook) -> None:
        self.workbook = workbook

        self.header_fmt = workbook.add_format({'bold': True, 'font_size': 16})
        self.bold_fmt = workbook.add_format({'bold': True})
        self.unit_fmt = workbook.add_format({'align': 'center'})
        self.right_fmt = workbook.add_format({'align': 'right'})
        self.center_fmt = workbook.add_format({'align': 'center'})
        self.bigfloat_fmt = workbook.add_format({'num_format': '#,##0'})
        self.smallfloat_fmt = workbook.add_format({'num_format': '#,##0.00'})
        self.superscript_fmt = workbook.add_format({'font_script': 1})

    def _write_str(self, sheet: Worksheet, row: int, title: str, val: Any) -> int:
        """Write string parameter with label to worksheet."""
        sheet.write(row, TITLE_COL, title)
        sheet.write(row, VALUE_COL, val, self.right_fmt)
        return row + 1

    def _write_int(self, sheet: Worksheet, row: int, title: str, val: int, unit: str = '') -> int:
        """Write integer parameter with label to worksheet."""
        sheet.write(row, TITLE_COL, title)
        if unit in SUPERSCRIPT_UNITS:
            self._write_rich_unit(sheet, row, unit)
        elif unit != '':
            sheet.write(row, UNIT_COL, unit, self.unit_fmt)
        sheet.write(row, VALUE_COL, val)
        return row + 1

    def _write_num(
        self,
        sheet: Worksheet,
        row: int,
        title: str,
        val: float,
        unit: str = '',
        fmt: Optional[Format] = None,
    ) -> int:
        """Write numeric parameter with label to worksheet."""
        if fmt is None:
            fmt = self.bigfloat_fmt
        sheet.write(row, TITLE_COL, title)
        if unit in SUPERSCRIPT_UNITS:
            self._write_rich_unit(sheet, row, unit)
        elif unit != '':
            sheet.write(row, UNIT_COL, unit, self.unit_fmt)
        sheet.write_number(row, VALUE_COL, val, fmt)
        return row + 1

    def _write_rich_unit(self, sheet: Worksheet, row: int, unit: str) -> None:
        """Write unit with superscript exponent to cell."""
        if unit == 'm^2':
            sheet.write_rich_string(row, UNIT_COL, 'm', self.superscript_fmt, "2", self.center_fmt)
        elif unit == 'm^3':
            sheet.write_rich_string(row, UNIT_COL, 'm', self.superscript_fmt, "3", self.center_fmt)
        elif unit == 'kW/m^2':
            sheet.write_rich_string(row, UNIT_COL, 'kW/m', self.superscript_fmt, "2", self.center_fmt)

    def _write_position(
        self,
        sheet: Worksheet,
        row: int,
        pos: NDArray[np.floating],
        fmt: Optional[Format] = None,
    ) -> int:
        """Write XYZ position to worksheet row."""
        if fmt is None:
            fmt = self.bigfloat_fmt
        sheet.write_number(row, 1, pos[0], fmt)
        sheet.write_number(row, 2, pos[1], fmt)
        sheet.write_number(row, 3, pos[2], fmt)
        return row + 1

    def write_results(self, als: AnalysisData, has_gifs: bool = False) -> None:
        """Write the main Results worksheet with metadata and irradiance summary."""
        sheet = self.workbook.add_worksheet(name="Results")

        row = 1

        sheet.set_column(TITLE_COL, TITLE_COL, 30)
        sheet.set_column(UNIT_COL, UNIT_COL, 10)
        sheet.set_column(VALUE_COL, VALUE_COL, 20)

        sheet.write(row, TITLE_COL, f"Irradiance Analysis of {als.name}", self.header_fmt)
        row += 2

        row = self._write_str(sheet, row, "Site", als.name)
        row = self._write_str(sheet, row, "created using", f"FFIAM v{als.ffiam_version}")
        row = self._write_str(sheet, row, "file created", f"{als.created_at.strftime('%b %d, %Y')}")

        row += 1
        sheet.write(row, TITLE_COL, "FIELD PARAMETERS", self.bold_fmt)
        row += 1
        dt_str = als.analysis_dt.strftime('%m/%d/%Y %H:%M')
        row = self._write_str(sheet, row, "Assessed date/time", dt_str)
        row = self._write_str(sheet, row, "Latitude", als.latitude)
        row = self._write_str(sheet, row, "Longitude", als.longitude)
        row = self._write_str(sheet, row, "Timezone", als.timezone)
        row = self._write_int(sheet, row, "Num heliostats", als.num_heliostats)
        row = self._write_num(sheet, row, "Effective field radius", als.field_radius, unit='m')
        row = self._write_num(sheet, row, "Receiver height", als.tower_height, unit='m')

        row = self._write_str(sheet, row, "Aim strategy", str(als.aim_strategy))
        if als.aim_strategy == AimType.Point:
            sheet.write(row, TITLE_COL, "Aim point")
            sheet.write(row, UNIT_COL, "m", self.unit_fmt)
            sheet.write_number(row, VALUE_COL, als.aim_parameters[0], self.bigfloat_fmt)
            sheet.write_number(row, VALUE_COL + 1, als.aim_parameters[1], self.bigfloat_fmt)
            sheet.write_number(row, VALUE_COL + 2, als.aim_parameters[2], self.bigfloat_fmt)
            row += 1
        elif als.aim_strategy in [AimType.Ring, AimType.SplitRing]:
            row = self._write_num(sheet, row, "Ring offset", als.aim_parameters[0], unit='m')
            row = self._write_num(sheet, row, "Ring height", als.aim_parameters[1], unit='m')
        elif als.aim_strategy == AimType.Vector:
            sheet.write(row, TITLE_COL, "Aim vector")
            sheet.write(row, UNIT_COL, "m", self.unit_fmt)
            sheet.write_number(row, VALUE_COL, als.aim_parameters[0], self.smallfloat_fmt)
            sheet.write_number(row, VALUE_COL + 1, als.aim_parameters[1], self.smallfloat_fmt)
            sheet.write_number(row, VALUE_COL + 2, als.aim_parameters[2], self.smallfloat_fmt)
            row += 1

        row += 1
        sheet.write(row, TITLE_COL, "HELIOSTAT PARAMETERS", self.bold_fmt)
        row += 1
        row = self._write_num(sheet, row, "Num facets", als.num_facets)
        row = self._write_str(sheet, row, "Facet rows x cols", f"{int(als.num_facets / als.num_facet_cols)} x {als.num_facet_cols}")
        row = self._write_num(sheet, row, "Facet width", als.facet_width, 'm', self.smallfloat_fmt)
        row = self._write_num(sheet, row, "Facet height", als.facet_height, 'm', self.smallfloat_fmt)
        row = self._write_num(sheet, row, "Total movement", als.movement, 'deg')

        row += 1
        sheet.write(row, TITLE_COL, "AIRSPACE PARAMETERS", self.bold_fmt)
        row += 1
        row = self._write_num(sheet, row, "Minimum altitude", als.min_height, 'm')
        row = self._write_num(sheet, row, "Maximum altitude", als.max_height, 'm')
        row = self._write_num(sheet, row, "Voxel size", als.voxel_size, 'm')

        row = self._write_num(sheet, row, "Voxel volume", als.voxel_volume, unit='m^3')
        row = self._write_num(sheet, row, "Num voxels", als.num_voxels)
        row = self._write_num(sheet, row, "Voxels / altitude", als.num_voxels_x * als.num_voxels_y)

        row += 1
        sheet.write(row, TITLE_COL, "IRRADIANCE RESULTS", self.bold_fmt)
        row += 1

        sheet.write_rich_string(row, TITLE_COL, "Voxels above specified thresholds (kW/m", self.superscript_fmt, "2", "):")
        row += 1
        row = self._write_num(sheet, row, f"# above {als.threshold} (custom)", np.nansum(als.irrads > als.threshold))
        row += 1
        row = self._write_num(sheet, row, "# above 0.1", np.nansum(als.irrads > 0.1))
        row = self._write_num(sheet, row, "# above 1", np.nansum(als.irrads > 1))
        row = self._write_num(sheet, row, "# above 10", np.nansum(als.irrads > 10))
        row = self._write_num(sheet, row, "# above 100", np.nansum(als.irrads > 100))

        scaler = {"x_scale": 0.5, "y_scale": 0.5}

        if als.has_glare and als.has_plots:
            if has_gifs:
                sheet.insert_image("B41", als.en_heatmap_gif, scaler)
                sheet.insert_image("J41", als.en_heatmap, scaler)
                sheet.insert_image("V41", als.en_heatmap_zoomed, scaler)
                sheet.insert_image("B68", als.eu_heatmap_gif, scaler)
                sheet.insert_image("J68", als.eu_heatmap, scaler)
                sheet.insert_image("V68", als.eu_heatmap_zoomed, scaler)
                sheet.insert_image("B95", als.nu_heatmap_gif, scaler)
                sheet.insert_image("J95", als.nu_heatmap, scaler)
                sheet.insert_image("V95", als.nu_heatmap_zoomed, scaler)
            else:
                sheet.insert_image("B41", als.en_heatmap, scaler)
                sheet.insert_image("J41", als.en_heatmap_zoomed, scaler)
                sheet.insert_image("B68", als.eu_heatmap, scaler)
                sheet.insert_image("J68", als.eu_heatmap_zoomed, scaler)
                sheet.insert_image("B95", als.nu_heatmap, scaler)
                sheet.insert_image("J95", als.nu_heatmap_zoomed, scaler)

    def write_heliostat_positions(self, als: AnalysisData) -> None:
        """Write heliostat XYZ positions to a worksheet."""
        sheet = self.workbook.add_worksheet(name='Heliostats')

        sheet.write(1, 1, "HELIOSTAT POSITIONS (m)", self.bold_fmt)
        sheet.write(2, 1, "X")
        sheet.write(2, 2, "Y")
        sheet.write(2, 3, "Z")
        row = 3
        for pos in als.helio_locs:
            row = self._write_position(sheet, row, pos)

    def write_irrad_voxel_positions(self, als: AnalysisData) -> None:
        """Write voxel XYZ positions and irradiance above threshold to a worksheet."""
        sheet = self.workbook.add_worksheet(name='Glaring Voxels')
        sheet.write(1, 1, "VOXEL POSITIONS ABOVE THRESHOLD (m)", self.bold_fmt)
        sheet.write(2, 1, "X")
        sheet.write(2, 2, "Y")
        sheet.write(2, 3, "Z")
        sheet.write(2, 4, "Irradiance (kW/m^2)")
        row = 3
        for i, pos in enumerate(als.threshold_voxel_locs):
            sheet.write_number(row, 4, als.threshold_irrads[i], self.smallfloat_fmt)
            row = self._write_position(sheet, row, pos)

    def write_path_data(self, als: AnalysisData, path: PathData) -> None:
        """Write path voxel positions, irradiance, and embedded plots to a worksheet."""
        sheet = self.workbook.add_worksheet(name=path.name)
        sheet.write(1, 1, "PATH VOXEL LOCATIONS", self.bold_fmt)
        sheet.write(2, 1, "X (m)")
        sheet.write(2, 2, "Y (m)")
        sheet.write(2, 3, "Z (m)")
        sheet.write(2, 4, "Irradiance (kW/m^2)")
        sheet.write_rich_string(2, 3, 'Irradiance (kW/m', self.superscript_fmt, "2", ")")
        row = 3
        for i, pos in enumerate(path.voxel_locs):
            sheet.write_number(row, 4, path.irrads[i], self.smallfloat_fmt)
            row = self._write_position(sheet, row, pos)

        scaler = {"x_scale": 0.5, "y_scale": 0.5}
        sheet.insert_image("H4", path.en_heatmap, scaler)
        sheet.insert_image("T4", path.en_heatmap_zoomed, scaler)
        sheet.insert_image("H31", path.eu_heatmap, scaler)
        sheet.insert_image("T31", path.eu_heatmap_zoomed, scaler)
        sheet.insert_image("H57", path.exposure_cumsum_plot, scaler)


# Legacy module-level API (for backwards compatibility)
HEADER_FORMAT: Optional[Format] = None
BOLD_FORMAT: Optional[Format] = None
UNIT_FORMAT: Optional[Format] = None
RIGHT_FORMAT: Optional[Format] = None
CENTER_FORMAT: Optional[Format] = None
BIGFLOAT_FORMAT: Optional[Format] = None
SMALLFLOAT_FORMAT: Optional[Format] = None
SUPERSCRIPT_FORMAT: Optional[Format] = None

_writer: Optional[ExcelWriter] = None


def initialize(workbook: Workbook) -> None:
    """Initialize the module-level writer. Prefer ExcelWriter directly for new code."""
    global HEADER_FORMAT, BOLD_FORMAT, UNIT_FORMAT, RIGHT_FORMAT
    global BIGFLOAT_FORMAT, SMALLFLOAT_FORMAT, CENTER_FORMAT, SUPERSCRIPT_FORMAT
    global _writer

    _writer = ExcelWriter(workbook)
    HEADER_FORMAT = _writer.header_fmt
    BOLD_FORMAT = _writer.bold_fmt
    UNIT_FORMAT = _writer.unit_fmt
    RIGHT_FORMAT = _writer.right_fmt
    CENTER_FORMAT = _writer.center_fmt
    BIGFLOAT_FORMAT = _writer.bigfloat_fmt
    SMALLFLOAT_FORMAT = _writer.smallfloat_fmt
    SUPERSCRIPT_FORMAT = _writer.superscript_fmt


def write_results(workbook: Workbook, als: AnalysisData, has_gifs: bool = False) -> None:
    """Legacy API. Prefer ExcelWriter.write_results()."""
    if _writer is None:
        raise RuntimeError("xls.initialize() must be called before write_results()")
    _writer.write_results(als, has_gifs)


def write_heliostat_positions(workbook: Workbook, als: AnalysisData) -> None:
    """Legacy API. Prefer ExcelWriter.write_heliostat_positions()."""
    if _writer is None:
        raise RuntimeError("xls.initialize() must be called before write_heliostat_positions()")
    _writer.write_heliostat_positions(als)


def write_irrad_voxel_positions(workbook: Workbook, als: AnalysisData) -> None:
    """Legacy API. Prefer ExcelWriter.write_irrad_voxel_positions()."""
    if _writer is None:
        raise RuntimeError("xls.initialize() must be called before write_irrad_voxel_positions()")
    _writer.write_irrad_voxel_positions(als)


def write_path_data(workbook: Workbook, als: AnalysisData, path: PathData) -> None:
    """Legacy API. Prefer ExcelWriter.write_path_data()."""
    if _writer is None:
        raise RuntimeError("xls.initialize() must be called before write_path_data()")
    _writer.write_path_data(als, path)
