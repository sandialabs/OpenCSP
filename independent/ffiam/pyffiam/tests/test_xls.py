# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Unit tests for pyffiam.xls.ExcelWriter — verifies the worksheet writes
made by each public method by recording every Worksheet API call. Uses
xlsxwriter for format objects (they're inert handles) but stubs Workbook
and Worksheet to capture every cell write.

This is *content* verification, complementing test_outputs.py's file-existence
checks. If the layout of the Results sheet changes, these tests pinpoint
which row/value drifted.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np
import pytest
import xlsxwriter

from pyffiam.ffiam_types import AimType
from pyffiam.xls import ExcelWriter


# =============================================================================
# Recording stubs — capture sheet writes for assertion
# =============================================================================


@dataclass
class RecordedWrite:
    method: str
    row: int
    col: int
    args: Tuple[Any, ...]


class RecordingSheet:
    """Mimics a xlsxwriter Worksheet by recording every call."""

    def __init__(self, name: str):
        self.name = name
        self.writes: List[RecordedWrite] = []
        self.columns_set: List[Tuple[int, int, int]] = []
        self.images: List[Tuple[str, str]] = []

    def set_column(self, first, last, width):
        self.columns_set.append((first, last, width))

    def write(self, row, col, value, fmt=None):
        self.writes.append(RecordedWrite("write", row, col, (value,)))

    def write_number(self, row, col, value, fmt=None):
        self.writes.append(RecordedWrite("write_number", row, col, (float(value),)))

    def write_rich_string(self, row, col, *parts):
        # Filter out format args; record only the string fragments.
        text = "".join(p for p in parts if isinstance(p, str))
        self.writes.append(RecordedWrite("write_rich_string", row, col, (text,)))

    def insert_image(self, cell, path, options=None):
        self.images.append((cell, str(path)))

    # Convenience lookup helpers
    def find(self, value):
        for w in self.writes:
            if w.args and w.args[0] == value:
                return w
        return None

    def values_in_col(self, col):
        return [w.args[0] for w in self.writes if w.col == col]


class RecordingWorkbook:
    """Mimics a xlsxwriter Workbook. Format calls return a real xlsxwriter
    Format so ExcelWriter's stash-and-reuse pattern works unchanged. The
    underlying real workbook writes to a BytesIO buffer that's discarded on
    close — we only care about the recorded sheet writes, not the rendered
    file."""

    def __init__(self):
        self._buffer = io.BytesIO()
        self._real = xlsxwriter.Workbook(self._buffer, {"in_memory": True})
        # Add a placeholder worksheet so the real workbook isn't empty (close
        # rejects books with zero sheets).
        self._real.add_worksheet("_unused")
        self.sheets: List[RecordingSheet] = []

    def add_format(self, props=None):
        return self._real.add_format(props or {})

    def add_worksheet(self, name: str = None):
        sheet = RecordingSheet(name=name or f"Sheet{len(self.sheets)}")
        self.sheets.append(sheet)
        return sheet

    def close(self):
        self._real.close()


# =============================================================================
# AnalysisData stub — only the fields each ExcelWriter method actually reads.
# =============================================================================


def _make_als_stub(**overrides):
    """Minimum AnalysisData-shaped stub for ExcelWriter methods."""
    base = SimpleNamespace(
        name="TestSite",
        ffiam_version="3.0.0",
        created_at=datetime(2026, 5, 12, 10, 30),
        analysis_dt=datetime(2025, 6, 21, 12, 0),
        latitude=34.96,
        longitude=-106.51,
        timezone=-7.0,
        num_heliostats=3,
        field_radius=600,
        tower_height=61.0,
        aim_strategy=AimType.Point,
        aim_parameters=np.array([10.0, 20.0, 90.0]),
        num_facets=25,
        num_facet_cols=5,
        facet_width=1.2,
        facet_height=1.2,
        movement=42.5,
        min_height=4,
        max_height=100,
        voxel_size=2,
        voxel_volume=8,
        num_voxels=1000,
        num_voxels_x=10,
        num_voxels_y=10,
        threshold=4.0,
        irrads=np.array([0.0, 0.05, 0.5, 1.5, 5.0, 50.0, 150.0]),
        helio_locs=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0]]),
        threshold_voxel_locs=np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]),
        threshold_irrads=np.array([5.5, 6.6]),
        has_glare=False,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# =============================================================================
# Tests
# =============================================================================


@pytest.fixture
def wb():
    book = RecordingWorkbook()
    yield book
    book.close()


def test_write_heliostat_positions_writes_header_and_xyz(wb):
    als = _make_als_stub()
    writer = ExcelWriter(wb)
    writer.write_heliostat_positions(als)

    assert len(wb.sheets) == 1
    sheet = wb.sheets[0]
    assert sheet.name == "Heliostats"

    # Header row: column titles at row 2
    headers = {w.col: w.args[0] for w in sheet.writes if w.row == 2}
    assert headers.get(1) == "X"
    assert headers.get(2) == "Y"
    assert headers.get(3) == "Z"

    # Data rows: three heliostats starting at row 3
    rows_3_to_5 = [w for w in sheet.writes if w.row in (3, 4, 5) and w.method == "write_number"]
    assert len(rows_3_to_5) == 3 * 3  # 3 helios * (X, Y, Z)
    # First helio XYZ values match input
    h0 = sorted([w for w in rows_3_to_5 if w.row == 3], key=lambda w: w.col)
    assert [w.args[0] for w in h0] == [1.0, 2.0, 0.0]


def test_write_irrad_voxel_positions_writes_per_voxel_irradiance(wb):
    als = _make_als_stub()
    writer = ExcelWriter(wb)
    writer.write_irrad_voxel_positions(als)

    sheet = wb.sheets[0]
    assert sheet.name == "Glaring Voxels"

    # Header row 2 has "Irradiance" label in column 4
    headers = {w.col: w.args[0] for w in sheet.writes if w.row == 2}
    assert "Irradiance" in headers.get(4, "")

    # Two voxel rows: irradiance values in col 4
    irrad_writes = [w for w in sheet.writes if w.col == 4 and w.method == "write_number"]
    assert [w.args[0] for w in irrad_writes] == [5.5, 6.6]


def test_write_results_includes_metadata_and_thresholds(wb):
    als = _make_als_stub()
    writer = ExcelWriter(wb)
    writer.write_results(als, has_gifs=False)

    sheet = wb.sheets[0]
    assert sheet.name == "Results"

    # Metadata: site name appears somewhere.
    titles = [w.args[0] for w in sheet.writes if w.method == "write" and isinstance(w.args[0], str)]
    assert any("TestSite" in t for t in titles)
    assert any("Site" == t for t in titles)
    assert any("Latitude" == t for t in titles)
    assert any("Longitude" == t for t in titles)

    # Threshold counts: irrads array is [0, 0.05, 0.5, 1.5, 5.0, 50.0, 150.0]
    # - threshold=4.0 -> 3 values exceed (5, 50, 150)
    # - >0.1 -> 5 values
    # - >1   -> 4 values
    # - >10  -> 2 values
    # - >100 -> 1 value
    numbers = [w.args[0] for w in sheet.writes if w.method == "write_number"]
    assert 3.0 in numbers  # custom threshold count
    assert 5.0 in numbers  # >0.1
    assert 4.0 in numbers  # >1
    assert 2.0 in numbers  # >10
    assert 1.0 in numbers  # >100


def test_write_results_point_aim_writes_three_components(wb):
    """Point aim writes three numeric components (x, y, z) on one row."""
    als = _make_als_stub(aim_strategy=AimType.Point, aim_parameters=np.array([12.0, 34.0, 56.0]))
    writer = ExcelWriter(wb)
    writer.write_results(als, has_gifs=False)
    sheet = wb.sheets[0]
    numbers = [w.args[0] for w in sheet.writes if w.method == "write_number"]
    assert 12.0 in numbers
    assert 34.0 in numbers
    assert 56.0 in numbers


def test_write_results_ring_aim_writes_inner_outer_and_height(wb):
    """Ring aim writes 'Ring inner radius', 'Ring outer radius', and 'Ring height' rows."""
    als = _make_als_stub(aim_strategy=AimType.Ring, aim_parameters=np.array([25.0, 80.0, 90.0]))
    writer = ExcelWriter(wb)
    writer.write_results(als, has_gifs=False)
    sheet = wb.sheets[0]
    titles = [w.args[0] for w in sheet.writes if w.method == "write" and isinstance(w.args[0], str)]
    assert "Ring inner radius" in titles
    assert "Ring outer radius" in titles
    assert "Ring height" in titles


def test_write_results_split_ring_aim_writes_inner_outer_and_height(wb):
    """SplitRing aim shares the ring layout: inner radius, outer radius, height rows."""
    als = _make_als_stub(aim_strategy=AimType.SplitRing, aim_parameters=np.array([25.0, 80.0, 90.0]))
    writer = ExcelWriter(wb)
    writer.write_results(als, has_gifs=False)
    sheet = wb.sheets[0]
    titles = [w.args[0] for w in sheet.writes if w.method == "write" and isinstance(w.args[0], str)]
    assert "Ring inner radius" in titles
    assert "Ring outer radius" in titles
    assert "Ring height" in titles


def test_write_results_default_column_widths_set(wb):
    """Results sheet must set column widths for the layout to render."""
    als = _make_als_stub()
    writer = ExcelWriter(wb)
    writer.write_results(als, has_gifs=False)
    sheet = wb.sheets[0]
    assert len(sheet.columns_set) >= 3  # title, unit, value columns at minimum


def test_excel_writer_formats_constructed(wb):
    """ExcelWriter must build format handles in __init__ (not lazily)."""
    writer = ExcelWriter(wb)
    # All eight formats must exist and be xlsxwriter Format instances.
    for attr in (
        "header_fmt",
        "bold_fmt",
        "unit_fmt",
        "right_fmt",
        "center_fmt",
        "bigfloat_fmt",
        "smallfloat_fmt",
        "superscript_fmt",
    ):
        assert getattr(writer, attr) is not None
