"""Dataclasses tracking generated FFIAM output files (plots, GIFs, Excel reports)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pyffiam.app_config import FFIAM_VERSION


@dataclass
class PlotPaths:
    """Paths to generated plot files for a single dimension pair."""

    heatmap: Optional[Path] = None
    heatmap_zoomed: Optional[Path] = None
    heatmap_gif: Optional[Path] = None


@dataclass
class AnalysisOutput:
    """Generated output file paths from a FFIAM analysis run."""

    output_directory: Path

    xls_filename: Optional[str] = None
    xls_file: Optional[Path] = None

    # EN / EU / NU heatmaps (full, zoomed, animated)
    en_heatmap: Optional[Path] = None
    en_heatmap_zoomed: Optional[Path] = None
    en_heatmap_gif: Optional[Path] = None

    eu_heatmap: Optional[Path] = None
    eu_heatmap_zoomed: Optional[Path] = None
    eu_heatmap_gif: Optional[Path] = None

    nu_heatmap: Optional[Path] = None
    nu_heatmap_zoomed: Optional[Path] = None
    nu_heatmap_gif: Optional[Path] = None

    agg_path_eu_heatmap: Optional[Path] = None

    ffiam_version: str = FFIAM_VERSION

    @property
    def has_plots(self) -> bool:
        return self.en_heatmap is not None

    @property
    def has_gifs(self) -> bool:
        return self.en_heatmap_gif is not None

    @property
    def has_xls(self) -> bool:
        return self.xls_file is not None and self.xls_file.exists()

    @classmethod
    def create_for_session(cls, app_result_dir: Path, session_name: str) -> 'AnalysisOutput':
        """Create an AnalysisOutput and mkdir the session output directory."""
        output_dir = app_result_dir / session_name
        output_dir.mkdir(parents=True, exist_ok=True)

        xls_filename = f"{session_name}.xlsx"

        return cls(output_directory=output_dir, xls_filename=xls_filename, xls_file=output_dir / xls_filename)

    def set_en_plots(self, heatmap: Optional[Path], heatmap_zoomed: Optional[Path], heatmap_gif: Optional[Path] = None):
        self.en_heatmap = heatmap
        self.en_heatmap_zoomed = heatmap_zoomed
        self.en_heatmap_gif = heatmap_gif

    def set_eu_plots(self, heatmap: Optional[Path], heatmap_zoomed: Optional[Path], heatmap_gif: Optional[Path] = None):
        self.eu_heatmap = heatmap
        self.eu_heatmap_zoomed = heatmap_zoomed
        self.eu_heatmap_gif = heatmap_gif

    def set_nu_plots(self, heatmap: Optional[Path], heatmap_zoomed: Optional[Path], heatmap_gif: Optional[Path] = None):
        self.nu_heatmap = heatmap
        self.nu_heatmap_zoomed = heatmap_zoomed
        self.nu_heatmap_gif = heatmap_gif
