# Copyright Sandia National Laboratories. All rights reserved.

"""Unit tests for the batch-JSON runner pipeline:
- pyffiam.utils.get_site_config_dict_from_json (the parser)
- pyffiam.run_site_analysis_from_file.run_from_config_file (the loop)

Coverage:
- Happy path: a valid JSON parses into an analysis-ready dict.
- Malformed JSON: parser raises JSONDecodeError; runner logs and continues.
- Missing optional sections: parser returns a partial dict (no exception).
- List-shaped JSON: parser prints error and returns empty dict.
- Runner: skips non-.json files, handles missing directory, calls analysis()
  with the parsed dict.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# =============================================================================
# Helpers
# =============================================================================

def _full_json():
    """Minimum JSON matching the UE schema, parseable into analysis() kwargs."""
    return {
        "RunId": "TestSite",
        "SiteType": "Site_Nsttf",
        "Threshold": 5,
        "HelioDesign": {
            "NumFacets": 25,
            "NumCols": 5,
            "FacetWidth": 1.0,
            "FacetHeight": 1.0,
            "FacetFile": "NSTTF_Facet_Centroids.csv",
        },
        "Field": {
            "Latitude": 34.96,
            "Longitude": -106.51,
            "Timezone": -7.0,
            "FieldRadius": 600,
            "MinHeight": 4,
            "MaxHeight": 100,
            "TowerHeight": 61.5,
            "NumHelios": 218,
            "HelioCoordinateFile": "nsttf.csv",
        },
        "DateTime": {"Year": 2025, "Month": 6, "Day": 21, "Hour": 13},
        "AimStrategy": {
            "Type": "Aim_Point",
            "Params": {"X": 0.0, "Y": 0.0, "Z": 100.0},
            "File": "",
        },
    }


def _write_json(path: Path, content):
    if isinstance(content, str):
        path.write_text(content)
    else:
        path.write_text(json.dumps(content))


# =============================================================================
# get_site_config_dict_from_json — JSON parser
# =============================================================================

class TestGetSiteConfigDictFromJson:

    def test_happy_path_yields_analysis_kwargs(self, tmp_path):
        from pyffiam.utils import get_site_config_dict_from_json
        from pyffiam.ffiam_types import CspSite, AimType
        f = tmp_path / "nsttf.json"
        _write_json(f, _full_json())

        result = get_site_config_dict_from_json(f)

        assert result["site"] == CspSite.NSTTF
        assert result["threshold"] == 5
        assert result["n_facets"] == 25
        assert result["facet_w"] == 1.0
        assert result["lat"] == 34.96
        assert result["field_r"] == 600
        assert result["year"] == 2025
        assert result["aim_strat"] == AimType.Point
        assert isinstance(result["aim_params"], np.ndarray)
        assert result["aim_params"].tolist() == [0.0, 0.0, 100.0]

    def test_malformed_json_raises(self, tmp_path):
        from pyffiam.utils import get_site_config_dict_from_json
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            get_site_config_dict_from_json(f)

    def test_list_json_returns_empty_dict(self, tmp_path, capsys):
        from pyffiam.utils import get_site_config_dict_from_json
        f = tmp_path / "list.json"
        _write_json(f, [{"RunId": "site1"}, {"RunId": "site2"}])
        result = get_site_config_dict_from_json(f)
        # Documented behavior: returns empty dict and prints an error.
        assert result == {}
        out = capsys.readouterr().out
        assert "single site" in out.lower() or "list" in out.lower()

    def test_partial_json_skips_missing_sections(self, tmp_path):
        """Each parser section is independent — missing ones leave the dict
        without those keys but don't raise."""
        from pyffiam.utils import get_site_config_dict_from_json
        f = tmp_path / "partial.json"
        _write_json(f, {
            "RunId": "Partial",
            "SiteType": "Site_Custom",
            # No HelioDesign, Field, DateTime, AimStrategy.
        })
        result = get_site_config_dict_from_json(f)
        assert "site" in result
        assert "threshold" in result  # gets default 4
        assert result["threshold"] == 4
        assert "n_facets" not in result
        assert "year" not in result
        assert "aim_strat" not in result

    def test_custom_threshold_value_preserved(self, tmp_path):
        from pyffiam.utils import get_site_config_dict_from_json
        f = tmp_path / "thr.json"
        data = _full_json()
        data["Threshold"] = 7.5
        _write_json(f, data)
        result = get_site_config_dict_from_json(f)
        assert result["threshold"] == 7.5

    def test_create_gifs_boolean_accepted(self, tmp_path):
        from pyffiam.utils import get_site_config_dict_from_json
        f = tmp_path / "g.json"
        data = _full_json()
        data["create_gifs"] = True
        _write_json(f, data)
        result = get_site_config_dict_from_json(f)
        assert result["create_gifs"] is True

    def test_create_gifs_non_boolean_falls_back_to_false(self, tmp_path):
        from pyffiam.utils import get_site_config_dict_from_json
        f = tmp_path / "g.json"
        data = _full_json()
        data["create_gifs"] = "yes"
        _write_json(f, data)
        result = get_site_config_dict_from_json(f)
        assert result["create_gifs"] is False


# =============================================================================
# run_from_config_file — batch loop
# =============================================================================

class TestRunFromConfigFile:

    def test_missing_loaded_dir_prints_error_and_returns(self, tmp_path, monkeypatch, capsys):
        """The runner uses the script-relative path; if site_configs/loaded/
        doesn't exist, it prints an error and returns without raising."""
        from pyffiam import run_site_analysis_from_file as runner
        # Point __file__ at a directory with no site_configs/loaded.
        monkeypatch.setattr(runner, "__file__", str(tmp_path / "fake_module.py"))
        runner.run_from_config_file()
        out = capsys.readouterr().out
        assert "Loaded directory not found" in out

    def test_empty_loaded_dir_prints_message(self, tmp_path, monkeypatch, capsys):
        """Empty directory: prints "No configuration files found" and returns."""
        from pyffiam import run_site_analysis_from_file as runner
        # Create empty site_configs/loaded/ inside tmp.
        (tmp_path / "site_configs" / "loaded").mkdir(parents=True)
        monkeypatch.setattr(runner, "__file__", str(tmp_path / "fake_module.py"))
        runner.run_from_config_file()
        out = capsys.readouterr().out
        assert "No configuration files" in out

    def test_skips_non_json_files(self, tmp_path, monkeypatch, capsys):
        """Files without a .json extension are skipped with a message."""
        from pyffiam import run_site_analysis_from_file as runner
        loaded = tmp_path / "site_configs" / "loaded"
        loaded.mkdir(parents=True)
        (loaded / "readme.txt").write_text("not a config")
        monkeypatch.setattr(runner, "__file__", str(tmp_path / "fake_module.py"))
        runner.run_from_config_file()
        out = capsys.readouterr().out
        assert "Skipping non-JSON" in out

    def test_malformed_json_does_not_abort_batch(self, tmp_path, monkeypatch, capsys):
        """A malformed JSON file logs an error and the loop moves on. With one
        bad file and no good files, analysis() should never be called."""
        from pyffiam import run_site_analysis_from_file as runner
        loaded = tmp_path / "site_configs" / "loaded"
        loaded.mkdir(parents=True)
        (loaded / "bad.json").write_text("{not valid")
        monkeypatch.setattr(runner, "__file__", str(tmp_path / "fake_module.py"))
        # Patch analysis to detect any unexpected invocation.
        with patch.object(runner, "analysis") as mock_analysis:
            runner.run_from_config_file()
            mock_analysis.assert_not_called()
        out = capsys.readouterr().out
        assert "Error decoding JSON" in out

    def test_valid_json_calls_analysis_with_parsed_dict(self, tmp_path, monkeypatch, capsys):
        """A valid JSON file should reach analysis() with the parsed kwargs."""
        from pyffiam import run_site_analysis_from_file as runner
        loaded = tmp_path / "site_configs" / "loaded"
        loaded.mkdir(parents=True)
        _write_json(loaded / "site.json", _full_json())
        monkeypatch.setattr(runner, "__file__", str(tmp_path / "fake_module.py"))

        with patch.object(runner, "analysis") as mock_analysis:
            mock_analysis.return_value = None
            runner.run_from_config_file()
            mock_analysis.assert_called_once()
            kwargs = mock_analysis.call_args.kwargs
            # RunId is popped before analysis() is called.
            assert "RunId" not in kwargs
            assert kwargs.get("threshold") == 5
            assert kwargs.get("open_output_dir") is True

    def test_analysis_exception_does_not_abort_batch(self, tmp_path, monkeypatch, capsys):
        """If analysis() raises, the loop catches, logs, and continues to the
        next file."""
        from pyffiam import run_site_analysis_from_file as runner
        loaded = tmp_path / "site_configs" / "loaded"
        loaded.mkdir(parents=True)
        _write_json(loaded / "a.json", _full_json())
        _write_json(loaded / "b.json", _full_json())
        monkeypatch.setattr(runner, "__file__", str(tmp_path / "fake_module.py"))

        with patch.object(runner, "analysis", side_effect=RuntimeError("boom")) as mock_analysis:
            runner.run_from_config_file()
            assert mock_analysis.call_count == 2  # both files attempted
        out = capsys.readouterr().out
        assert "Error running analysis" in out
