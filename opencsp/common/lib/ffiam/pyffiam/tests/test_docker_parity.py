r"""
Docker vs native parity tests. Runs the same analysis in both environments
and compares numeric results. Reports all metrics before asserting.

    cd T:\ffiam\ffiam\pyffiam
    .\.venv\Scripts\python.exe -m pytest tests\test_docker_parity.py -v -s
    .\.venv\Scripts\python.exe -m pytest tests\test_docker_parity.py -v -s -k nsttf_point
"""

import json
import logging
import os
import subprocess
import unittest
import numpy as np

from pyffiam.analysis import analysis
from pyffiam.ffiam_types import CspSite, AimType

log = logging.getLogger(__name__)

DOCKER_IMAGE = "ffiam"

# cross-platform tolerance (MSVC/arch52 vs GCC/arch75+)
PEAK_RTOL = 0.005
TOTAL_RTOL = 0.005
GLARING_RTOL = 0.005

_DOCKER_SNIPPET = """\
import json, numpy as np
from pyffiam.analysis import analysis
from pyffiam.ffiam_types import CspSite, AimType
r = analysis({params}, create_plots=False, create_xls=False, create_gifs=False, open_output_dir=False)
print("RESULT:" + json.dumps({{
    "peak": float(r.peak_irrad),
    "total": float(r.total_irrad),
    "glaring": int(r.n_glaring_voxels),
    "n_helios": int(r.num_heliostats),
    "n_facets": int(r.num_facets),
    "total_thresh": float(r.total_irrad_threshold),
}}))
"""

RESULT_KEYS = ["peak", "total", "total_thresh", "glaring", "n_helios", "n_facets"]


def _run_docker(params_str):
    snippet = _DOCKER_SNIPPET.format(params=params_str)
    proc = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", DOCKER_IMAGE, "-c", snippet],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Docker failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):]), proc.stdout
    raise RuntimeError(f"No RESULT line in Docker output:\n{proc.stdout[-3000:]}")


def _run_native(**kwargs):
    r = analysis(**kwargs, create_plots=False, create_xls=False, create_gifs=False, open_output_dir=False)
    return {
        "peak": float(r.peak_irrad),
        "total": float(r.total_irrad),
        "glaring": int(r.n_glaring_voxels),
        "n_helios": int(r.num_heliostats),
        "n_facets": int(r.num_facets),
        "total_thresh": float(r.total_irrad_threshold),
    }


def _pct(a, b):
    if a == 0:
        return float('inf') if b != 0 else 0.0
    return abs(a - b) / abs(a) * 100


def _log_comparison(label, native, docker):
    lines = [
        "",
        "=" * 64,
        f"  {label}",
        "=" * 64,
        f"  {'metric':<20} {'native':>14} {'docker':>14} {'diff%':>8}",
        f"  {'-' * 60}",
    ]
    for key in RESULT_KEYS:
        n, d = native[key], docker[key]
        pct = _pct(n, d)
        flag = " <<<" if pct > 1.0 else ""
        if isinstance(n, int):
            lines.append(f"  {key:<20} {n:>14,} {d:>14,} {pct:>7.2f}%{flag}")
        else:
            lines.append(f"  {key:<20} {n:>14,.2f} {d:>14,.2f} {pct:>7.2f}%{flag}")
    lines.append("")
    log.warning("\n".join(lines))


CASES = [
    {
        "name": "nsttf_point",
        "kwargs": dict(
            site=CspSite.NSTTF, year=2025, month=6, day=21, hour=13,
            aim_strat=AimType.Point, aim_params=np.array([0, 0, 100]),
            threshold=4, voxel_size=4,
        ),
        "params_str": (
            "site=CspSite.NSTTF, year=2025, month=6, day=21, hour=13, "
            "aim_strat=AimType.Point, aim_params=np.array([0,0,100]), "
            "threshold=4, voxel_size=4"
        ),
    },
    {
        "name": "nsttf_ring",
        "kwargs": dict(
            site=CspSite.NSTTF, year=2025, month=6, day=21, hour=13,
            aim_strat=AimType.Ring, aim_params=np.array([30, 90, 0]),
            threshold=4, voxel_size=4,
        ),
        "params_str": (
            "site=CspSite.NSTTF, year=2025, month=6, day=21, hour=13, "
            "aim_strat=AimType.Ring, aim_params=np.array([30,90,0]), "
            "threshold=4, voxel_size=4"
        ),
    },
    {
        "name": "sample_v1_point",
        "kwargs": dict(
            site=CspSite.SampleV1, year=2025, month=6, day=21, hour=13,
            aim_strat=AimType.Point, aim_params=np.array([0, 0, 100]),
            threshold=4, voxel_size=4,
        ),
        "params_str": (
            "site=CspSite.SampleV1, year=2025, month=6, day=21, hour=13, "
            "aim_strat=AimType.Point, aim_params=np.array([0,0,100]), "
            "threshold=4, voxel_size=4"
        ),
    },
    {
        "name": "crescent_dunes_point",
        "kwargs": dict(
            site=CspSite.CrescentDunes, year=2025, month=6, day=21, hour=13,
            aim_strat=AimType.Point, aim_params=np.array([0, 0, 100]),
            threshold=4, voxel_size=4,
        ),
        "params_str": (
            "site=CspSite.CrescentDunes, year=2025, month=6, day=21, hour=13, "
            "aim_strat=AimType.Point, aim_params=np.array([0,0,100]), "
            "threshold=4, voxel_size=4"
        ),
    },
    {
        "name": "crescent_dunes_stow",
        "kwargs": dict(
            site=CspSite.CrescentDunes, year=2025, month=6, day=21, hour=13,
            aim_strat=AimType.Vector, aim_params=np.array([0, 0, 1]),
            threshold=4, voxel_size=4,
        ),
        "params_str": (
            "site=CspSite.CrescentDunes, year=2025, month=6, day=21, hour=13, "
            "aim_strat=AimType.Vector, aim_params=np.array([0,0,1]), "
            "threshold=4, voxel_size=4"
        ),
    },
]


class DockerParityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.path.exists("/.dockerenv"):
            raise unittest.SkipTest("Skipping docker-parity tests when running inside a container.")
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise unittest.SkipTest(f"Docker image '{DOCKER_IMAGE}' not found.")


def _make_test(case):
    def test_method(self):
        label = case["name"]
        native = _run_native(**case["kwargs"])
        docker, docker_stdout = _run_docker(case["params_str"])

        _log_comparison(label, native, docker)

        parse_errors = [l for l in docker_stdout.splitlines() if "parse error" in l.lower()]
        if parse_errors:
            log.warning("Docker emitted %d CSV parse error(s)", len(parse_errors))

        failures = []

        # heliostat/facet counts must be exact
        if native["n_helios"] != docker["n_helios"]:
            failures.append(f"n_helios: {native['n_helios']} vs {docker['n_helios']}")
        if native["n_facets"] != docker["n_facets"]:
            failures.append(f"n_facets: {native['n_facets']} vs {docker['n_facets']}")

        for key, rtol in [("peak", PEAK_RTOL), ("total", TOTAL_RTOL)]:
            n, d = native[key], docker[key]
            if n == 0 and d == 0:
                continue
            rdiff = abs(n - d) / max(abs(n), abs(d))
            if rdiff > rtol:
                failures.append(f"{key}: {n:,.2f} vs {d:,.2f} ({rdiff:.1%} > {rtol:.1%})")

        ng, dg = native["glaring"], docker["glaring"]
        if ng > 0 or dg > 0:
            gdiff = abs(ng - dg) / max(ng, dg)
            if gdiff > GLARING_RTOL:
                failures.append(f"glaring: {ng:,} vs {dg:,} ({gdiff:.1%} > {GLARING_RTOL:.1%})")

        if failures:
            self.fail(f"{label}:\n" + "\n".join(f"  - {f}" for f in failures))

    return test_method


for _case in CASES:
    setattr(DockerParityTest, f"test_{_case['name']}", _make_test(_case))


if __name__ == "__main__":
    unittest.main(verbosity=2)
