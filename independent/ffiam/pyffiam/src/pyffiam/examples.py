"""Example analyses for preset sites (NSTTF, radial, SampleV2)."""

from pyffiam.analysis import analysis
from pyffiam.ffiam_types import CspSite, AimType
import numpy as np


def assess_radial():
    result = analysis(
        site=CspSite.Radial,
        year=2025,
        month=6,
        day=21,
        hour=12,
        threshold=2,
        # threshold=0.1,
        create_gifs=False,
        create_xls=True,
        # aim_strat=AimType.Point,
        # aim_params=np.array([0, 0, 120]),
        aim_strat=AimType.Ring,
        aim_params=np.array([60, 100, 120]),
        # aim_strat=AimType.Vector,
        # aim_params=np.array([0, 0, 1]),
        paths=[
            [
                (-1000, 800, 60),
                (-820, 850, 60),
                (-700, 820, 60),
                (-550, 700, 60),
                (-400, 400, 60),
                (-300, 270, 60),
                (-100, 75, 60),
                (50, 50, 80),
                (150, -200, 80),
                (350, -250, 80),
                (500, -30, 80),
                (650, 200, 130),
                (800, 210, 130),
                (900, 150, 130),
                (1000, 300, 130),
            ]
        ],
        path_speeds=[10],
        open_output_dir=True,
    )
    print(f"Radial site total irradiance: {result.total_irrad:,.0f}")


def assess_radial_small():
    result = analysis(
        site=CspSite.RadialSmall,
        year=2025,
        month=6,
        day=21,
        hour=12,
        threshold=2,
        # threshold=0.1,
        create_gifs=False,
        create_xls=True,
        # aim_strat=AimType.Point,
        # aim_params=np.array([0, 0, 120]),
        aim_strat=AimType.Ring,
        aim_params=np.array([60, 100, 120]),
        # aim_strat=AimType.Vector,
        # aim_params=np.array([0, 0, 1]),
        paths=[
            [
                (-1000, 800, 60),
                (-820, 850, 60),
                (-700, 820, 60),
                (-550, 700, 60),
                (-400, 400, 60),
                (-300, 270, 60),
                (-100, 75, 60),
                (50, 50, 80),
                (150, -200, 80),
                (350, -250, 80),
                (500, -30, 80),
                (650, 200, 130),
                (800, 210, 130),
                (900, 150, 130),
                (1000, 300, 130),
            ]
        ],
        path_speeds=[10],
        open_output_dir=True,
    )
    print(f"Radial (small) site total irradiance: {result.total_irrad:,.0f}")


def assess_nsttf():
    preset_result = analysis(
        site=CspSite.NSTTF,
        threshold=2,
        create_gifs=False,
        create_xls=True,
        aim_strat=AimType.Point,
        aim_params=np.array([0, 0, 90]),
        # aim_strat=AimType.SplitRing,
        # aim_params=np.array([20, 80, 0]),
        paths=[
            [
                (-150, 100, 60),
                (-105, 40, 50),
                (-100, 40, 50),
                (-60, 60, 47),
                (-5, 152, 40),
                (0, 150, 40),
                (25, 75, 55),
                (50, 50, 60),
                (53, 52, 62),
                (150, 100, 50),
            ]
        ],
        path_speeds=[10],  # m/s
        open_output_dir=True,
    )
    print(f"NSTTF total irradiance: {preset_result.total_irrad:,.0f}")


def assess_sample_v2():
    result = analysis(
        site=CspSite.SampleV2,
        year=2024,
        month=6,
        day=21,
        hour=12,
        threshold=4,
        lat=35.0,
        lng=-115.0,
        timezone=-8,
        create_gifs=False,
        create_xls=True,
        # aim_strat=AimType.Point,
        # aim_params=np.array([0, 0, 120]),
        aim_strat=AimType.Ring,
        aim_params=np.array([30, 90, 120]),
        open_output_dir=True,
    )
    print(f"Sample site total irradiance: {result.total_irrad:,.0f}")


if __name__ == "__main__":
    assess_radial()
    # assess_radial_small()
    # assess_nsttf()
    # assess_sample_v2()
