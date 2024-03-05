"""
Converting units, e.g, inches to meters.



"""

MM_PER_INCH = 25.4
M_PER_INCH = MM_PER_INCH / 1000.0


# Inch <--> Meters
def inch_to_meter(value_inch: float) -> float:
    return value_inch * M_PER_INCH


def meter_to_inch(value_meter: float) -> float:
    return value_meter / M_PER_INCH


# Dots per Inch  <-->  Dots per Meters
def dpi_to_dpm(value_dpi: float) -> float:
    return value_dpi / M_PER_INCH


def dpm_to_dpi(value_dpi: float) -> float:
    return value_dpi * M_PER_INCH
