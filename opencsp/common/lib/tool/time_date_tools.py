"""

Convenience functions for manipulating time and dates.



"""

import dateutil.tz as dtz
from datetime import date, time, datetime, timezone, timedelta
import pytz
import time

import opencsp.common.lib.tool.log_tools as lt

tdt_ignore_legacy = False


def print_current_date_time() -> None:
    """
    Print the current date and time to the log.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    lt.info("Current Date and Time =" + current_date_time_string())


def print_current_date() -> None:
    """
    Print the current date to the log.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    lt.info("Current Date =" + current_date_string())


def print_current_time() -> None:
    """
    Print the current time to the log.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    lt.info("Current Time =" + current_time_string())


def current_date_time_string() -> str:
    """%Y-%m-%d %H:%M:%S"""
    now = datetime.now()
    return current_date_string(now.date()) + ' ' + current_time_string(now.time())


def current_date_string(now: date = None) -> str:
    """%Y-%m-%d"""
    if now is None:
        now = datetime.now().date()
    current_date_str = now.strftime('%Y-%m-%d')
    return current_date_str


def current_time_string(now: time = None) -> str:
    """%H:%M:%S"""
    if now is None:
        now = datetime.now().time()
    current_time_str = now.strftime('%H:%M:%S')
    return current_time_str


def date_string_forfile(d: date) -> str:
    """%Y%m%d"""
    date_str = d.strftime('%Y%m%d')
    return date_str


def current_date_string_forfile() -> str:
    """
    Get the current date as a formatted string suitable for filenames.

    The format of the string is "%Y%m%d".

    Returns
    -------
    str
        The current date as a string suitable for use in filenames.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    now = datetime.now()
    current_date_str = date_string_forfile(now.date())
    return current_date_str


def time_string_forfile(t: time) -> str:
    """%H%M%S"""
    time_str = t.strftime('%H%M%S')
    return time_str


def current_time_string_forfile() -> str:
    """
    Get the current time as a formatted string suitable for filenames.

    The format of the string is "%H%M%S".

    Returns
    -------
    str
        The current time as a string suitable for use in filenames.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    now = datetime.now()
    current_time_str = time_string_forfile(now.time())
    return current_time_str


def date_time_string_forfile(dt: datetime) -> str:
    """%Y%m%d_%H%M%S"""
    return date_string_forfile(dt.date()) + '_' + time_string_forfile(dt.time())


def current_date_time_string_forfile() -> str:
    """%Y%m%d_%H%M%S"""
    now = datetime.now()
    return date_time_string_forfile(now)


def current_time() -> float:
    """
    Get the current time in seconds since the epoch.

    Returns
    -------
    float
        The current time in seconds since the epoch (Unix time).
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return time.time()


def elapsed_time_since_start_sec(start_time: float) -> float:
    """
    Calculate the elapsed time in seconds since a given start time.

    Parameters
    ----------
    start_time : float
        The start time in seconds since the epoch.

    Returns
    -------
    float
        The elapsed time in seconds since the start time.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return time.time() - start_time


def to_datetime(ymdhmsz: list[int, int, int, int, int, float, int] | datetime):
    """Converts from a list of [year,month,day,hour,minute,second,utcoffset] to a datetime instance"""
    if isinstance(ymdhmsz, datetime):
        return ymdhmsz
    if len(ymdhmsz) != 7:
        raise RuntimeError(f"Unexpected datetime representation for the input list {ymdhmsz}!")
    tzinfo = timezone(timedelta(hours=ymdhmsz[6]))
    return datetime(ymdhmsz[0], ymdhmsz[1], ymdhmsz[2], ymdhmsz[3], ymdhmsz[4], ymdhmsz[5], tzinfo=tzinfo)


def from_datetime(dt: datetime) -> list[int, int, int, int, int, float, int]:
    """Converts from a datetime instance to a list of [year,month,day,hour,minute,second,utcoffset]"""
    tzdelta: timedelta = dt.utcoffset()
    utcoffset = int(tzdelta.total_seconds() / 3600)
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, utcoffset]


def tz(name_or_offset: str | float | timedelta):
    """Create a timezone based off of the name ("US/Mountain"), the
    short name ("MDT" or "MST"), the UTC offset (-6), or the UTC
    timedelta offset (datetime.timedelta(hours=-6))."""
    if isinstance(name_or_offset, timedelta):
        return timezone(name_or_offset)
    if isinstance(name_or_offset, int) | isinstance(name_or_offset, float):
        return timezone(timedelta(hours=name_or_offset))

    name: str = name_or_offset
    name_l = name_or_offset.lower()

    # try to match by whole name
    for name in pytz.common_timezones:
        if name.lower() == name_l:
            return dtz.gettz(name)

    # try to match by partial name
    partial_matches: list[str] = []
    for name in pytz.common_timezones:
        if name_l in name.lower():
            partial_matches.append(name)
    if len(partial_matches) == 1:
        return dtz.gettz(partial_matches[0])
    elif len(partial_matches > 1):
        lt.error_and_raise(
            RuntimeError,
            f"Error: in time_date_tools.tz(), ambiguous timezone name {name} has multiple possible matches: {partial_matches}",
        )

    # try to match by shorthand names
    now = datetime.now()
    for name in pytz.common_timezones:
        tz = dtz.gettz(name)
        if name_l == tz.tzname(now).lower():
            return tz

    # failed to match
    lt.error_and_raise(RuntimeError, f"Error: in time_date_tools.tz(), failed to find a timezone with the name {name}")


def add_seconds_to_ymdhmsz(
    ymdhmsz: list[int, int, int, int, int, float, int], time_sec: float, ignore_legacy=None
) -> list[int, int, int, int, int, float, int]:
    """
    Add a specified number of seconds to a date and time represented as a list.

    The input list is expected to be in the format:
    [year, month, day, hour, minute, second, zone].

    Parameters
    ----------
    ymdhmsz : list[int, int, int, int, int, float, int]
        A list representing the date and time, where:
        - year (int): The year.
        - month (int): The month (1-12).
        - day (int): The day of the month (1-31).
        - hour (int): The hour (0-23).
        - minute (int): The minute (0-59).
        - second (float): The second (0-59.999...).
        - zone (int): The time zone offset.
    time_sec : float
        The number of seconds to add to the date and time.
    ignore_legacy : bool, optional
        If set to False, a warning message will be printed indicating that this function is a legacy function.

    Returns
    -------
    list[int, int, int, int, int, float, int]
        A new list representing the updated date and time after adding the specified seconds.

    Raises
    ------
    AssertionError
        If rolling over a day boundary is not implemented.

    Notes
    -----
    This function currently does not handle rolling over a day boundary or month boundary.

    Examples
    --------
    >>> add_seconds_to_ymdhmsz([2023, 3, 15, 12, 30, 45.0, 0], 30)
    [2023, 3, 15, 12, 31, 15.0, 0]
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    ignore_legacy = tdt_ignore_legacy if (ignore_legacy == None) else ignore_legacy
    if ignore_legacy == False:
        print(
            'subtract_seconds_from_ymdhmsz is a legacy function. Consider using "to_datetime() + datetime.timedelta(0,s)" instead'
        )
    # Parse the ymdhmsz object.
    year = ymdhmsz[0]
    month = ymdhmsz[1]
    day = ymdhmsz[2]
    hour = ymdhmsz[3]
    minute = ymdhmsz[4]
    second = ymdhmsz[5]
    zone = ymdhmsz[6]
    # Add the given seconds, rolling over as necessary.
    if (second + time_sec) <= 60:
        second += time_sec
    else:
        if minute < 59:
            minute += 1
            second -= 60
            second += time_sec
        else:
            if hour < 23:
                hour += 1
                minute -= 60
                minute += 1
                second -= 60
                second += time_sec
            else:
                print("ERROR: In add_seconds_to_ymdhms(), rolling over a day boundary not implemented yet.")
                assert False
    # Return.
    return [year, month, day, hour, minute, second, zone]


def subtract_seconds_from_ymdhmsz(
    ymdhmsz: list[int, int, int, int, int, float, int], time_sec: float, ignore_legacy=None
) -> list[int, int, int, int, int, float, int]:
    """
    Subtract a specified number of seconds from a date and time represented as a list.

    The input list is expected to be in the format:
    [year, month, day, hour, minute, second, zone].

    Parameters
    ----------
    ymdhmsz : list[int, int, int, int, int, float, int]
        A list representing the date and time, where:
        - year (int): The year.
        - month (int): The month (1-12).
        - day (int): The day of the month (1-31).
        - hour (int): The hour (0-23).
        - minute (int): The minute (0-59).
        - second (float): The second (0-59.999...).
        - zone (int): The time zone offset.
    time_sec : float
        The number of seconds to subtract from the date and time.
    ignore_legacy : bool, optional
        If set to False, a warning message will be printed indicating that this function is a legacy function.

    Returns
    -------
    list[int, int, int, int, int, float, int]
        A new list representing the updated date and time after subtracting the specified seconds.

    Raises
    ------
    AssertionError
        If rolling over a month boundary is not implemented.

    Notes
    -----
    This function currently does not handle rolling over a month boundary.

    Examples
    --------
    >>> subtract_seconds_from_ymdhmsz([2023, 3, 15, 12, 30, 45.0, 0], 30)
    [2023, 3, 15, 12, 30, 15.0, 0]
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    ignore_legacy = tdt_ignore_legacy if (ignore_legacy == None) else ignore_legacy
    if ignore_legacy == False:
        print(
            'subtract_seconds_from_ymdhmsz is a legacy function. Consider using "to_datetime() - datetime.timedelta(0,s)" instead'
        )
    # Parse the ymdhmsz object.
    year = ymdhmsz[0]
    month = ymdhmsz[1]
    day = ymdhmsz[2]
    hour = ymdhmsz[3]
    minute = ymdhmsz[4]
    second = ymdhmsz[5]
    zone = ymdhmsz[6]
    # Subtract the given seconds, rolling over as necessary.
    if time_sec <= second:
        second -= time_sec
    else:
        if minute >= 1:
            minute -= 1
            second += 60
            second -= time_sec
        else:
            if hour >= 1:
                hour -= 1
                minute += 60
                minute -= 1
                second += 60
                second -= time_sec
            else:
                if day >= 1:
                    day -= 1
                    hour += 24
                    hour -= 1
                    minute += 60
                    minute -= 1
                    second += 60
                    second -= time_sec
                else:
                    print(
                        "ERROR: In subtract_seconds_from_ymdhms(), rolling over a month boundary not implemented yet."
                    )
                    assert False
    # Return.
    return [year, month, day, hour, minute, second, zone]
