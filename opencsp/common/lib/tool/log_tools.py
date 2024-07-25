"""
Utilities for managing logs for multi-processing.

"""

import logging as log
import multiprocessing as mp
import os
import re
import socket
import sys
from typing import Callable

# Don't import any other opencsp libraries here. Log tools _must_ be able to be
# imported before any other opencsp code. Instead, if there are other
# dependencies, import them at the time of use.

global_singleprocessing_logger: log.Logger = None
global_multiprocessing_logger: log.Logger = None


def logger(log_dir_body_ext: str = None, level: int = log.INFO, delete_existing_log: bool = True) -> log.Logger:
    """Initialize logging for single-process programs.

    Creates a fresh log file, deleting the existing log file if it exists as indicated by delete_existing_log_file.
    Once this method is called, then the debug(), info(), warn(), error(), and critical() methods will use the logger created here.

    Example usage::

        lt.logger(home_dir() + 'current.log')
        lt.info('Starting program ' + __file__)

    Args:
        - log_dir_body_ext (str): Fully qualified path to desired log file. Default [OpenCSP/dir]/../[timedate]_log.txt
        - level (int,optional): Significance threshold for writing messages to this log.
            Level values::

                10 debug    Detailed information, typically of interest only when diagnosing problem1s.
                            Can lead to lots of system-generated information in the log file, such as
                            matplotlib font declarations, etc.
                20 info     Confirmation that things are working as expected.
                30 warning  An indication that something unexpected happened, or indicative of some problem in the
                            near future (e.g. 'disk space low'). The software is still working as expected.
                40 error    Due to a more serious problem, the software has not been able to perform some function.
                50 critical A serious error, indicating that the program itself may be unable to continue running.

            Messages are written to the log if their severity >= level.
            See https://docs.python.org/3/howto/logging.html
        - delete_existing_log (bool, optional): Whether to delete a log from a previous run if encountered. Default True.
            If False, new log information is appended to an existing file if it exists.
    """
    global global_singleprocessing_logger

    # import here instead of at the top of the file to avoid cyclic import issues
    import opencsp.common.lib.tool.file_tools as ft
    import opencsp.common.lib.tool.time_date_tools as tdt

    # Default log name
    if log_dir_body_ext is None:
        import opencsp.common.lib.opencsp_path.opencsp_root_path as orp

        log_body_ext = tdt.current_date_time_string_forfile() + "_log.txt"
        log_dir_body_ext = os.path.join(orp.opencsp_code_dir(), "..", log_body_ext)

    # Delete previous log.
    if delete_existing_log and ft.file_exists(log_dir_body_ext):
        ft.delete_file(log_dir_body_ext)

    # Initialize log.
    logfile_dir = ft.path_components(log_dir_body_ext)[0]
    ft.create_directories_if_necessary(logfile_dir)
    log.basicConfig(filename=log_dir_body_ext, level=level)
    global_singleprocessing_logger = log.getLogger("singleprocessing")

    # Also log to console
    _add_stream_handlers(global_singleprocessing_logger, level)

    # If we attach our own handlers, then we should set propagate to 0 or False
    # global_singleprocessing_logger.propagate = 0

    # Standard initial lines.
    if not delete_existing_log:
        info('')  # Blank line to separate from previous log, if any.
    info('Start run ' + tdt.current_date_time_string())

    return global_singleprocessing_logger


def multiprocessing_logger(log_dir_body_ext=None, level=log.INFO) -> log.Logger:
    """Create a logger for logging across many processes.

    For multiprocessing logs, it is recommended that the existing log is deleted by the user or a batch process script.
    Once this method is called, then the info(), warn(), and error() methods will use the logger created here.

    Example usage::

        lt.multiprocessing_logger(experiment_dir() + '/2021-05-13_FastScan2/4_Post/Construction/20210525/1325_NS_U/090c_PredictHeliostats/latest_run.log')
        lt.info('Starting program ' + __file__)

    Args:
        - log_dir_body_ext (str): Fully qualified path to desired log file. None for a stream handler.
        - level (int,optional): Significance threshold for writing messages to this log.
            See https://docs.python.org/3/howto/logging.html
    """
    global global_multiprocessing_logger

    # import here instead of at the top of the file to avoid cyclic import issues
    import opencsp.common.lib.tool.file_tools as ft
    import opencsp.common.lib.tool.time_date_tools as tdt

    # parse the log file
    if log_dir_body_ext is not None:
        # Extract path components.
        log_dir = ft.path_components(log_dir_body_ext)[0]
        # Create output directory if necessary.
        ft.create_directories_if_necessary(log_dir)

    # Construct logger.
    global_multiprocessing_logger = mp.get_logger()
    global_multiprocessing_logger.setLevel(level)

    # Get the host and process name
    process_name = '%(processName)s'
    hn_match = re.match(".*?([0-9]+).*", socket.gethostname())
    if hn_match:
        process_name = hn_match.groups()[0] + ":" + process_name

    # Set formatter.
    formatter = log.Formatter(f"[%(asctime)s| %(levelname)s| {process_name}] %(message)s")
    if log_dir_body_ext is not None:
        handler = log.FileHandler(log_dir_body_ext)
        handler.setFormatter(formatter)
        global_multiprocessing_logger.addHandler(handler)

    # Also log to console
    _add_stream_handlers(global_multiprocessing_logger, level, formatter)

    # # This will make sure you won't have duplicated messages in the output.
    # if not len(global_multiprocessing_logger.handlers):
    #     global_multiprocessing_logger.addHandler(handler)

    # Standard initial lines.
    info('Start run ' + tdt.current_date_time_string())

    # Return.
    return global_multiprocessing_logger


def _add_stream_handlers(logger_: log.Logger, level: int, formatter: log.Formatter = None) -> None:
    """Adds streams to the given logger. Prints

    From https://stackoverflow.com/questions/16061641/python-logging-split-between-stdout-and-stderr
    """
    # stdout: log everything between level and warning
    h1 = log.StreamHandler(sys.stdout)
    h1.setLevel(level)
    h1.addFilter(lambda record: record.levelno < log.WARNING)
    h1.setFormatter(formatter)
    logger_.addHandler(h1)

    # stderr: log everything warning and greater (warning, error, critical)
    h2 = log.StreamHandler(sys.stderr)
    h2.setLevel(log.WARNING)
    h2.setFormatter(formatter)
    logger_.addHandler(h2)


def _suppress_newlines(logger: log.Logger, **kwargs):
    suppressed_terminators: dict[log.Handler, str] = {}
    if "end" in kwargs:
        for handler in logger.handlers:
            if isinstance(handler, log.StreamHandler):
                suppressed_terminators[handler] = handler.terminator
                handler.terminator = kwargs["end"]
        del kwargs["end"]
    return suppressed_terminators, kwargs


def _reset_newlines(suppressed_terminator: dict[log.Handler, str]):
    for handler, terminator in suppressed_terminator.items():
        handler.terminator = terminator


def get_log_method_for_level(level: int) -> Callable:
    """
    Returns one of the log methods (debug, info, warning, error, critical) based on the given level.

    Parameters
    ----------
    level : int
        One of log.DEBUG, log.INFO, log.WARNING, log.ERROR, or log.CRITICAL
    """
    if level == log.DEBUG:
        return debug
    if level == log.INFO:
        return info
    if level == log.WARNING:
        return warning
    if level == log.ERROR:
        return error
    if level == log.CRITICAL:
        return critical
    error_and_raise(ValueError, f"Error in log_tools.get_log_method_for_level(): unknown log level {level}")


def _log(logger: log.Logger, log_method: Callable, *vargs, **kwargs):
    suppressed_newlines, kwargs = _suppress_newlines(logger, **kwargs)
    log_method(*vargs, **kwargs)
    _reset_newlines(suppressed_newlines)


def debug(*vargs, **kwargs) -> int:
    """Output debugging information, both to console and log file.

    This is for logging detailed information, typically of interest only when
    diagnosing problems. Can lead to lots of system-generated information in
    the log file, such as matplotlib font declarations, etc.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.debug('In my_function(), input_file  = ' + str(input_file_dir_body_ext))
        lt.debug('In my_function(), output_file = ' + str(output_file_dir_body_ext))
    """
    if global_multiprocessing_logger is not None:
        _log(global_multiprocessing_logger, global_multiprocessing_logger.debug, *vargs, **kwargs)
    else:
        if global_singleprocessing_logger is not None:
            _log(global_singleprocessing_logger, global_singleprocessing_logger.debug, *vargs, **kwargs)
        else:
            print(*vargs, **kwargs)
    return 0


def info(*vargs, **kwargs) -> int:
    """Report program progress, both to console and log file.

    Use this level to confirm that things are working as expected.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.info('In my_function(), writing file: ' + str(output_file_dir_body_ext) + '...')
    """
    if global_multiprocessing_logger is not None:
        _log(global_multiprocessing_logger, global_multiprocessing_logger.info, *vargs, **kwargs)
    else:
        if global_singleprocessing_logger is not None:
            _log(global_singleprocessing_logger, global_singleprocessing_logger.info, *vargs, **kwargs)
        else:
            print(*vargs, **kwargs)
    return 0


def warning(*vargs, **kwargs):
    """Warning message, both to console and log file.

    Use this level as an indication that something unexpected happened, or
    indicative of some problem in the near future (e.g. 'disk space low').
    The software should still be working as expected.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.warn('In my_function(), Plot x label is empty.')
    """
    if global_multiprocessing_logger is not None:
        _log(global_multiprocessing_logger, global_multiprocessing_logger.warning, *vargs, **kwargs)
    else:
        if global_singleprocessing_logger is not None:
            _log(global_singleprocessing_logger, global_singleprocessing_logger.warning, *vargs, **kwargs)
        else:
            print(*vargs, **kwargs)
    return 0


warn = warning


def error(*vargs, **kwargs) -> int:
    """Error message, both to console and log file.

    Due to a more serious problem, the software has not been able to perform some function.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.error_and_raise(ValueError, 'In my_function(), non-positive value x=' + str(x) + ' encountered.')
        # or
        lt.error('In my_function(), non-positive value x=' + str(x) + ' encountered.')
    """
    if global_multiprocessing_logger is not None:
        _log(global_multiprocessing_logger, global_multiprocessing_logger.error, *vargs, **kwargs)
    else:
        if global_singleprocessing_logger is not None:
            _log(global_singleprocessing_logger, global_singleprocessing_logger.error, *vargs, **kwargs)
        else:
            print(*vargs, **kwargs, file=sys.stderr)
    return 0


def critical(*vargs, **kwargs) -> int:
    """Critical error message, both to console and log file.

    A serious error, indicating that the program itself may be unable to continue running.
    You should most likely be using the critical_and_raise version of this function.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.critical_and_raise(ValueError, 'In my_function(), Negative x should be impossible, but x=' + str(x) + ' encountered.')
        # or
        lt.critical('In my_function(), Negative x should be impossible, but x=' + str(x) + ' encountered.')
    """
    if global_multiprocessing_logger is not None:
        _log(global_multiprocessing_logger, global_multiprocessing_logger.critical, *vargs, **kwargs)
    else:
        if global_singleprocessing_logger is not None:
            _log(global_singleprocessing_logger, global_singleprocessing_logger.critical, *vargs, **kwargs)
        else:
            print(*vargs, **kwargs, file=sys.stderr)
    return 0


def error_and_raise(exception_class: Exception.__class__, msg: str) -> None:
    """Logs the given message at the "error" level and raises the given exception, also with this message.

    Args:
        exception_class (Exception.__class__): An exception class. See below for built-in exception types.
        msg (str): The message to go along with the exception.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.error_and_raise(ValueError, 'In my_function(), non-positive value x=' + str(x) + ' encountered.')

    See https://docs.python.org/3/library/exceptions.html for a list of built-in exceptions::

        BaseException
        ├── BaseExceptionGroup
        ├── GeneratorExit
        ├── KeyboardInterrupt
        ├── SystemExit
        └── Exception
            ├── ArithmeticError
            │    ├── FloatingPointError
            │    ├── OverflowError
            │    └── ZeroDivisionError
            ├── AssertionError
            ├── AttributeError
            ├── BufferError
            ├── EOFError
            ├── ImportError
            │    └── ModuleNotFoundError
            ├── LookupError
            │    ├── IndexError
            │    └── KeyError
            ├── MemoryError
            ├── NameError
            │    └── UnboundLocalError
            ├── OSError
            │    ├── BlockingIOError
            │    ├── ChildProcessError
            │    ├── ConnectionError
            │    │    ├── BrokenPipeError
            │    │    ├── ConnectionAbortedError
            │    │    ├── ConnectionRefusedError
            │    │    └── ConnectionResetError
            │    ├── FileExistsError
            │    ├── FileNotFoundError
            │    ├── InterruptedError
            │    ├── IsADirectoryError
            │    ├── NotADirectoryError
            │    ├── PermissionError
            │    ├── ProcessLookupError
            │    └── TimeoutError
            ├── ReferenceError
            ├── RuntimeError
            │    ├── NotImplementedError
            │    └── RecursionError
            ├── StopAsyncIteration
            ├── StopIteration
            ├── SyntaxError
            │    └── IndentationError
            │         └── TabError
            ├── SystemError
            ├── TypeError
            ├── ValueError
            │    └── UnicodeError
            │         ├── UnicodeDecodeError
            │         ├── UnicodeEncodeError
            │         └── UnicodeTranslateError
            └── Warning
                ├── BytesWarning
                ├── DeprecationWarning
                ├── EncodingWarning
                ├── FutureWarning
                ├── ImportWarning
                ├── PendingDeprecationWarning
                ├── ResourceWarning
                ├── RuntimeWarning
                ├── SyntaxWarning
                ├── UnicodeWarning
                └── UserWarning
    """
    msg = str(msg)  # Ensure that message is a string, to enable concatenation.
    error(msg)
    try:
        e = exception_class(msg)
    except Exception as exc:
        raise RuntimeError(msg) from exc
    raise e


def critical_and_raise(exception_class: Exception.__class__, msg: str) -> None:
    """Logs the given message at the "critical" level and raises the given exception, also with this message.

    Args:
        exception_class (Exception.__class__): An exception class. See error_and_raise() for a description of built-in exceptions.
        msg (str): The message to go along with the exception.

    Example::

        import log_tools as lt
        lt.logger(home_dir() + 'current.log')
        lt.critical_and_raise(ValueError, 'In my_function(), Negative x should be impossible, but x=' + str(x) + ' encountered.')
    """
    msg = str(msg)  # Ensure that message is a string, to enable concatenation.
    critical(msg)
    try:
        e = exception_class(msg)
    except Exception as exc:
        raise RuntimeError(msg) from exc
    raise e


def log_and_raise_value_error(local_logger, msg) -> None:
    """Logs an error and raises a ValueError with given message

    Parameters
    ----------
    local_logger : Logger
        Unused, kept for backwards compatibility
    msg : str
        Error message
    """
    error(msg)
    raise ValueError(msg)
