"""
Paths to OpenCSP root directories.
"""

import os
import inspect
import opencsp

from opencsp.common.lib.opencsp_path import opencsp_settings
import opencsp.common.lib.tool.log_tools as lt

# OpenCSP root directories.


def opencsp_code_dir():
    """The directory containing the opencsp code.

    For development with the full git project, this is the directory
    "[...]/[cloned git folder]/opencsp". This means it will contain the "app"
    and "common" folders."""
    return os.path.dirname(inspect.getfile(opencsp))


def opencsp_doc_dir():
    """The directory containing the opencsp documentation."""
    return os.path.join(
        opencsp_code_dir(), '..', 'doc'
    )  # TODO BGB: will this always live next to the 'opencsp' directory?


def opencsp_data_example_dir():
    """The directory containing the opencsp example data, for examples that have very large data inputs."""
    example_data_dir: str = opencsp_settings["opencsp_root_path"]["example_data_dir"]
    if example_data_dir != None:
        return example_data_dir
    return os.path.join(opencsp_code_dir(), '..', 'opencsp_data_example')


def opencsp_data_test_dir():
    """This method deprecated. For most tests you can find the data in the neighboring \"data\" directory, inside the \"test\" directory."""
    lt.warn(
        "Deprecation warning (opencsp_data_test_dir): " + opencsp_data_test_dir.__doc__
    )


def opencsp_scratch_dir(project_dir=None) -> str:
    """The scratch directory to read/write from.

    This directory holds a large amount of data, most of which should be treated
    as temporary. For running parallel execution programs on a cluster, it is
    recommended that this directory be used for all input and output, as it is
    designed for parallel access. When running on a cluster, this directory is a
    shared directory between multiple computers (aka network file system)."""
    scratch_dir: str = opencsp_settings["opencsp_root_path"]["scratch_dir"]
    if scratch_dir != None and os.path.exists(scratch_dir):
        actual_scratch_dir = os.path.join(
            scratch_dir, opencsp_settings["opencsp_root_path"]["scratch_name"]
        )
        return (
            actual_scratch_dir
            if project_dir == None
            else os.path.join(actual_scratch_dir, project_dir)
        )

    if os.name == "nt":
        # Check for a scratch mirror directory on the user's computer.
        actual_scratch_dir = os.path.join(
            opencsp_code_dir(),
            '..',
            opencsp_settings["opencsp_root_path"]["scratch_name"],
        )
        actual_scratch_dir = (
            actual_scratch_dir
            if project_dir == None
            else os.path.join(actual_scratch_dir, project_dir)
        )
        if os.path.isdir(actual_scratch_dir):
            return actual_scratch_dir

        # This is a directory on windows that we should be able to write to
        actual_scratch_dir = os.path.join(
            os.path.expandvars("%LOCALAPPDATA%"),
            "opencsp",
            opencsp_settings["opencsp_root_path"]["scratch_name"],
        )
        return (
            actual_scratch_dir
            if project_dir == None
            else os.path.join(actual_scratch_dir, project_dir)
        )
    else:
        # On the cluster nodes, we should be writing to the scratch file system for multi-node programs.
        # Aka don't do this:
        # return os.path.join(os.path.expanduser('~'), ".opencsp/cache")
        actual_scratch_dir = (
            f"/{opencsp_settings['opencsp_root_path']['scratch_name']}/"
        )
        return (
            actual_scratch_dir
            if project_dir == None
            else os.path.join(actual_scratch_dir, project_dir)
        )


def opencsp_cache_dir():
    """The directory to save cache files to."""
    return os.path.join(opencsp_scratch_dir(), "cache")


def opencsp_temporary_dir():
    """A directory that should be safe to save temporary files to."""
    return os.path.join(opencsp_scratch_dir(), "tmp")


def _opencsp_settings_dirs():
    """Returns a list of possible locations for settings files,
    from highest to lowest importance (higher importance overrides lower importance).

    This function is marked as protected because it is depended on by __init__.py, and
    so this function can't be dependent on anything in __init__.py.

    This function looks for the environmental variable "OPENCSP_SETTINGS_DIRS"
    and, if "None", returns an empty list (for running unit tests). For any
    other value of the env var, directories should be delimited with semicolons
    and are appended to the end of the returned list.
    """
    ret: list[str] = []

    if os.name == "nt":
        ret.append(os.path.join(os.path.expanduser("~"), ".opencsp", "settings"))
        # TODO add more directories?
        # ret.append(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "opencsp", "settings"))
        # ret.append(os.path.join(os.path.expandvars("%APPDATA%"), "opencsp", "settings"))
    else:
        ret.append(
            os.path.join(os.path.expanduser("~"), ".config", "opencsp", "settings")
        )

    if "OPENCSP_SETTINGS_DIRS" in os.environ:
        if os.environ["OPENCSP_SETTINGS_DIRS"] == "None":
            return []
        else:
            additional_dirs = os.environ["OPENCSP_SETTINGS_DIRS"].split(";")
            for i, dir in enumerate(additional_dirs):
                additional_dirs[i] = dir.replace("~", os.path.expanduser("~"))
            ret += additional_dirs

    return ret


def relative_opencsp_data_test_dir():
    """This method deprecated. For most tests you can find the data in the neighboring \"data\" directory, inside the \"test\" directory."""
    lt.warn(
        "Deprecation warning (relative_opencsp_data_test_dir): "
        + relative_opencsp_data_test_dir.__doc__
    )
