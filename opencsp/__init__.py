"""
OpenCSP: 
========
Open source python libraies for concentrating solar power reserach and development

Contents
--------
app: Contains applications that that use the OpenCSP base classes.
common: Contains base classes that are used in multiple csp applications.

Tests
-----
OpenCSP uses the pytest library.
"""

import configparser
import os
import copy
import sys
import argparse


def _opencsp_settings_dirs() -> list[str]:
    """Returns a list of possible locations for settings files,
    from lowest to highest importance (higher importance overrides lower importance).

    This function looks for the environmental variable "OPENCSP_SETTINGS_DIRS"
    and, if "None", returns an empty list (for running unit tests). For any
    other value of the env var, directories should be delimited with semicolons
    and are appended to the end of the returned list.
    """
    ret: list[str] = []

    # home directories
    if os.name == "nt":
        ret.append(os.path.join(os.path.expandvars("%USERPROFILE%"), ".opencsp", "settings"))
        # TODO add more directories?
        # ret.append(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "opencsp", "settings"))
        # ret.append(os.path.join(os.path.expandvars("%APPDATA%"), "opencsp", "settings"))
    else:
        ret.append(os.path.join(os.path.expanduser("~"), ".config", "opencsp", "settings"))

    # environmental directories
    if "OPENCSP_SETTINGS_DIRS" in os.environ:
        if os.environ["OPENCSP_SETTINGS_DIRS"] == "None":
            return []
        else:
            additional_dirs = os.environ["OPENCSP_SETTINGS_DIRS"].split(";")
            for i, dir in enumerate(additional_dirs):
                additional_dirs[i] = dir.replace("~", os.path.expanduser("~"))
            ret += additional_dirs

    return ret


def apply_command_line_arguments(settings_from_ini: configparser.ConfigParser) -> configparser.ConfigParser:
    settings_mixed = copy.copy(settings_from_ini)

    # parse the command line
    parser = argparse.ArgumentParser(
        prog="OpenCSP/__init__.py", description="OpenCSP settings parser", add_help=False, exit_on_error=False
    )
    parser.add_argument(
        "--dir-input",
        dest="dir_input",
        default="",
        type=str,
        help="Use the given directory value as the input directory instead of [opencsp_root_path]/[large_data_example_dir].",
    )
    parser.add_argument(
        "--dir-output",
        dest="dir_output",
        default="",
        type=str,
        help="Use the given directory value as the output directory instead of [opencsp_root_path]/[scratch_dir]/[scratch_name].",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])
    dir_input: str = args.dir_input
    dir_output: str = args.dir_output
    sys.argv = [sys.argv[0]] + remaining
    overridden_values: list[tuple[str, str]] = []

    # apply the command line arguments to the settings
    if dir_input != "":
        settings_mixed["opencsp_root_path"]["large_data_example_dir"] = dir_input
        overridden_values.append(("opencsp_root_path/large_data_example_dir", dir_input))
    if dir_output != "":
        dir_output_path, dir_output_name = os.path.dirname(dir_output), os.path.basename(dir_output)
        try:
            os.makedirs(dir_output)
        except FileExistsError:
            pass
        settings_mixed["opencsp_root_path"]["scratch_dir"] = dir_output_path
        settings_mixed["opencsp_root_path"]["scratch_name"] = dir_output_name
        overridden_values.append(("opencsp_root_path/scratch_dir", dir_output_path))
        overridden_values.append(("opencsp_root_path/scratch_name", dir_output_name))

    # let the user know if values have been overridden
    if len(overridden_values) > 0:
        print("Some settings have been overridden from the command line:")
        for setting_name, command_line_value in overridden_values:
            print(f"\t{setting_name}: {command_line_value}")

    return settings_mixed


_settings_files: list[str] = []

# default settings file
_default_dir = os.path.dirname(__file__)
_settings_files.append(os.path.join(_default_dir, "default_settings.ini"))

# locate other settings files
for _dirname in _opencsp_settings_dirs():
    _settings_file = os.path.join(_dirname, "opencsp_settings.ini")
    if os.path.exists(_settings_file):
        _settings_files.append(_settings_file)

# load the settings
opencsp_settings = configparser.ConfigParser(allow_no_value=True)
opencsp_settings.read(_settings_files)

for section in opencsp_settings.sections():
    for key in opencsp_settings[section]:
        print(f"opencsp_settings[{section}][{key}]={opencsp_settings[section][key]}")

opencsp_settings = apply_command_line_arguments(opencsp_settings)
__all__ = ["opencsp_settings"]
