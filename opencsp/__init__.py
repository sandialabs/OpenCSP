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

import argparse
import configparser
import copy
import importlib
import os
import platform
import shutil
import sys

_ENV_VAR_SETTINGS_DIRS = "OPENCSP_SETTINGS_DIRS"
_ENV_VAR_EDIT_SETTINGS = "OPENCSP_SETTINGS_EDIT"
_ENV_VAR_DEBUG_SETTINGS = "OPENCSP_SETTINGS_DEBUG"
OPENCSP_SETTINGS_FILE_NAME_EXT = "opencsp_settings.ini"


class LazyLoader:
    """
    The class provides a LazyLoader for delayed importing of modules.
    It allows for lazy loading of modules until they are actually needed,
    which can help improve performance by reducing load times in most cases.
    """

    # written by chatgpt
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self.module = None
        self.class_instance = None

    def _load(self):
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        if self.class_instance is None:
            self.class_instance = getattr(self.module, self.class_name)
        return self.class_instance

    def __call__(self, *args, **kwargs):
        # Return an instance of the class when called
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate attribute access to the class instance
        return getattr(self._load(), name)

    def __instancecheck__(self, instance):
        # Delegate instance check to the class instance
        return isinstance(instance, self._load())

    def __subclasscheck__(self, instance):
        # Delegate subclass check to the class instance
        return issubclass(instance, self._load())


if platform.system() == 'Darwin':
    # On Mac, force matplotlib to use the TkAgg.
    # Maybe we consider doing this for all systems?
    import matplotlib

    matplotlib.use('TkAgg')


def _opencsp_system_dir() -> str:
    """Returns the default directory for the OpenCSP settings based on the operating system."""
    if os.name == "nt":
        return os.path.abspath(os.path.join(os.path.expandvars("%USERPROFILE%"), ".opencsp", "settings"))
    else:
        return os.path.abspath(os.path.join(os.path.expanduser("~"), ".config", "opencsp", "settings"))


def _open_ini_file_in_default_editor(file_path) -> bool:
    """
    Opens the specified file in its default system editor.
    """
    # generated with ChatGPT
    try:
        if os.name == "nt":  # Windows
            os.startfile(file_path)
        elif platform.system() == 'Darwin':  # MacOS
            os.system(f'open {file_path}')
        else:  # Linux
            # For INI files, use a text editor
            os.system(f'xdg-open {file_path} || xterm -e nano {file_path}')
        return True
    except Exception:
        return False


def _edit_opencsp_settings(default_settings_file: str):
    """Copies the default OpenCSP settings to the default system directory and
    opens the settings in VS Code."""
    system_dir = _opencsp_system_dir()

    # Create the directory if it doesn't exist.
    # Note that we can't use file_tools here because OpenCSP hasn't been initialized yet.
    if not os.path.exists(system_dir):
        print(f"Creating default system directory \"{system_dir}\".")
        try:
            os.makedirs(system_dir)
        except Exception as ex:
            print(repr(ex), file=sys.stderr)
            print(
                f"Failed to create default system directory \"{system_dir}\". "
                + "Please create this directory and try again.",
                file=sys.stderr,
            )
            return False

    # Copy the file if it doesn't exist
    system_settings_file = os.path.abspath(os.path.join(system_dir, OPENCSP_SETTINGS_FILE_NAME_EXT))
    if not os.path.exists(system_settings_file):
        print(f"Copying default settings from \"{default_settings_file}\" to" + f"\"{system_settings_file}\".")
        try:
            shutil.copyfile(default_settings_file, system_settings_file)
        except Exception as ex:
            print(repr(ex), file=sys.stderr)
            print(
                f"Failed to copy default settings file from "
                + f"\"{default_settings_file}\" to \"{system_settings_file}\". "
                + "Please copy this file and try again.",
                file=sys.stderr,
            )
            return False

    # Open the file for editing
    print(f"Opening settings file \"{system_settings_file}\" in default editor.")
    if not _open_ini_file_in_default_editor(system_settings_file):
        print(
            f"Failed to open file \"{system_settings_file}\" in the default editor. "
            + "Please edit this file and try again.",
            file=sys.stderr,
        )
        return False

    return True


def _opencsp_settings_dirs() -> list[str]:
    """Returns a list of possible locations for settings files,
    from lowest to highest importance (higher importance overrides lower importance).

    This function looks for the environmental variable _ENV_VAR_SETTINGS_DIRS
    and, if "None", returns an empty list (for running unit tests). For any
    other value of the env var, directories should be delimited with semicolons
    and are appended to the end of the returned list.
    """
    ret: list[str] = []

    # home directories
    ret.append(_opencsp_system_dir())
    if os.name == "nt":
        pass
        # TODO add more directories?
        # ret.append(os.path.join(os.path.expandvars("%LOCALAPPDATA%"), "opencsp", "settings"))
        # ret.append(os.path.join(os.path.expandvars("%APPDATA%"), "opencsp", "settings"))

    # environmental directories
    if _ENV_VAR_SETTINGS_DIRS in os.environ:
        if os.environ[_ENV_VAR_SETTINGS_DIRS] == "None":
            return []
        else:
            additional_dirs = os.environ[_ENV_VAR_SETTINGS_DIRS].split(";")
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
_default_settings_file = os.path.join(_default_dir, "default_settings.ini")
_settings_files.append(_default_settings_file)

# create an initial copy of the settings file
if os.environ.get(_ENV_VAR_EDIT_SETTINGS, "0").lower() in ["1", "true"]:
    print(
        "Opening the opencsp settings file. "
        + f"Unset the environment variable \"{_ENV_VAR_EDIT_SETTINGS}\" or "
        + "set it equal to \"0\" to skip this step."
    )
    if not _edit_opencsp_settings(_default_settings_file):
        sys.exit(1)

# locate other settings files
for _dirname in _opencsp_settings_dirs():
    _settings_file = os.path.join(_dirname, OPENCSP_SETTINGS_FILE_NAME_EXT)
    if os.path.exists(_settings_file):
        _settings_files.append(_settings_file)

# load the settings
if _ENV_VAR_DEBUG_SETTINGS in os.environ:
    print("Loading OpenCSP settings from files:")
    for _settings_file in _settings_files:
        if os.path.exists(_settings_file):
            print(f"\t{_settings_file}")
opencsp_settings = configparser.ConfigParser(allow_no_value=True)
opencsp_settings.read(_settings_files)

# debugging information
if _ENV_VAR_DEBUG_SETTINGS in os.environ:
    for section in opencsp_settings.sections():
        for key in opencsp_settings[section]:
            print(f"opencsp_settings[{section}][{key}]={opencsp_settings[section][key]}")
if _ENV_VAR_DEBUG_SETTINGS in os.environ:
    print(f"Set the environment variable \"{_ENV_VAR_EDIT_SETTINGS}='1'\" to edit the settings.")

opencsp_settings = apply_command_line_arguments(opencsp_settings)
__all__ = ["opencsp_settings"]
