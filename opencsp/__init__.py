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

__all__ = ['opencsp_settings']
