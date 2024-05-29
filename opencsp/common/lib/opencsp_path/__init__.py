import copy
import importlib
import importlib.util
import inspect
import os
import sys

import opencsp.common.lib.tool.log_tools as lt

_orp_settings_key = "opencsp_root_path"
_orp_settings_default = {
    "example_data_dir": None,
    "scratch_dir": None,
    "scratch_name": "scratch",
    "collaborative_dir": None,
}
""" example_data_dir: The directory containing the opencsp example data, for examples that have very large data inputs.
scratch_dir: The directory containing the scratch folder, for use with HPC clusters.
scratch_name: The name of the scratch directory. Default to "scratch".
collaborative_dir: A shared directory where experimental data is collected
"""

_settings_list = [[_orp_settings_key, _orp_settings_default]]
_opencsp_settings_packages = ["common.lib.tool"]
_opencsp_code_settings_packages = ["contrib.scripts"]

opencsp_settings = {}


def __load_settings_files():
    """Get the settings for each of the 'settings.json' files found in the
    _opencsp_settings_dirs(). The settings files should contain a dictionary in
    JSON form, with group name keys to group dictionaries. For example::

        {
            'opencsp_root_path': {
                'scratch_name': 'actual_scratch'
            }
        }

    Returns:
    --------
    settings_per_file: dict[str,dict[str,any]]
        The settings names and values, with one dict of values per
        'settings.json' file. The first key is the file name. The second key is
        settings group's name ('opencsp_root_path' for this file's settings).
        The third key, then, is the setting's name.
    """
    import os
    from opencsp.common.lib.opencsp_path.opencsp_root_path import _opencsp_settings_dirs
    import opencsp.common.lib.tool.file_tools as ft

    ret: dict[str, dict[str, dict[str, any]]] = {}

    # read settings for each file
    for dir in _opencsp_settings_dirs():
        settings_file_name_path_ext = os.path.join(dir, 'settings.json')

        # would use file_tools.directory_exists() except that I don't want to depend on any other part of opencsp
        if os.path.exists(settings_file_name_path_ext) and os.path.isfile(settings_file_name_path_ext):
            settings_path, settings_name, settings_ext = ft.path_components(settings_file_name_path_ext)
            settings = ft.read_json("global settings", settings_path, settings_name + settings_ext)

            # verify the types for the loaded settings
            err_msg_preamble = (
                f"Error in opencsp_path.__init__(): while parsing settings file {settings_file_name_path_ext}, "
            )
            found_err = False
            if not isinstance(settings, dict):
                lt.error(
                    err_msg_preamble
                    + f"the settings should be a dict but is instead of type {type(group_name)}. Ignoring file."
                )
                lt.info(err_msg_preamble + f"'{settings=}'")
                found_err = True

            for group_name in settings:
                if not isinstance(group_name, str):
                    lt.error(
                        err_msg_preamble
                        + f"the settings group name should be a string but is instead of type {type(group_name)}. Ignoring file."
                    )
                    lt.info(err_msg_preamble + f"'{group_name=}'")
                    found_err = True
                    break
                if not isinstance(settings[group_name], dict):
                    lt.error(
                        err_msg_preamble
                        + f"the settings group should be a dict but is instead of type {type(group_name)}. Ignoring file."
                    )
                    lt.info(err_msg_preamble + f"'{settings[group_name]=}'")
                    found_err = True
                    break

                for setting_name in settings[group_name]:
                    if not isinstance(setting_name, str):
                        lt.error(
                            err_msg_preamble
                            + f"the settings name should be a string but is instead of type {type(group_name)}. Ignoring file."
                        )
                        lt.info(err_msg_preamble + f"'{setting_name=}'")
                        found_err = True
                        break
                if found_err:
                    break

            if found_err:
                continue

            # type checks passed, append to the returned values
            ret[settings_file_name_path_ext] = settings

    return ret


def __populate_settings_list() -> list[tuple[str, dict[str, any]]]:
    import opencsp

    ret = copy.copy(_settings_list)

    # populate the package settings, which are contained within the "opencsp" package
    for package_name in _opencsp_settings_packages:
        package = importlib.import_module("opencsp." + package_name)
        ret += package._settings_list

    # populate settings from the larger opencsp_code repository
    opencsp_path = os.path.dirname(inspect.getfile(opencsp))
    for package_name in _opencsp_code_settings_packages:
        module_name = "opencsp_code." + package_name
        package_dir = os.path.abspath(os.path.join(opencsp_path, "..", package_name.replace(".", "/")))
        if os.path.exists(package_dir):
            spec = importlib.util.spec_from_file_location(module_name, package_dir + "/__init__.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            ret += module._settings_list

    return ret


def __populate_settings():
    settings_list = __populate_settings_list()

    # defaults
    opencsp_settings.clear()
    for settings_key, settings_defaults in settings_list:
        if settings_key not in opencsp_settings:
            opencsp_settings[settings_key] = {}
        for k in settings_defaults:
            opencsp_settings[settings_key][k] = settings_defaults[k]

    # override default settings with settings_file-specific settings
    settings_per_file = __load_settings_files()
    for settings_file_name_path_ext in settings_per_file:
        settings = settings_per_file[settings_file_name_path_ext]
        for settings_key, settings_defaults in settings_list:
            if settings_key not in settings:
                continue
            for k in settings[settings_key]:
                v = settings[settings_key][k]
                if v != None:
                    opencsp_settings[settings_key][k] = v


__populate_settings()
