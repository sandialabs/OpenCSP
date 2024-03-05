_ss_settings_key = "sensitive_strings"
_ss_settings_default: dict[str, None] = {
    "sensitive_strings_dir": None,
    "sensitive_strings_file": None,
    "allowed_binaries_file": None,
    "cache_file": None,
}
"""
sensitive_strings_dir: Where to save log output to when checking for sensitive strings.
sensitive_strings_file: Where to find the sensitive_strings.csv files, for use with opencsp_code/contrib/scripts/sensitive_strings.
allowed_binaries_file: Where to find the sensitive_strings_allowed_binary_files.csv file, for use with opencsp_code/contrib/scripts/sensitive_strings.
cache_file: Greatly improves the speed of searching for sensitive strings by remembering which files were checked previously, for use with opencsp_code/contrib/scripts/sensitive_strings.
"""

_settings_list = [
    [_ss_settings_key, _ss_settings_default]
]
