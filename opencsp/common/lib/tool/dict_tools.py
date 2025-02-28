import opencsp.common.lib.tool.file_tools as ft


# ACCESS


def number_of_keys(input_dict):
    """
    Return the number of keys in the given dictionary.

    Parameters
    ----------
    input_dict : dict
        The dictionary for which to count the keys.

    Returns
    -------
    int
        The number of keys in the input dictionary.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    return len(input_dict.keys())


def sorted_keys(input_dict):
    """
    Return the keys of the dictionary in a sorted list.

    Parameters
    ----------
    input_dict : dict
        The dictionary whose keys are to be sorted.

    Returns
    -------
    list
        A sorted list of the keys in the input dictionary.

    Notes
    -----
    This function constructs a list of keys and sorts it, which may be slower than using `keys()`,
    but provides the keys in a predictable order.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    key_list = list(input_dict.keys())
    key_list.sort()
    return key_list


def list_of_values_in_sorted_key_order(input_dict):
    """
    Return a list of values from the dictionary in the order of sorted keys.

    Parameters
    ----------
    input_dict : dict
        The dictionary from which to extract values.

    Returns
    -------
    list
        A list of values corresponding to the sorted keys of the input dictionary.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    output_list = []
    for key in sorted_keys(input_dict):
        output_list.append(input_dict[key])
    return output_list


# RENDER


def print_dict(
    input_dict,  # Dictionary to print.
    max_keys=10,  # Maximum number of keys to print.  Elipsis after that.
    max_value_length=70,  # Maximum value length to print.  Elipsis after that.
    indent=None,
):  # Number of blanks to print at the beginning of each line.
    """
    Print a simple dictionary, limiting the output length both laterally and vertically.

    Parameters
    ----------
    input_dict : dict
        The dictionary to print.
    max_keys : int, optional
        The maximum number of keys to print. An ellipsis will be shown if there are more keys.
    max_value_length : int, optional
        The maximum length of values to print. An ellipsis will be shown if values exceed this length.
    indent : int, optional
        The number of spaces to print at the beginning of each line for indentation.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Sort key list for consistent output order.
    key_list = sorted_keys(input_dict)
    # Content.
    for key in key_list[0:max_keys]:
        # Fetch value.
        value = input_dict[key]
        # Convert to string, and limit its length.
        value_str = str(value)
        trimmed_value_str = value_str[0:max_value_length]
        if len(value_str) > len(trimmed_value_str):
            trimmed_value_str += "..."
        # Print key : value.
        if indent == None:
            indent_str = ""
        else:
            indent_str = " " * indent
        print(indent_str + str(key) + " : " + trimmed_value_str)
    # Postamble.
    if max_keys < len(key_list):
        print(indent_str + "...")


def print_dict_of_dicts(
    input_dict,  # Dictionary to print.
    max_keys_1=5,  # Maximum number of level 1 keys to print.  Elipsis after that.
    max_keys_2=8,  # Maximum number of level 2 keys to print.  Elipsis after that.
    max_value_2_length=70,  # Maximum level 2 value length to print.  Elipsis after that.
    indent_1=0,  # Number of blanks to print at the beginning of each top-level line.
    indent_2=4,
):  # Number of additional blanks to print for each second-level line.
    """
    Print a one-level nested dictionary, limiting the output length both laterally and vertically.

    Parameters
    ----------
    input_dict : dict
        The dictionary to print, where values are themselves dictionaries.
    max_keys_1 : int, optional
        The maximum number of top-level keys to print. An ellipsis will be shown if there are more keys.
    max_keys_2 : int, optional
        The maximum number of second-level keys to print. An ellipsis will be shown if there are more keys.
    max_value_2_length : int, optional
        The maximum length of second-level values to print. An ellipsis will be shown if values exceed this length.
    indent_1 : int, optional
        The number of spaces to print at the beginning of each top-level line for indentation.
    indent_2 : int, optional
        The number of additional spaces to print for each second-level line.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Indentation.
    indent_level_1 = indent_1
    indent_level_2 = indent_1 + indent_2
    indent_str_1 = " " * indent_level_1
    indent_str_2 = " " * indent_level_2
    # Content.
    key_list_1 = sorted_keys(input_dict)  # Sort key list for consistent output order.
    for key_1 in key_list_1[0:max_keys_1]:
        print(indent_str_1 + str(key_1) + ":")
        # Fetch level 2 dictionary.
        dict_2 = input_dict[key_1]
        key_list_2 = sorted_keys(dict_2)  # Sort key list for consistent output order.
        # Determine the length of the longest key in this level 2 dictionary.
        level_2_key_max_len = max([len(str(x)) for x in key_list_2])
        format_str = "{0:<" + str(level_2_key_max_len + 2) + "s}{1:s}"
        for key_2 in key_list_2[0:max_keys_2]:
            # Fetch value.
            value_2 = dict_2[key_2]
            # Convert to string, and limit its length.
            value_2_str = str(value_2)
            trimmed_value_2_str = value_2_str[0:max_value_2_length]
            if len(value_2_str) > len(trimmed_value_2_str):
                trimmed_value_2_str += "..."
            # Print key : value.
            print(indent_str_2 + format_str.format(str(key_2) + ":", trimmed_value_2_str))
        # Level 2 postamble.
        if max_keys_2 < len(key_list_2):
            print(indent_str_2 + "...")
    # Level 1 postamble.
    if max_keys_1 < len(key_list_1):
        print(indent_str_1 + "...")


def print_dict_of_dict_of_dicts(
    input_dict,  # Dictionary to print.
    max_keys_1=5,  # Maximum number of level 1 keys to print.  Elipsis after that.
    max_keys_2=5,  # Maximum number of level 2 keys to print.  Elipsis after that.
    max_keys_3=8,  # Maximum number of level 3 keys to print.  Elipsis after that.
    max_value_3_length=70,  # Maximum level 3 value length to print.  Elipsis after that.
    indent_1=0,  # Number of blanks to print at the beginning of each top-level line.
    indent_2=4,  # Number of additional blanks to print for each second-level line.
    indent_3=4,
):  # Number of additional blanks to print for each third-level line.
    """
    Print a two-level nested dictionary, limiting the output length both laterally and vertically.

    Parameters
    ----------
    input_dict : dict
        The dictionary to print, where values are dictionaries that themselves contain dictionaries.
    max_keys_1 : int, optional
        The maximum number of top-level keys to print. An ellipsis will be shown if there are more keys.
    max_keys_2 : int, optional
        The maximum number of second-level keys to print. An ellipsis will be shown if there are more keys.
    max_keys_3 : int, optional
        The maximum number of third-level keys to print. An ellipsis will be shown if there are more keys.
    max_value_3_length : int, optional
        The maximum length of third-level values to print. An ellipsis will be shown if values exceed this length.
    indent_1 : int, optional
        The number of spaces to print at the beginning of each top-level line for indentation.
    indent_2 : int, optional
        The number of additional spaces to print for each second-level line.
    indent_3 : int, optional
        The number of additional spaces to print for each third-level line.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Indentation.
    indent_level_1 = indent_1
    indent_level_2 = indent_1 + indent_2
    indent_level_3 = indent_1 + indent_2 + indent_3
    indent_str_1 = " " * indent_level_1
    indent_str_2 = " " * indent_level_2
    indent_str_3 = " " * indent_level_3
    # Content.
    key_list_1 = sorted_keys(input_dict)  # Sort key list for consistent output order.
    for key_1 in key_list_1[0:max_keys_1]:
        print(indent_str_1 + str(key_1) + ":")
        # Fetch level 2 dictionary.
        dict_2 = input_dict[key_1]
        key_list_2 = sorted_keys(dict_2)  # Sort key list for consistent output order.
        # Determine the length of the longest key in this level 2 dictionary.
        level_2_key_max_len = max([len(str(x)) for x in key_list_2])
        format_str = "{0:<" + str(level_2_key_max_len + 2) + "s}{1:s}"
        for key_2 in key_list_2[0:max_keys_2]:
            print(indent_str_2 + str(key_2) + ":")
            # Fetch level 3 dictionary.
            dict_3 = dict_2[key_2]
            key_list_3 = sorted_keys(dict_3)  # Sort key list for consistent output order.
            # Determine the length of the longest key in this level 3 dictionary.
            level_3_key_max_len = max([len(str(x)) for x in key_list_3])
            format_str = "{0:<" + str(level_3_key_max_len + 2) + "s}{1:s}"
            for key_3 in key_list_3[0:max_keys_3]:
                # Fetch value.
                value_3 = dict_3[key_3]
                # Convert to string, and limit its length.
                value_3_str = str(value_3).replace("\n", "")
                trimmed_value_3_str = value_3_str[0:max_value_3_length]
                if len(value_3_str) > len(trimmed_value_3_str):
                    trimmed_value_3_str += "..."
                # Print key : value.
                print(indent_str_3 + format_str.format(str(key_3) + ":", trimmed_value_3_str))
            # Level 3 postamble.
            if max_keys_3 < len(key_list_3):
                print(indent_str_3 + "...")
        # Level 2 postamble.
        if max_keys_2 < len(key_list_2):
            print(indent_str_2 + "...")
    # Level 1 postamble.
    if max_keys_1 < len(key_list_1):
        print(indent_str_1 + "...")


# ---------------------------------------------------------------------------------------------------------------------------------------
#
# WRITE
#

# LIST ONE-LEVEL DICTIONARIES


def save_list_of_one_level_dicts(
    list_of_one_level_dicts, output_dir, output_body, explain, error_if_dir_not_exist, first_key=None
):
    """
    Save a list of one-level dictionaries to a CSV file, one dictionary per line.

    Parameters
    ----------
    list_of_one_level_dicts : list of dict
        A list of dictionaries to save to the CSV file.
    output_dir : str
        The directory where the CSV file will be saved.
    output_body : str
        The body of the output filename (without extension).
    explain : str
        An explanatory string to include in notification output.
    error_if_dir_not_exist : bool
        If True, raise an error if the output directory does not exist; if False, create the directory if necessary.
    first_key : str, optional
        A key to prioritize in the CSV heading line.

    Returns
    -------
    str
        The full path to the saved CSV file.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if len(list_of_one_level_dicts) == 0:
        explain += " (EMPTY)"
    key_list, heading_line = one_level_dict_csv_heading_line(list_of_one_level_dicts, first_key)
    data_lines = []
    for one_level_dict in list_of_one_level_dicts:
        data_lines.append(one_level_dict_csv_data_line(key_list, one_level_dict))
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_body,  # Body of output filename; extension is ".csv"
        heading_line,  # First line to write to file.  None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=error_if_dir_not_exist,
    )  # If True, error if not exist.  If False, create dir if necessary.
    # Return.
    return output_dir_body_ext


def one_level_dict_csv_heading_line(list_of_one_level_dicts, first_key):
    """
    Generate the heading line for a CSV file from a list of one-level dictionaries.

    Parameters
    ----------
    list_of_one_level_dicts : list of dict
        A list of one-level dictionaries to extract headings from.
    first_key : str, optional
        A key to prioritize in the heading line.

    Returns
    -------
    tuple
        A tuple containing the list of keys and the heading line as a string.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Catch empty input case.
    if len(list_of_one_level_dicts) == 0:
        return "No data."
    # Determine headings from keys of first dictionary.
    first_one_level_dict = list_of_one_level_dicts[0]
    # Analyze.
    return one_level_dict_csv_heading_line_aux(first_one_level_dict, first_key)


def one_level_dict_csv_heading_line_aux(first_one_level_dict, first_key):
    """
    Auxiliary function to generate the heading line for a CSV file from a one-level dictionary.

    Parameters
    ----------
    first_one_level_dict : dict
        The first one-level dictionary to extract headings from.
    first_key : str, optional
        A key to prioritize in the heading line.

    Returns
    -------
    tuple
        A tuple containing the list of keys and the heading line as a string.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    key_list = []
    first_key_found = False
    for key in first_one_level_dict.keys():
        if key == first_key:
            first_key_found = True
        else:
            key_list.append(key)
    # Verify that selceted first key is present.
    if (first_key != None) and (not first_key_found):
        print(
            "ERROR: In one_level_dict_heading_line_aux(), expected first key=" + str(first_key) + " not found in keys:",
            key_list,
        )
        assert False
    # Add first key, if specified.
    if first_key != None:
        key_list = [first_key] + key_list
    # Ensure all keys are strings.
    key_str_list = [str(k) for k in key_list]
    # Construct csv heading line.
    heading_line = ",".join(key_str_list)
    # Return.
    return key_list, heading_line


def one_level_dict_csv_data_line(key_list, one_level_dict):
    """
    Generate a CSV data line from a one-level dictionary.

    Parameters
    ----------
    key_list : list
        A list of keys to extract values from the dictionary.
    one_level_dict : dict
        The one-level dictionary to convert to a CSV data line.

    Returns
    -------
    str
        A string representing the CSV data line.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Assemble list of deta entry strings.
    data_str_list = []
    for key in key_list:
        data_str_list.append(str(one_level_dict[key]))
    # Convert to a csv data line.
    data_line = ",".join(data_str_list)
    return data_line


# LIST ONE-LEVEL DICTIONARY PAIRS


def save_list_of_one_level_dict_pairs(
    list_of_one_level_dict_pairs,
    output_dir,
    output_body,
    explain,
    error_if_dir_not_exist,
    first_key_1=None,
    first_key_2=None,
):
    """
    Save a list of one-level dictionary pairs to a CSV file, one dict per line.

    Parameters
    ----------
    list_of_one_level_dict_pairs : list of list
        A list of pairs of one-level dictionaries to save to the CSV file.
    output_dir : str
        The directory where the CSV file will be saved.
    output_body : str
        The body of the output filename (without extension).
    explain : str
        An explanatory string to include in notification output.
    error_if_dir_not_exist : bool
        If True, raise an error if the output directory does not exist; if False, create the directory if necessary.
    first_key_1 : str, optional
        A key to prioritize in the heading line for the first dictionary.
    first_key_2 : str, optional
        A key to prioritize in the heading line for the second dictionary.

    Returns
    -------
    str
        The full path to the saved CSV file.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    if len(list_of_one_level_dict_pairs) == 0:
        explain += " (EMPTY)"
    key_list_1, key_list_2, heading_line = one_level_dict_pair_csv_heading_line(
        list_of_one_level_dict_pairs, first_key_1, first_key_2
    )
    data_lines = []
    for one_level_dict_pair in list_of_one_level_dict_pairs:
        data_lines.append(one_level_dict_pair_csv_data_line(key_list_1, key_list_2, one_level_dict_pair))
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_body,  # Body of output filename; extension is ".csv"
        heading_line,  # First line to write to file.  None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=error_if_dir_not_exist,
    )  # If True, error if not exist.  If False, create dir if necessary.
    # Return.
    return output_dir_body_ext


def one_level_dict_pair_csv_heading_line(list_of_one_level_dict_pairs, first_key_1, first_key_2):
    """
    Generate the heading line for a CSV file from a list of one-level dictionary pairs.

    This routine extracts the headings from each of the two dicts, adding suffixes "_1" or "_2" to the first and
    second dictionary keys, respectively.  These suffixes are added regardless of whether the dictionaries have
    the same or different keys.

    Parameters
    ----------
    list_of_one_level_dict_pairs : list of list
        A list of pairs of one-level dictionaries to extract headings from.
    first_key_1 : str, optional
        A key to prioritize in the heading line for the first dictionary.
    first_key_2 : str, optional
        A key to prioritize in the heading line for the second dictionary.

    Returns
    -------
    tuple
        A tuple containing the list of keys for the first dictionary, the list of keys for the second dictionary,
        and the combined heading line as a string.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # Catch empty input case.
    if len(list_of_one_level_dict_pairs) == 0:
        return "No data."
    # Fetch the first pair.
    first_pair = list_of_one_level_dict_pairs[0]
    # Determine headings from keys of first dictionary.
    one_level_dict_1 = first_pair[0]
    key_list_1, heading_line_1 = one_level_dict_csv_heading_line_aux(one_level_dict_1, first_key_1)
    heading_line_tokens_1 = heading_line_1.split(",")
    heading_line_tokens_1b = [token + "_1" for token in heading_line_tokens_1]
    heading_line_1b = ",".join(heading_line_tokens_1b)
    # Determine headings from keys of second dictionary.
    one_level_dict_2 = first_pair[1]
    key_list_2, heading_line_2 = one_level_dict_csv_heading_line_aux(one_level_dict_2, first_key_2)
    heading_line_tokens_2 = heading_line_2.split(",")
    heading_line_tokens_2b = [token + "_2" for token in heading_line_tokens_2]
    heading_line_2b = ",".join(heading_line_tokens_2b)
    # Construct combined heading line.
    heading_line = heading_line_1b + "," + heading_line_2b
    # Return.
    return key_list_1, key_list_2, heading_line


def one_level_dict_pair_csv_data_line(key_list_1, key_list_2, one_level_dict_pair):
    """
    Generate a CSV data line from a pair of one-level dictionaries.

    Parameters
    ----------
    key_list_1 : list
        A list of keys to extract values from the first dictionary.
    key_list_2 : list
        A list of keys to extract values from the second dictionary.
    one_level_dict_pair : list
        A pair of one-level dictionaries to convert to a CSV data line.

    Returns
    -------
    str
        A string representing the CSV data line for the pair of dictionaries.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # First dictionary data line.
    one_level_dict_1 = one_level_dict_pair[0]
    data_line_1 = one_level_dict_csv_data_line(key_list_1, one_level_dict_1)
    # Second dictionary data line.
    one_level_dict_2 = one_level_dict_pair[1]
    data_line_2 = one_level_dict_csv_data_line(key_list_2, one_level_dict_2)
    # Construct combined heading line.
    data_line = data_line_1 + "," + data_line_2
    return data_line
