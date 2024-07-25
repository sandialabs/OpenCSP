"""
Utilities for file_handling.



"""

import csv
from datetime import datetime
import glob
import json
import os
import os.path
import shutil
import tempfile
from typing import Optional

# try to import as few other opencsp libraries as possible
import opencsp.common.lib.file.CsvInterface as csvi
import opencsp.common.lib.process.subprocess_tools as subt
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.math_tools as mt


def path_components(input_dir_body_ext: str):
    """
    Breaks input string "~/foo/bar/blech.grrr.txt" into:
        dir:  "~/foo/bar"
        body: "blech.grrr"
        ext:  ".txt"

    Returns:
    --------
        tuple: dir (sans last slash), body (name of the file), ext (includes the leading ".")

    See also: body_ext_given_file_dir_body_ext()
    """
    dir = os.path.dirname(input_dir_body_ext)
    body_ext = os.path.basename(input_dir_body_ext)
    body, ext = os.path.splitext(body_ext)
    return dir, body, ext


def path_to_cmd_line(path: str):
    """Normalizes and surrounds a path with quotes, as necessary."""
    ret = os.path.normpath(path)
    if " " in ret:
        ret = "\"" + ret + "\""
    return ret


def norm_path(path: str, allow_extended_length_path=True):
    """
    Normalizes the given path (use system-style slashes) and prepend the
    long-path signifier, as necessary.
    """
    # normalize the path to use all the same slashes
    path = os.path.normpath(path)

    # check for long paths
    if os.name == "nt":
        # On windows, paths are limited to 260 characters, including the null
        # terminator. However, if you add "\\?\" at the beginning.
        # https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
        extended_path_prefix = "\\\\?\\"
        if len(path) > 259:
            # Don't do it this way:
            #     if not path.startswith(extended_path_prefix):
            # The reason to use "in" instead of "startswith" is to allow for
            # normalizing of weird path edge cases that I (BGB) can't think of
            # here. Maybe something like a web address or some such...
            if not extended_path_prefix in path:
                if allow_extended_length_path:
                    path = extended_path_prefix + path
                else:
                    lt.warn(f"Long path name \"{path}\" detected.")

    return path


def join(*path_components: str):
    """
    Joins and normalize the given path components. For example:

        join("a", "b/c.txt")

    ...is the same as:

        norm_path(os.path.join("a", "b/c.txt"))
    """
    return norm_path(os.path.join(*path_components))


def body_ext_given_file_dir_body_ext(inputdir_body_ext: str):
    """Return the name+ext for the given path+name+ext.

    See also: path_components()"""
    dir, body, ext = path_components(inputdir_body_ext)
    return body + ext


def resolve_symlink(input_dir_body_ext: str):
    """If the input is a symbolic link (os.path.islink), then get the real path."""
    if os.path.islink(input_dir_body_ext):
        return os.path.realpath(input_dir_body_ext)
    return input_dir_body_ext


def file_exists(input_dir_body_ext: str, error_if_exists_as_dir=True, follow_symlinks=False):
    """
    Determines whether the given file exists.
    If the specified input path exists but is a directory instead of a file,
    then halts with an error, if error_if_exists_as_dir==True.
    """
    if follow_symlinks:
        input_dir_body_ext = resolve_symlink(input_dir_body_ext)
    # Check to see if the file exists as a directory.
    if os.path.isdir(input_dir_body_ext):
        if error_if_exists_as_dir == True:
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In file_exists(), requested input path exists and is a directory: ' + str(input_dir_body_ext),
            )
        else:
            return False
    # Check to see if the file exists.
    if os.path.isfile(input_dir_body_ext):
        return True
    else:
        return False


def directory_exists(input_dir: str, error_if_exists_as_file=True, follow_symlinks=False):
    """
    Determines whether the given directory exists.
    If the specified input directory exists but is a file instead of a directory,
    then halts with an error, if error_if_exists_as_file==True.
    """
    if follow_symlinks:
        input_dir = resolve_symlink(input_dir)
    # Check to see if the directory exists as a file.
    if os.path.isfile(input_dir):
        if error_if_exists_as_file == True:
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In directory_exists(), requested input path exists and is a file: ' + str(input_dir),
            )
        else:
            return False
    # Check to see if the directory exists.
    if os.path.isdir(input_dir):
        return True
    else:
        return False


def directory_is_empty(input_dir):
    """
    Determines whether the given directory is empty.
    The standard Unix files "." and ".." are ignored.
    Halts with an error if the directory does not exist.
    """
    # Check input.
    if os.path.isfile(input_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In directory_is_empty(), requested input path exists and is a file: ' + str(input_dir)
        )
    if not os.path.isdir(input_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In directory_is_empty(), requested input directory does not exist: ' + str(input_dir)
        )
    # Probe the directory contents to determine whether any contents are there.
    # The standard Unix files "." and ".." don't count as contents.
    # This test does not require scanning the entire directory contents, making a
    # list of all the filenames, etc, and thus is faster than using os.listdir().
    with os.scandir(input_dir) as it:
        for entry in it:
            if (entry.name == '.') or (entry.name == '..'):
                # Ignore these files.
                continue
            # We found an entry, so the directory is not empty.
            return False
    return True


def count_items_in_directory(
    input_dir, name_prefix=None, name_suffix=None  # Only entries with names thata start with name_prefix are counted.
):  # Only entries with names thata end with name_suffix are counted.
    """
    Counts the number of items in the given directory.
    Does not discriminate between directories, files, or symbolic links -- all are counted.
    The standard Unix files "." and ".." are ignored.
    Only counts names with given name prefix and/or suffix, if given (case sensitive).
    Halts with an error if the directory does not exist.
    """
    # Check input.
    if not os.path.exists(input_dir):
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In count_items_in_directory(), requested input directory does not exist: ' + str(input_dir),
        )
    if not os.path.isdir(input_dir):
        if os.path.isfile(input_dir):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In count_items_in_directory(), requested input path exists and is a file: ' + str(input_dir),
            )
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In count_items_in_directory(), requested input directory is not a directory: ' + str(input_dir),
        )
    # Walk the directory contents to determine whether any contents are there.
    # The standard Unix files "." and ".." don't count as contents.
    # This test does not construct a list of the entire directory contents, and
    # thus is faster than using os.listdir().
    count = 0
    with os.scandir(input_dir) as it:
        for entry in it:
            # Ignore standard Unix files.
            if (entry.name == '.') or (entry.name == '..'):
                # Ignore these files.
                continue
            # Check prefix and suffix.
            if (name_prefix == None) or (entry.name.startswith(name_prefix)):
                prefix_ok = True
            else:
                prefix_ok = False
            # Check suffix.
            if (name_suffix == None) or (entry.name.endswith(name_suffix)):
                suffix_ok = True
            else:
                suffix_ok = False
            if (not prefix_ok) or (not suffix_ok):
                # Ignore these files.
                continue
            # Valid entry.
            count += 1
    return count


def binary_count_items_in_directory(input_dir: str, name_pattern: str, start=1) -> int:
    """Counts the number of items n in a directory by looking for one file at a time.

    Starts with file 1, then 3, then 7, ..., then n, then n-n/2+n/4, ..., n.

    Args:
        input_dir (str): The directory to search in.
        name_pattern (str): The pattern for the filenames to match. For example "C0044_s2m25_d870.MP4.id_%06.JPG".
        start (int): Where to start searching at. Most useful for files indexed at 0.

    Returns:
        int: The number of files found.
    """
    curr_val = start
    prev_val = curr_val
    found_file = False

    search_size = 1
    while os.path.exists(os.path.join(input_dir, name_pattern % curr_val)):
        search_size *= 2
        prev_val = curr_val
        curr_val += search_size
        found_file = True

    while curr_val != prev_val:
        if not os.path.exists(os.path.join(input_dir, name_pattern % curr_val)):
            if curr_val == prev_val + 1:
                return prev_val - start + 1
            else:
                search_size = -int((curr_val - prev_val) / 2)
                curr_val += search_size
        else:
            prev_val = curr_val
            search_size = 1 if (search_size < 0) else search_size * 2
            curr_val += search_size

    if not found_file:
        return 0
    return curr_val - start + 1


def file_size(input_dir_body_ext, error_if_exists_as_dir=True):
    """
    Returns the size of the given file in bytes.
    """
    if not file_exists(input_dir_body_ext, error_if_exists_as_dir):
        lt.error_and_raise(
            FileNotFoundError,
            'ERROR: In file_size(), input_dir_body_ext '
            + 'was not found.\n\tinput_dir_body_ext ='
            + input_dir_body_ext,
        )
    return os.path.getsize(input_dir_body_ext)


def file_size_pair_name(file_size_pair) -> str:
    """Get the file name from a [file,size] pair, such as from file_size() or files_in_directory_with_associated_sizes()"""
    return file_size_pair[0]


def file_size_pair_size(file_size_pair) -> int:
    """Get the file size from a [file,size] pair, such as from file_size() or files_in_directory_with_associated_sizes()"""
    return file_size_pair[1]


def files_in_directory(input_dir, sort=True, files_only=False, recursive=False):
    """Returns a list [ file1, file2, ...] of files in the given directory.

    The returned values include the "name.ext" of the file.
    For files_only=False, all entries in the directory, including the names of directories, are returned.
    Does not include the unix "." (current) and ".." (parent) directories.

    Args:
    -----
    sort: bool
        If True, then the list is sorted in order of ascending file name. Default True.
    files_only: bool
        If True, then return only the file entries. Default False.
    recursive: bool
        If true, then walk through all files in the given directory and all
        subdirectories. Does not follow symbolic links to directories. Default
        False.

    Returns:
    --------
        files_name_ext (list[str]): The list of file name_exts (example ["a.csv", "b.csv", ...])
    """
    scanned_files: list[str] = []
    file_list: list[str] = []

    if not recursive:
        # Walk the directory and assemble a list of files.
        with os.scandir(input_dir) as iter:
            for entry in iter:
                # Ignore directories
                if files_only:
                    if not entry.is_file():
                        continue

                # Valid entry.
                file_name = entry.name
                scanned_files.append(file_name)
    else:
        # Walk the directory and all subdirectories and assemble a list of files.
        norm_input_dir = norm_path(input_dir)
        for root, dirnames, file_names_exts in os.walk(norm_input_dir):
            # remove the front of the path, in order to get the path relative to the input_dir
            relative_path: str = root.replace(norm_input_dir, "")
            # don't include any leading ./
            if relative_path.startswith("./") or relative_path.startswith(".\\"):
                relative_path = relative_path[2:]
            # don't include any leading /
            relative_path = relative_path.lstrip("\\/")

            scanned_files += [os.path.join(relative_path, file_name_ext) for file_name_ext in file_names_exts]

            # Ignore directories
            if not files_only:
                scanned_files += [os.path.join(relative_path, dirname) for dirname in dirnames]

    # Ignore standard Unix files.
    for file_relpath_name_ext in scanned_files:
        relpath, name, ext = path_components(file_relpath_name_ext)
        if (name == '.') or (name == '..'):
            # Ignore these files.
            continue
        file_list.append(file_relpath_name_ext)

    # Sort, if desired.
    if sort:
        file_list.sort()

    # Return.
    return file_list


def files_in_directory_with_associated_sizes(input_dir, sort=True, follow_symlinks=True):
    """
    Returns a list [ [file1 size1], [file2, size2], ...] of files name_ext and associated sizes.
    If sort==True, then the list is sorted in order of ascending file name.

    See also: file_size_pair_name(), file_size_pair_size()
    """
    # Walk the directory and assemble a list of [file, size] pairs.
    file_size_pair_list: list[tuple[str, int]] = []
    with os.scandir(input_dir) as it:
        for entry in it:
            # Ignore standard Unix files.
            if (entry.name == '.') or (entry.name == '..'):
                # Ignore these files.
                continue
            # Valid entry.
            file_name = entry.name
            stat_result = entry.stat(follow_symlinks=follow_symlinks)
            file_size = stat_result.st_size
            file_size_pair_list.append([file_name, file_size])

    # Sort by filename, if desired.
    if sort:
        file_size_pair_list.sort(key=file_size_pair_name)

    # Return.
    return file_size_pair_list


def files_in_directory_by_extension(
    input_dir: str, extensions: list[str], sort=True, case_sensitive=False, recursive=False
):
    """Generates a list of { ext: [file1, file2, ...], ... }. Only returns the files
    with one of the given extensions.

    Arguments:
    ----------
    input_dir: str
        directory to search for files
    extensions: list[str]
        list of extensions for the files (with or without leading periods ".")
    sort: bool
        if True, then sort the files by name before returning. Defaults to True
    case_sensitive: bool
        If True, then the file extensions matching are case sensitive. Defaults to False
    recursive: bool
        If true, then walk through all files in the given directory and all
        subdirectories. Does not follow symbolic links to directories. Default
        False.

    Returns:
    --------
    files: dict[str,list[str]]
        The found file name_exts, by their associated extension.
        For example {  "csv": ["a.csv","b.csv"], "txt": ["foo.txt","bar.txt"]  }
    """
    ret: dict[str, list[str]] = {}
    search_extensions = {}

    for ext in extensions:
        ext = str(ext)
        ret[ext] = []
        iext = ext if case_sensitive else ext.lower()
        if ext.startswith("."):
            search_extensions[ext] = iext
        else:
            search_extensions["." + ext] = iext

    for file in files_in_directory(input_dir, sort, files_only=True, recursive=recursive):
        _, file_ext = os.path.splitext(file)
        ifile_ext = file_ext if case_sensitive else file_ext.lower()
        ext = search_extensions.get(ifile_ext)
        if ext != None:
            ret[ext].append(file)

    return ret


def create_file(input_dir_body_ext: str, error_on_exists=True, delete_if_exists=False):
    """Creates the given file, and checks that it was successfully created."""
    input_dir, _, _ = path_components(input_dir_body_ext)
    input_body_ext = body_ext_given_file_dir_body_ext(input_dir_body_ext)

    # check for existance
    if not directory_exists(input_dir):
        lt.error_and_raise(
            RuntimeError,
            f"Error: in create_file(), requested directory doesn't exist: '{input_dir}' (file '{input_body_ext}')",
        )
    if file_exists(input_dir_body_ext):
        if delete_if_exists:
            delete_file(input_dir_body_ext)
        elif error_on_exists:
            lt.error_and_raise(
                RuntimeError, f"Error: in create_file(), requested file already exists: '{input_dir_body_ext}'"
            )
        else:
            return

    # create the file
    try:
        with open(input_dir_body_ext, "w"):
            pass
    except:
        lt.error(f"Error: in create_file(), failed to create file '{input_dir_body_ext}'")
        raise

    # check for success
    if not file_exists(input_dir_body_ext):
        lt.error_and_raise(FileNotFoundError, "Error: in create_file(), failed to create file")


def delete_file(input_dir_body_ext, error_on_not_exists=True):
    """
    Deletes the given file, after checking to make sure it is a file.
    """
    # Check input.
    if not os.path.isfile(input_dir_body_ext):
        if not os.path.exists(input_dir_body_ext):
            if error_on_not_exists:
                lt.error_and_raise(
                    RuntimeError,
                    'ERROR: In delete_file(), requested input path is not an existing file: ' + str(input_dir_body_ext),
                )
            return
        lt.error_and_raise(
            RuntimeError, 'ERROR: In delete_file(), requested input path is not a file: ' + str(input_dir_body_ext)
        )
    try:
        os.remove(input_dir_body_ext)
    except OSError as e:
        print(
            'ERROR: In delete_file()), attempt to delete file '
            + input_dir_body_ext
            + ' resulted in error: '
            + e.strerror
        )
        raise  # if this should NOT raise an exception then that should be documented, as well as the reason why it shouldn't ~BGB230119


def delete_files_in_directory(input_dir: str, globexp: str, error_on_dir_not_exists=True):
    """
    Deletes files in the given directory matching the globexp.

    Does not remove the directory, its subdirectories, or files not matching the globexp.
    If you absolute MUST remove a directory in its entirety, you're probably doing it wrong
    (maybe use globexp="*" instead?).

    Note: One can imagine a more aggressive version that removes the entire directory tree, using shutil.rmtree(dir_path).
          (See for example https://linuxize.com/post/python-delete-files-and-directories/.)
          But I intentionally decided *not* to do this, because of the hazard of a disastrous file deletion accident.
          This could occur if someone programming the updatream calling code inadvertently passes in a directory that
          is high in the tree of files we care about.  This could happen, for example, by setting a path variable to
          include a root or middle node of a tree of interest, but forgetting to fill in all the steps to the leaf
          directory.  By implementing this routine to only delete the files in the specific directory, damage in
          such an event is limited.

    Args:
    -----
        - input_dir (str): The directory to remove files from.
        - globexp (str): The glob expression used to match files to be deleted, eg "*.jpg" for image files. "*" for all files.

    Returns:
    --------
        - removed (list[str]): The files that were matched and removed.
    """
    # Check input.
    if os.path.isfile(input_dir):
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In delete_items_in_directory(), requested input path exists and is a file: ' + str(input_dir),
        )
    if not os.path.isdir(input_dir):
        if error_on_dir_not_exists:
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In delete_items_in_directory(), requested input directory does not exist: ' + str(input_dir),
            )
    # Delete the files.
    localized_globexp = os.path.join(input_dir, globexp)
    files = glob.glob(localized_globexp)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            if e.strerror == "No such file or directory":
                # This occurs when running in parallel, and another
                # process/server has already deleted the file before we could.
                pass
            else:
                print(
                    'ERROR: In delete_files_in_directory(), attempt to delete file '
                    + f
                    + ' resulted in error: '
                    + e.strerror
                )

    return files


def create_directories_if_necessary(input_dir):
    """
    Ensures that the given directory existins, along with all of its parents.
    """
    # Check input.
    if os.path.isfile(input_dir):
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In create_directories_if_necessary(), requested input path exists and is a file: ' + str(input_dir),
        )
    # Create the directory and its parents, if they do not exist.
    if not os.path.isdir(input_dir):
        try:
            os.makedirs(input_dir)
        except FileExistsError:
            # probably just created this directory in another thread
            pass


def create_subdir_path(base_dir: str, dir_name: str):
    """
    Constructs and returns path including subdirectory name, without interacting with the file system.
    Therefore the returned subdirectory path might be invalid.
    """
    return os.path.join(base_dir, dir_name)


def create_subdir(base_dir: str, dir_name: str, error_if_exists=True):
    """
    Constructs and returns path including subdirectory name, and also creates the specified
    subdirectory, checking for file system errors.

    See also: create_directories_if_necessary()
    """
    # Check input.
    if not os.path.exists(base_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In create_subdir(): Containing directory does not exist:\n\t' + base_dir
        )
    result_dir = os.path.join(base_dir, dir_name)
    if os.path.exists(result_dir):
        if error_if_exists:
            lt.error_and_raise('ERROR: In create_subdir(), Result directory exists:\n\t' + result_dir)
        else:
            pass
    else:
        try:
            os.mkdir(result_dir)
        except:
            lt.error_and_raise(RuntimeError, 'ERROR: In create_subdir(): Could not create directory: ' + result_dir)
    return result_dir


def directories_in_directory(directory, sort=True):
    """
    Returns a list of all the directory names contained within the input directory.
    """
    dir_list = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    if sort:
        dir_list.sort()
    return dir_list


def directories_with_no_leading_underscore(directory, sort=True):
    """
    Returns a list of all the directory names contained within the input directory,
    which do not begin with a  leading underscore.
    """
    dir_list = directories_in_directory(directory, sort=sort)
    return [d for d in dir_list if not d.startswith('_')]


def convert_string_to_file_body(txt):
    """Convert the given string into something more filesystem friendly by replacing special characters and double underscores."""

    # Create output variable to modify.
    output_file_body = txt
    # Eliminate special characters.
    output_file_body = output_file_body.replace(',', '_')
    output_file_body = output_file_body.replace('/', '_div_')
    output_file_body = output_file_body.replace(':', '_')
    output_file_body = output_file_body.replace('^', '')
    output_file_body = output_file_body.replace('=', '_eq_')
    output_file_body = output_file_body.replace('(', '_')
    output_file_body = output_file_body.replace(')', '_')
    # Replace blanks with underscores.
    output_file_body = output_file_body.replace(' ', '_')
    # Eliminate double underscores.
    while '__' in output_file_body:
        output_file_body = output_file_body.replace('__', '_')
    # Return.
    return output_file_body


_output_paths = {}


def default_output_path(file_path_name_ext: Optional[str] = None) -> str:
    """Get the default output directory. Defaults to '<execution_dir>/../Y_m_d_HM'.
    The intended use for this function is for human-consumable results. Also consider
    opencsp_root_path.opencsp_temporary_dir() for other kinds of results.

    The same output will always be produced for a given file_path_name_ext during
    the duration of the python execution.

    Arguments
    ---------
        file_path_name_ext (str): If a existing file or directory is given, then make
                                  the output directory relative to this directory.

    Returns
    -------
        file_path (str): The directory to save the given file in

    See also: time_date_tools.current_date_time_string_forfile()
    """

    # return the previously generated output path, if any
    if file_path_name_ext in _output_paths:
        return _output_paths[file_path_name_ext]

    # generate a new output path
    output_dir_name = 'output_' + datetime.now().strftime('%Y_%m_%d_%H%M')
    if file_path_name_ext != None and os.path.exists(file_path_name_ext):
        file_dir = file_path_name_ext if os.path.isdir(file_path_name_ext) else os.path.dirname(file_path_name_ext)
        if file_dir != "":
            _output_paths[file_path_name_ext] = os.path.join(file_dir, output_dir_name)
        else:
            _output_paths[file_path_name_ext] = os.path.join(
                'common', 'lib', 'tool', 'test', 'data', 'output', 'file_tools', output_dir_name
            )
    else:
        _output_paths[file_path_name_ext] = os.path.join(
            'common', 'lib', 'tool', 'test', 'data', 'output', 'file_tools', output_dir_name
        )

    return _output_paths[file_path_name_ext]


def rename_file(input_dir_body_ext: str, output_dir_body_ext: str, is_file_check_only=False):
    """Move a file from input to output.

    Verifies that input is a file, and that the output doesn't exist. We check
    for existence of the output file after the rename, and raise a
    FileNotFoundError if it can't be found.

    This operation could fail if the source and destination directories aren't
    on the same file system. We check for existence of the output file after
    the rename, and raise a FileNotFoundError if it can't be found.

    Args:
        - is_file_check_only (bool): If True, then only check that input_dir_body_ext is a file. Otherwise, check everything.

    See also: copy_file(), copy_and_delete_file()
    """
    if os.path.normpath(input_dir_body_ext) == os.path.normpath(output_dir_body_ext):
        return

    # Check input.
    if not is_file_check_only:
        if os.path.isdir(input_dir_body_ext):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In rename_file(), requested input path exists and is a directory: ' + str(input_dir_body_ext),
            )
    if not os.path.isfile(input_dir_body_ext):
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In rename_file(), requested input file to rename does not exist: ' + str(input_dir_body_ext),
        )
    if not is_file_check_only:
        if os.path.isfile(output_dir_body_ext):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In rename_file(), requested output file exists and is a file: ' + str(output_dir_body_ext),
            )
        if os.path.isdir(output_dir_body_ext):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In rename_file(), requested output file exists as a directory: ' + str(output_dir_body_ext),
            )
        if os.path.exists(output_dir_body_ext):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In rename_file(), requested output file exists as something besides a file or directory: '
                + str(output_dir_body_ext),
            )
        if not os.path.isdir(os.path.dirname(output_dir_body_ext)):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In rename_file(), requested output path does not exist or is not a directory: '
                + str(os.path.dirname(output_dir_body_ext)),
            )
    # Rename the file.
    os.rename(input_dir_body_ext, output_dir_body_ext)
    # Verify the rename
    if not is_file_check_only:
        if not os.path.exists(output_dir_body_ext):
            lt.error_and_raise(
                FileNotFoundError,
                f"Error: In rename_file(), failed to find output file after rename: '{input_dir_body_ext}' --> '{output_dir_body_ext}'",
            )


def copy_and_delete_file(input_dir_body_ext: str, output_dir_body_ext: str):
    """Like rename_file(), but it works across file systems.

    See also: copy_file(), rename_file()"""
    if os.path.normpath(input_dir_body_ext) == os.path.normpath(output_dir_body_ext):
        return

    # copy and rename
    output_dir, output_body, output_ext = path_components(output_dir_body_ext)
    output_body_ext = output_body + output_ext
    copy_file(input_dir_body_ext, output_dir, output_body_ext)

    # delete the original
    delete_file(input_dir_body_ext, error_on_not_exists=False)


def copy_file(input_dir_body_ext: str, output_dir: str, output_body_ext: str = None):
    """Copies a file from input to output.

    Verifies that input is a file, and that the output doesn't exist.

    We check for existance of the output file after the copy, and raise a FileNotFoundError if it can't be found.

    Returns
    -------
    input_dir_body_ext: str
        The "dir/body.ext" of the file to be copied.
    output_dir: str
        Where to copy the file to.
    output_body_ext: str, optional
        What to name the file in the destination directory. If None, then use
        the "body.ext" from input_dir_body_ext. Default None.

    See also: copy_and_delete_file(), rename_file()
    """
    # Check input.
    input_dir_body_ext = norm_path(input_dir_body_ext)
    if os.path.isdir(input_dir_body_ext):
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In copy_file(), requested input path exists and is a directory: ' + str(input_dir_body_ext),
        )
    if not os.path.isfile(input_dir_body_ext):
        lt.error_and_raise(
            RuntimeError,
            'ERROR: In copy_file(), requested input file to copy does not exist: ' + str(input_dir_body_ext),
        )
    if os.path.isfile(output_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In copy_file(), requested output path exists and is a file: ' + str(output_dir)
        )
    if not os.path.isdir(output_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In copy_file(), requested output directory does not exist: ' + str(output_dir)
        )
    # Assemble the output file path.
    if output_body_ext == None:
        input_body_ext = body_ext_given_file_dir_body_ext(input_dir_body_ext)
        output_body_ext = input_body_ext
    output_dir_body_ext = os.path.join(output_dir, output_body_ext)
    output_dir_body_ext = norm_path(output_dir_body_ext)
    # Check output.
    if file_exists(output_dir_body_ext):
        lt.error_and_raise(
            FileExistsError, 'ERROR: In copy_file(), requested output file already exists: ' + str(output_dir_body_ext)
        )

    # Copy the file.
    if input_dir_body_ext == output_dir_body_ext:
        return output_body_ext
    shutil.copyfile(input_dir_body_ext, output_dir_body_ext)

    # Verify the copy
    if not os.path.exists(output_dir_body_ext):
        lt.error_and_raise(
            FileNotFoundError,
            f"Error: In copy_file(), failed to find output file after copy: '{input_dir_body_ext}' --> '{output_dir_body_ext}'",
        )

    return output_body_ext


def get_temporary_file(suffix: str = None, dir: str = None, text: bool = True) -> tuple[int, str]:
    """
    Creates a temporary file to write to. Example usage::

        fd, fname = ft.get_temporary_file()
        try:
            with open(fd, 'w') as fout:
                fout.write("foo")
            ...
        finally:
            ft.delete_file(fname)

    Args:
    -----
        - suffix (str): Suffix of the file name. Be sure to include the leading "." for file extension suffixes.
        - dir (str): Where to save the temporary file with the list of frame names.
                     Defaults to the opencsp_temporary_dir if writable, or else the home directory if writable, or else /tmp.
        - text (bool): True to open the file in text mode. False to open it in byte mode.

    Returns:
    --------
        - int: The file descriptor for the new file.
        - str: The path_name_ext of the new file.
    """
    # import here to reduce import loops possibilities
    import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
    import opencsp.common.lib.tool.system_tools as st

    default_possible_dirs = [orp.opencsp_temporary_dir(), os.path.expanduser('~'), tempfile.gettempdir()]
    possible_dirs = [dir] if dir != None else default_possible_dirs
    success = False

    for dirname in possible_dirs:
        fd, fname = -1, ""
        try:
            fd, fname = tempfile.mkstemp(suffix=suffix, dir=dirname, text=text)
        except FileNotFoundError:
            continue
        success = True
        break

    if not success:
        raise FileNotFoundError(f"Could not create a tempory file in the directory '{dirname}'!")
    return fd, fname


def merge_files(in_files: list[str], out_file: str, overwrite=False, remove_in_files: bool = False):
    """Merges the given files into a single file. The order of the input files
    is preserved in the output file.

    Args:
        in_files (list[str]): A list of file path_name_exts to be merged
        out_file (str): A file path_name_ext to merge into
        overwrite (bool): If False, then check that out_file doesn't already exist, and error if it does. Defaults to False.
        remove_in_files (bool): Remove the input files as they are added to the output file. Defaults to False.
    """
    if not overwrite:
        if file_exists(out_file):
            lt.error_and_raise(
                RuntimeError, f"Error: in file_tools.merge_files: output file already exists \"{out_file}\""
            )
    with open(out_file, 'wb') as wfd:
        for f in in_files:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
            if remove_in_files:
                os.remove(f)


def convert_shortcuts_to_symlinks(dirname: str):
    if os.name == "nt":
        # update dirname to use windows "\" seperators
        dirname = os.path.normpath(dirname)

        # get a shell instance, to use for retrieving shortcut targets
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")

        # convert all shortcuts in the given directory
        for filename in os.listdir(dirname):
            shortcut_dir_path_ext = os.path.join(dirname, filename)
            if shortcut_dir_path_ext.endswith(".lnk"):
                # Get the source path that the shortcut points to. Do this
                # process recursively, in case the shortcut points to a shortcut
                # points to shortcut...
                source_path_name_ext = shortcut_dir_path_ext
                while source_path_name_ext.endswith(".lnk"):
                    try:
                        source_path_name_ext = shell.CreateShortCut(shortcut_dir_path_ext).Targetpath
                    except Exception as e:
                        lt.error(f"Failed to read the shortcut {shortcut_dir_path_ext}")
                        raise

                # get the destination (link) path
                _, shortcut_name, shortcut_ext = path_components(filename)
                link_name_ext = shortcut_name + shortcut_ext
                if link_name_ext.endswith(" - Shortcut.lnk"):
                    link_name_ext = link_name_ext[: -len(" - Shortcut.lnk")]
                elif link_name_ext.endswith(".lnk"):
                    link_name_ext = link_name_ext[: -len(".lnk")]
                link_path_name_ext = os.path.join(dirname, link_name_ext)
                if os.path.islink(link_path_name_ext):
                    lt.debug(f"In file_tools.convert_shortcuts(): link {link_path_name_ext} already exists")
                    continue
                elif file_exists(
                    link_path_name_ext, error_if_exists_as_dir=False, follow_symlinks=True
                ) or directory_exists(link_path_name_ext, error_if_exists_as_file=False, follow_symlinks=True):
                    lt.error_and_raise(
                        FileExistsError,
                        "Error in file_tools.convert_shortcuts(): "
                        + f"the destination symbolic link {link_path_name_ext} already exists as a file or directory!",
                    )

                # create the link
                directory_flag = ""
                if directory_exists(source_path_name_ext, error_if_exists_as_file=False, follow_symlinks=True):
                    # is a directory
                    directory_flag = "/D"
                try:
                    subt.run(f"mklink {directory_flag} {link_path_name_ext} {source_path_name_ext}")
                except Exception as e:
                    link_dir, link_name_ext, link_ext = path_components(link_path_name_ext)
                    lt.error(
                        f"Failed to create the symbolic link '{link_name_ext}{link_ext}' in '{link_dir}' to '{source_path_name_ext}'"
                    )
                    raise
    else:
        lt.debug("No shortcuts to convert. Shortcuts are only found on Windows.")


# TEXT FILES


def write_text_file(
    description: str, output_dir: str, output_file_body: str, output_string_list: list[any], error_if_dir_not_exist=True
) -> str:
    """Writes a strings to a ".txt" file, with each string on a new line.

    Parameters
    ----------
    description : str
        Explanatory string to include in notification output.  None to skip.
    output_dir : str
        Which directory to write the file to.  See below if not exist.
    output_file_body : str
        Name of the file without an extension. A standard ".txt" extension will
        be automatically appended.
    output_string_list : list
        List of values to write to the file, one per line. Newlines "\\n" will
        be automatically appended to each line.
    error_if_dir_not_exist : bool, optional
        If True and output_dir doesn't exists, raise an error. If False, create
        output_dir as necessary. By default True.

    Returns
    -------
    output_dir_body_ext : str
        The "path/name.ext" of the newly created file.
    """
    # Check status of output_dir.
    if os.path.isfile(output_dir):
        lt.error_and_raise(
            FileExistsError,
            'ERROR: In write_text_file(), requested output path exists and is a file: ' + str(output_dir),
        )
    if error_if_dir_not_exist == True:
        if not directory_exists(output_dir):
            lt.error_and_raise(
                FileNotFoundError,
                'ERROR: In write_text_file(), requested output directory does not exist: ' + str(output_dir),
            )
    else:
        create_directories_if_necessary(output_dir)
    # Write output file.
    output_body_ext = convert_string_to_file_body(output_file_body) + '.txt'
    output_dir_body_ext = os.path.join(output_dir, output_body_ext)
    if description is not None:
        print('Saving ' + description + ': ', output_dir_body_ext)
    with open(output_dir_body_ext, 'w') as output_stream:
        # Write strings.
        for output_str in output_string_list:
            output_stream.write(
                str(output_str) + '\n'
            )  # Call str, just in case somebody passes in a list of ints, etc.
    # Return.
    return output_dir_body_ext


def read_text_file(input_dir_body_ext):
    """
    Reads a text file.
    Assumes input file exists, and is a text file.
    File extension may be arbitrary.
    Returns a list of strings, one per line.  No parsing.
    """
    # Check input.
    if not file_exists(input_dir_body_ext):
        lt.error_and_raise(IOError, 'ERROR: In read_text_file(), file does not exist: ' + str(input_dir_body_ext))
    # Open and read the file.
    with open(input_dir_body_ext, newline='') as input_stream:
        lines = input_stream.readlines()
    return lines


# CSV FILES


def write_csv_file(
    description,  # Explanatory string to include in notification output.  None to skip.
    output_dir,  # Directory to write file.  See below if not exist.
    output_file_body,  # Body of output filename; automatically appends ".csv" extension
    heading_line,  # First line to write to file.  None to skip.
    data_lines,  # Subsequent lines to write to file.
    error_if_dir_not_exist=True,  # If True, error if not exist.  If False, create dir if necessary.
    log_warning=True,
):
    """Deprecated. Use to_csv() instead, to better match naming in pandas.

    Writes a ".csv" file with a heading line and subsequent data lines.
    This implementation expects that each line is a simple string, with "," separators already added.
    """
    # TODO remove "write_csv_file" in favor of "to_csv" to match naming convention from pandas
    if log_warning:
        lt.info("'write_csv_file' is deprecated in favor of 'to_csv'")
    return to_csv(description, output_dir, output_file_body, heading_line, data_lines, error_if_dir_not_exist)


def to_csv(
    description: str | None,
    output_dir: str,
    output_file_body: str,
    heading_line: str | bool | None,
    data_lines: list[str | csvi.CsvInterface],
    error_if_dir_not_exist: bool = True,
    overwrite=False,
):
    """Writes a ".csv" file with a heading line and subsequent data lines.

    This implementation expects that each line is a simple string, with "," separators already added.

    Parameters
    ----------
        description: str|None
                     Explanatory string to include in notification output.  None to skip.
        heading_line: str|None|True
                      First line to write to file.  None to skip.  True to use data_lines[0].csv_header() for CsvInterface data types.
        error_if_dir_not_exist: bool
                                If True, error if not exist.  If False, create dir if necessary.
    """
    output_body_ext = convert_string_to_file_body(output_file_body) + '.csv'
    output_dir_body_ext = join(output_dir, output_body_ext)

    # Check status of output_dir.
    if os.path.isfile(output_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In write_csv_file(), requested output path exists and is a file: ' + str(output_dir)
        )
    if error_if_dir_not_exist == True:
        if not directory_exists(output_dir):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In write_csv_file(), requested output directory does not exist: ' + str(output_dir),
            )
    else:
        create_directories_if_necessary(output_dir)

    # Check status of output file
    if file_exists(output_dir_body_ext):
        if not overwrite:
            lt.error_and_raise(
                FileExistsError, f"The destination file for {description} ('{output_dir_body_ext}') already exists!"
            )

    # Write output file.
    if description is not None:
        lt.info('Saving ' + description + ': ' + output_dir_body_ext)
    output_stream = open(output_dir_body_ext, 'w')
    # Write heading lines.
    if heading_line == None:
        pass
    elif heading_line == True:
        if len(data_lines) > 0 and isinstance(data_lines[0], csvi.CsvInterface):
            output_stream.write(data_lines[0].csv_header() + '\n')
        else:
            output_stream.write("empty_csv_header\n")
    else:
        output_stream.write(heading_line + '\n')
    # Write data rows.
    if len(data_lines) > 0 and isinstance(data_lines[0], csvi.CsvInterface):
        for data_line in data_lines:
            output_stream.write(data_line.to_csv_line() + '\n')
    else:
        for data_line in data_lines:
            output_stream.write(data_line + '\n')
    output_stream.close()
    # Return.
    return output_dir_body_ext


def read_csv_file(description, input_path, input_file_name, log_warning=True):
    """Deprecated. Use from_csv() instead, to better match naming in pandas."""
    if log_warning:
        lt.info("'read_csv_file' is deprecated in favor of 'from_csv'")
    return from_csv(description, input_path, input_file_name)


def from_csv(description: str | None, input_path: str, input_file_name_ext: str):
    """Reads a csv file and returns the rows, including the header row.

    Concise example::

        parser = scsv.SimpleCsv("example file", file_path, file_name_ext)
        for row_dict in parser:
            print(row_dict)

    Verbose example::

        lines = ft.from_csv("example file", file_path, file_name_ext)
        header_row = lines[0]
        cols = csv.CsvColumns.SimpleColumns(header_row)

        data_rows = lines[1:]
        for row in data_rows:
            row_dict = cols.parse_data_row(row)
            print(row_dict)

    Parameters:
    -----------
        description (str): printed to the log
        input_path (str): directory containing the file to read
        input_file_name_ext (str): name and extension of the file to read

    Returns:
    --------
        rows: list[list[str]]
              All the rows from the csv file, split on the comma ',' delimeter."""
    # In many cases this would be better to return as a Pandas data frame.  Pending future implementation.
    # However, this version works well with csv files where row lengths are irregular.
    # Consruct input file path and name.
    input_path_file = os.path.join(input_path, input_file_name_ext)
    if description is not None:
        lt.info('Reading ' + description + ': ' + input_path_file + ' ...')
    # Read csv file.
    data_rows: list[list[str]] = []
    with open(input_path_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            data_rows.append(row)
    return data_rows


# DICTIONARY FILES


# ?? SCAFFOLDING RCB -- THERE ARE SIMILAR ROUTINES IN dict_tools.py.  RESOLVE THIS INCONSISTENCY IN FILE PLACEMENT.
def write_dict_file(
    description,  # Explanatory string to include in notification output.  None to skip.
    output_dir,  # Directory to write file.  See below if not exist.
    output_body,  # Body of output filename; extension is ".csv"
    output_dict,  # Dictionary to write.
    decimal_places=9,  # Number of decimal places to write for floating-point values.
    error_if_dir_not_exist=True,
):  # If True, error if not exist.  If False, create dir if necessary.
    """
    Writes a dictionary to a ".csv" file, with a comma separating dictionary keys from values.
    Calls str(value) on all dictionary values except floats.  For floats, writes a decimal
    with the specified number of decimal places.
    """
    # Check status of output_dir.
    if os.path.isfile(output_dir):
        lt.error_and_raise(
            RuntimeError, 'ERROR: In write_dict_file(), requested output path exists and is a file: ' + str(output_dir)
        )
    if error_if_dir_not_exist == True:
        if not directory_exists(output_dir):
            lt.error_and_raise(
                RuntimeError,
                'ERROR: In write_dict_file(), requested output directory does not exist: ' + str(output_dir),
            )
    else:
        create_directories_if_necessary(output_dir)

    # Write output file.
    output_body_ext = convert_string_to_file_body(output_body) + '.csv'
    output_dir_body_ext = os.path.join(output_dir, output_body_ext)
    if description != None:
        print('Saving ' + description + ': ', output_dir_body_ext)
    output_stream = open(output_dir_body_ext, 'w')
    # Write dictionary data.
    for key in output_dict.keys():
        value = output_dict[key]
        if isinstance(value, float):
            format_str = '.' + str(decimal_places) + 'f'
            value_str = ('{0:' + format_str + '}').format(value)
        else:
            value_str = str(value)
        output_stream.write(str(key) + ',' + value_str + '\n')
    output_stream.close()
    # Return.
    return output_dir_body_ext


def add_row_to_output_dict(input_row: tuple[any, any], output_dict: dict):
    """Add the [key,value] to the given dictionary.

    If either key or value is a string that can be represented as an integer, then
    it is converted to an integer.

    If the value is a string that can be represented as an float, then
    it is converted to an float."""
    # Check input.
    if len(input_row) != 2:
        lt.error_and_raise(
            RuntimeError, 'ERROR: In add_row_to_output_dict(), input row is not of length 2: ', input_row
        )
    # Fetch key and value strings read from file.
    key_str = input_row[0]
    value_str = input_row[1]
    # Convert key string.
    if mt.string_is_integer(key_str):
        key = int(key_str)
    else:
        key = key_str
    # Convert value string.
    if mt.string_is_integer(value_str):
        value = int(value_str)
    elif mt.string_is_float(value_str):
        value = float(value_str)
    else:
        value = value_str
    # Add to dictionary.
    output_dict[key] = value


def read_dict(input_dict_dir_body_ext):
    """
    Reads a dictionary.
    Assumes input file exists, is a csv file, and each row has two entries.
    The first entry is the key, and the second entry is a value.
    If the key can be parsed as an integer, it is converted to an integer object.
    If the value can be parsed as a float or integer, it is converted to the corresponding
    type; otherwise, the value is added to the dict as a literal string.
    """
    # Check input.
    if not file_exists(input_dict_dir_body_ext):
        lt.error_and_raise(RuntimeError, 'ERROR: In read_dict(), file does not exist: ' + str(input_dict_dir_body_ext))
    input_dir, input_body, input_ext = path_components(input_dict_dir_body_ext)
    if input_ext.lower() != '.csv':
        lt.error_and_raise(
            RuntimeError, 'ERROR: In read_dict(), input file is not a csv file: ' + str(input_dict_dir_body_ext)
        )

    # Open and read the file.
    output_dict = {}
    with open(input_dict_dir_body_ext, newline='') as input_stream:
        reader = csv.reader(input_stream, delimiter=',')
        for input_row in reader:
            add_row_to_output_dict(input_row, output_dict)
    return output_dict


def write_json(
    description: str | None, output_dir: str, output_file_body: str, output_object: any, error_if_dir_not_exist=True
):
    """
    Like json.dump(output_object, output_file_body) but with a few more safety checks and automatic ".json" extension appending.

    Parameters
    ----------
    description : str | None
        A human-readable description of what this file is for, to be logged to the command line. If None, then no log is created.
    output_dir : str
        The destination directory for the file.
    output_file_body : str
        The destination name for the file. Should not include an extension. For example: "foo" is ok, but "foo.json" is not.
    output_object : any
        The object to be saved to the given file.
    error_if_dir_not_exist : bool, optional
        If True, then first check if the given output_dir exists. By default True.
    """
    # normalize input
    output_name_ext = output_file_body
    if not output_file_body.lower().endswith(".json"):
        output_name_ext = output_name_ext + ".json"
    output_path_name_ext = os.path.join(output_dir, output_name_ext)

    # validate input
    if error_if_dir_not_exist:
        if not directory_exists(output_dir):
            lt.error_and_raise(
                FileNotFoundError, "Error in file_tools.write_json(): " + f"the directory {output_dir} does not exist!"
            )
    if file_exists(output_path_name_ext):
        lt.error_and_raise(
            FileExistsError, "Error in file_tools.write_json(): " + f"the file {output_path_name_ext} already exists!"
        )

    # save the file
    if description != None:
        print('Saving ' + description + ': ', output_path_name_ext)
    with open(output_path_name_ext, "w") as fout:
        json.dump(output_object, fout)


def read_json(description: str | None, input_dir: str, input_file_body_ext: str) -> any:
    """
    Like json.loads(file_contents) but with more safety checks, and ignoring any lines starting with "//" as comments.

    Parameters
    ----------
    description : str | None
        A human-readable description of what this file is for, to be logged to the command line. If None, then doesn't log.
    input_dir : str
        The source directory where the file exists.
    input_file_body_ext : str
        The source name+ext of the file. For example "foo.json".

    Returns
    -------
    any
        The json-parsed contents of the file.
    """
    # TODO should we switch to https://pypi.org/project/pyjson5/? I'm not doing that now, because it would mean another
    # dependency, and this is good enough for now.

    # normalize input
    input_path_name_ext = os.path.join(input_dir, input_file_body_ext)

    # validate input
    if not file_exists(input_path_name_ext):
        lt.error_and_raise(
            FileNotFoundError, "Error in file_tools.read_json(): " + f"the file {input_path_name_ext} does not exist!"
        )

    # read the file
    if description is not None:
        lt.info('Reading ' + description + ': ' + input_path_name_ext + ' ...')
    with open(input_path_name_ext, 'r') as fin:
        lines = fin.readlines()
    lines = map(lambda l: "" if l.strip().startswith("//") else l, lines)
    return json.loads("\n".join(lines))


# PICKLE FILES

# def write_pickle_file(description,                   # Explanatory string to include in notification output.  None to skip.
#                       output_dir,                    # Directory to write file.  See below if not exist.
#                       output_file_body,              # Body of output filename; extension is ".csv"
#                       output_object,                 # Object to write; must be able to be pickled (see docs).
#                       error_if_dir_not_exist=True):  # If True, error if not exist.  If False, create dir if necessary.
#     """
#     Writes a Python object to a ".pkl" file.
#     For background, see https://wiki.python.org/moin/UsingPickle.
#     """
#     # Check status of output_dir.
#     if os.path.isfile(output_dir):
#         print('ERROR: In write_pickle_file(), requested output path exists and is a file: ' + str(output_dir))
#         assert False
#     if error_if_dir_not_exist == True:
#         if not directory_exists(output_dir):
#             print('ERROR: In write_pickle_file(), requested output directory does not exist: ' + str(output_dir))
#             assert False
#     else:
#         create_directories_if_necessary(output_dir)
#     # Write output file.
#     output_body_ext = convert_string_to_file_body(output_file_body) + '.pkl'
#     output_dir_body_ext = os.path.join(output_dir, output_body_ext)
#     if description != None:
#         print('Saving ' + description + ': ', output_dir_body_ext)
#     output_stream = open(output_dir_body_ext, 'wb')
#     # Write pickle.
#     pickle.dump(output_object, output_stream)
#     output_stream.close()
#     # Return.
#     return output_dir_body_ext


# def read_pickle_file(input_dir_body_ext):
#     """
#     Reads a pickle file.
#     Assumes input file exists, and is a pickle file.
#     File extension may be arbitrary.
#     Returns the pickled Python object.
#     """
#     # Check input.
#     if not file_exists(input_dir_body_ext):
#         print('ERROR: In read_pickle_file(), file does not exist: ' + str(input_dir_body_ext))
#         assert False
#     # Open and read the file.
#     with open(input_dir_body_ext, 'rb') as input_stream:
#         object = pickle.load(input_stream)
#     return object


if __name__ == '__main__':
    print("directories_with_no_leading_underscore('.') = ", directories_with_no_leading_underscore('.'))
