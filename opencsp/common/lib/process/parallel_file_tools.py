import copy
import time

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

def _file_or_alternate_existing_file(path_name_ext: str, alternates: list[str]):
    if ft.file_exists(path_name_ext):
        return path_name_ext
    else:
        for alternate in alternates:
            if ft.file_exists(alternate):
                return alternate
    return None

def wait_on_files(files: list[str], timeout: float=1000, alternates: dict[str,list[str]]=None, msg: str=None):
    """ Waits up to 'timeout' seconds for all the files to exist.

    Note: there is no guarantee that all files exist when this function completes.
    Rather, this function guarantees that all of the files existed at some point
    during its execution (ie some of the files could have been deleted after
    there existance was verified).
    
    Arguments:
    ----------
        - files (list[str]): The files to check for the existance of.
        - timeout (float): How many seconds to wait for the given files to exist.
        - alternates (dict[str,list[str]]): Alternate files that can exist in the
            place of the given files. An example use case could be to check if an
            error file was created from a process that we're waiting on the
            output from.
    
    Returns:
    --------
        - A list of the found files.
    
    Raises:
    -------
        - TimeoutError: if the timeout triggers and not all files exist yet.
    """
    files = copy.deepcopy(files)
    waiting = copy.deepcopy(files)
    alternates = copy.deepcopy(alternates)
    found_files: dict[str,str] = {}
    has_printed_wait = False
    msg = "" if msg == None else " ("+msg+")"

    # populate alternates, as necessary
    if alternates == None:
        alternates = {}
    for file in files:
        if file not in alternates:
            alternates[file] = []

    # wait for all files to exist
    tstart, tend = time.time(), time.time()
    while tend - tstart < timeout: # wait for up to timeout seconds
        new_waiting = []
        for f in waiting:
            found_path_name_ext = _file_or_alternate_existing_file(f, alternates[f])
            if found_path_name_ext == None:
                new_waiting.append(f)
            else:
                found_files[f] = found_path_name_ext
        waiting = new_waiting
        
        if not has_printed_wait:
            lt.info(f"Waiting on {len(waiting)}/{len(files)} files{msg}...")
            has_printed_wait = True
        
        if len(waiting) == 0:
            break
        
        tend = time.time()
        tsleep = min(1, timeout-(tend-tstart))
        if tsleep > 0:
            time.sleep(tsleep)
            tend += tsleep
    
    # check, did all servers finish?
    if len(waiting) > 0:
        # timed out, check one more time
        new_waiting = []
        for f in waiting:
            found_path_name_ext = _file_or_alternate_existing_file(f, alternates[f])
            if found_path_name_ext == None:
                new_waiting.append(f)
            else:
                found_files[f] = found_path_name_ext
        waiting = new_waiting

        # still waiting, throw error
        if len(waiting) > 0:
            missing_files_str = "\n\tMissing files:\n\t" + ", ".join(waiting)
            if tend - tstart > timeout:
                lt.error_and_raise(TimeoutError, f"Could not find {len(waiting)}/{len(files)} server files{msg}. Timed out at {tend-tstart:0.1f} seconds.{missing_files_str}")
            lt.error_and_raise(TimeoutError, f"Could not find {len(waiting)}/{len(files)} server files{msg}. Probably not all servers finished their work.{missing_files_str}")

    # return the list of found files
    ret = [found_files[f] for f in files]
    return ret