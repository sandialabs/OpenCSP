import io
import os
import subprocess
import time

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.process.lib.ProcessOutputLine as pol
import opencsp.common.lib.process.lib.CalledProcessError as cpe


def get_executable_path(executable_name: str, dont_match: str = None) -> str:
    """Returns the first "path/name.ext" for the given executable. If
    dont_match is specified, then paths containing that string are excluded from
    the returned results.

    Parameters
    ----------
    executable_name : str
        The name of the executable to search for. On windows, either the plain
        name or the .exe name may be searched for equally.
    dont_match : str, optional
        If specified, then don't include results that contain this string. By
        default None.

    Returns
    -------
    executable_path_name_ext: str
        The "path/name.exe" of the found executable.
    """
    dont_match = dont_match.lower()
    search_cmd = "which"
    if os.name == "nt":
        search_cmd = "where"
        if executable_name.endswith(".exe"):
            executable_name = executable_name[:-4]

    lines = run(f"{search_cmd} {executable_name}", stdout="collect", stderr="collect")
    lines = filter_lines(lines, keep_stderr=False)
    for line in lines:
        if (dont_match == None) or (dont_match not in line.val.lower()):
            return line.val.strip()

    return executable_name


def filter_lines(lines: list[pol.ProcessOutputLine], keep_stdout=True, keep_stderr=True):
    """
    Filters a list of process output lines based on specified criteria.

    This function allows you to filter out standard output (stdout) or standard error (stderr) lines
    from a list of process output lines. You can choose to keep only the stdout lines, only the stderr
    lines, or both.

    Parameters
    ----------
    lines : list[pol.ProcessOutputLine]
        A list of process output lines to filter.
    keep_stdout : bool, optional
        If True, keeps the stdout lines. If False, removes them. Defaults to True.
    keep_stderr : bool, optional
        If True, keeps the stderr lines. If False, removes them. Defaults to True.

    Returns
    -------
    list[pol.ProcessOutputLine]
        A list of filtered process output lines based on the specified criteria.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    ret: list[pol.ProcessOutputLine] = list(lines)

    if not keep_stdout:
        ret = filter(lambda line: line.is_err, ret)
    if not keep_stderr:
        ret = filter(lambda line: not line.is_err, ret)

    if not isinstance(ret, list):
        ret = list(ret)
    return ret


def print_lines(lines: list[pol.ProcessOutputLine]):
    """
    Prints the process output lines to the console.

    This function iterates through a list of process output lines and prints each line to the console.
    If the line is an error line, it uses the error logging function; otherwise, it uses the info logging function.

    Parameters
    ----------
    lines : list[pol.ProcessOutputLine]
        A list of process output lines to print.

    Returns
    -------
    None
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    for line in lines:
        if line.is_err:
            lt.error(line.val)
        else:
            lt.info(line.val)


def _collect_lines(
    lines: list[pol.ProcessOutputLine], curr_lineno: int, buffer: str, is_error=False, collect_last=False
):
    lineno = curr_lineno

    if "\n" in buffer:
        buffer_lines = buffer.split("\n")
        for buffer_line in buffer_lines[:-1]:
            lines.append(pol.ProcessOutputLine(buffer_line, is_error, lineno))
            lineno += 1
        if collect_last:
            if len(buffer_lines) > 1 and buffer_lines[-1] != "":
                lines.append(pol.ProcessOutputLine(buffer_lines[-1], is_error, lineno))
                lineno += 1
            return "", lineno
        else:
            return buffer_lines[-1], lineno
    else:
        if collect_last:
            if buffer != "":
                lines.append(pol.ProcessOutputLine(buffer, is_error, lineno))
                lineno += 1
            return "", lineno
        else:
            return buffer, lineno


def _is_process_running(proc: subprocess.Popen):
    return proc.poll() == None


def _is_timed_out(proc: subprocess.Popen, start: float, timeout: float | None):
    if timeout == None:
        # no timeout set, not timed out
        return False

    if _is_process_running(proc):  # process is still running
        # check the timeout
        if time.time() - start < timeout:
            return False

        # process has timed out
        proc.terminate()
        time.sleep(0.1)
        if _is_process_running(proc):
            proc.kill()
        return True
    else:  # process has finished
        return False


def run(
    cmd: str, cwd: str = None, stdout: str = None, stderr: str = None, ignore_return_code=False, timeout: float = None
):
    """
    Runs the given command in the given directory, prints the output to the logger, and checks the return code.

    This method is most useful for running a sub-process and capturing mixed stderr/stdout.
    For all other use cases, it is probably better to use proc=multiprocessing.Process(), proc.start(), proc.join().

    Args:
    -----
        cwd (str): The directory to change to before starting the subprocess.
        stdout (str): One of "collect" which returns all the stdout lines, "print", or "collect+print". Default is None (collect+print).
        stderr (str): One of "collect" which returns all the stderr lines, "print", or "collect+print". Default is None (collect+print).
        ignore_return_code (bool): If true, then don't raise an error for a non-zero return code.
        timeout (float): If not None, then terminate/kill the process after timeout seconds. If terminated, there might not be any stdout/stderr to collect. None for no timeout. Default is None.

    Raises:
    -------
        subprocess.CalledProcessError: Raised if the subprocess returns an error code.

    Returns:
    --------
        list[ProcessOutputLine]: The lines the subprocess output. On windows, the lines are collected with stdout first, then stderr.
    """
    if cwd == None:
        cwd = os.getcwd()
    if stdout == None:
        stdout = "collect+print"
    if stderr == None:
        stderr = "collect+print"
    print_stdout = "print" in stdout
    print_stderr = "print" in stderr
    collect_stdout = "collect" in stdout
    collect_stderr = "collect" in stderr

    lines: list[pol.ProcessOutputLine] = []
    lineno = 0

    if cwd != os.getcwd():
        lt.info("changing directory to " + cwd)
    lt.info("starting " + cmd)
    text_mode = True  # converts stdout/stderr to strings instead of bytes
    proc = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text_mode)
    start_time = time.time()

    new_lines: list[pol.ProcessOutputLine] = []
    if os.name == "nt":
        # on windows, proc.stdout.read(1) hangs, use communicate instead
        outbuf, errbuf = proc.communicate(timeout=timeout)
        # kill the process if timed out
        _is_timed_out(proc, start_time, timeout)
    else:
        # read stdout and stderr non-blocking
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)

        # wait for the process to finish, and collect any output from stdout/stderr
        outbuf = ""
        errbuf = ""
        is_timed_out = False
        while _is_process_running(proc):
            new_out, new_err = proc.stdout.read(1), proc.stderr.read(1)

            # collect the read stdout/stderr bytes into lines
            if new_out == "" and new_err == "":
                # no new lines at the moment, take this time to filter & print existing lines
                print_lines(filter_lines(new_lines, keep_stdout=print_stdout, keep_stderr=print_stderr))
                lines += filter_lines(new_lines, keep_stdout=collect_stdout, keep_stderr=collect_stderr)
                new_lines = []

                # check timeout
                if _is_timed_out(proc, start_time, timeout):
                    break

                # sleep while waiting for some stdout/stderr from the process
                time.sleep(0.1)
            else:
                # collect the read stdout/stderr bytes
                outbuf, lineno = _collect_lines(new_lines, lineno, outbuf + new_out, is_error=False)
                errbuf, lineno = _collect_lines(new_lines, lineno, errbuf + new_err, is_error=True)

        # finish collecting from stdout and stderr
        outbuf += proc.stdout.read()
        errbuf += proc.stderr.read()

    # parse any unparsed output
    outbuf, lineno = _collect_lines(new_lines, lineno, outbuf, is_error=False, collect_last=True)
    errbuf, lineno = _collect_lines(new_lines, lineno, errbuf, is_error=True, collect_last=True)
    proc.stdout.close()
    proc.stderr.close()

    # print any newly collected lines
    print_lines(filter_lines(new_lines, keep_stdout=print_stdout, keep_stderr=print_stderr))
    lines += filter_lines(new_lines, keep_stdout=collect_stdout, keep_stderr=collect_stderr)

    # throw an error code if the subprocess failed, or return the unprinted lines otherwise
    if proc.returncode != 0 and not ignore_return_code:
        stdout_str = None
        stderr_str = None
        try:
            stdout_str = "\n".join([line.val for line in filter_lines(lines, True, False)])
            stderr_str = "\n".join([line.val for line in filter_lines(lines, False, True)])
        except:
            pass
        raise cpe.CalledProcessError(proc.returncode, proc.args, stdout_str, stderr_str)
    return lines
