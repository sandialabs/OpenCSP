import os
import random
import time

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.process.lib.ServerSynchronizerError as sse
import opencsp.common.lib.process.parallel_file_tools as pft
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class ServerSynchronizer:
    """
    Helper class to force all servers to wait at specified synchronization points.
    This is particularly useful for scatter-gather type workflows.
    """

    path = os.path.join(orp.opencsp_temporary_dir(), "synchronize_servers_by_file")

    def __init__(
        self, num_servers: int, server_index: int, propagate_errors=True, timeout: int = 1000, do_initial_wait=True
    ):
        """Helper class to forces all servers to wait at specified synchronization points.
        This is particularly useful for scatter-gather type workflows.

        At each call to wait(), including during this constructor if
        do_initial_wait, all servers will stop execution until all
        other servers have reached the same wait call. The same is true
        for the stop() call at the end of program execution.

        Signaling between servers is achieved through the shared network
        file system that all of the servers share. In particular, the
        opencsp_temporary_dir() is utilized for this purpose. Because of
        the nature of multiprocessing and networked file systems, there
        is a decent chance that there are bugs in this implementation.
        If you find any bugs, please let me (BGB) know!

        Ideas for future improvement:
         - Use network communication as the signaling mechanism instead
           of a file system.
         - Provide a unique and/or random identifier when starting the
           execution, so that independent execution (by independent
           programmers) can be differentiated.

        Example usage::

            # num_servers = 8
            # server_index = 0-7
            synchronizer = ss.ServerSynchronizer(num_servers, server_index)
            try:
                # some work
                synchronizer.wait()
                # more work
                synchronizer.wait()
                ...
                # final work
            except Exception as ex:
                synchronizer.stop(ex)
            finally:
                synchronizer.stop()

        Args:
            num_servers (int): How many servers this code is running across
            server_index (int): Which server this is (indexing starts at 0)
            propagate_errors (bool): Whether to re-raise errors encountered in other servers during the wait method.
            timeout (int): Maximum amount of time to wait, in seconds. Default 1000.
        """
        self.num_servers = num_servers
        self.server_index = server_index
        self.propagate_errors = propagate_errors
        self.timeout = timeout
        self._synchronization_index = 0
        self._stopped = False

        # Clean out any existing wait, stop, error, and value files
        ft.create_directories_if_necessary(self.__class__.path)
        try:
            ft.delete_files_in_directory(self.__class__.path, "*wait*")
            if server_index == 0:
                ft.delete_files_in_directory(self.__class__.path, "*value*.txt")
        except FileNotFoundError:
            # Probably another server deleted a file we were also trying to delete.
            pass
        if server_index == 0:
            self._remove_all_stop_files()
        else:
            time.sleep(5)  # have all other servers wait for server 0 to remove the stop and error files

        # Let the system know that this server is executing
        lt.info(f"ServerSynchronizer @{self.server_index} started")

        # Wait here for all other servers to have started
        if do_initial_wait and num_servers > 1:
            self._wait(check_for_stopped_servers=False)

    def _wait_on_files(
        self, wait_file_path_name_exts: list[str], stop_file_path_name_exts: list[str] = None, msg: str = None
    ):
        """Wait for all of the given "wait" indicator files (or their corresponding "stop" indicator files) to exist."""
        alternates: dict[str, list[str]] = {}
        if stop_file_path_name_exts != None:
            for i in range(len(wait_file_path_name_exts)):
                alternates[wait_file_path_name_exts[i]] = [stop_file_path_name_exts[i]]
        pft.wait_on_files(wait_file_path_name_exts, self.timeout, alternates=alternates, msg=msg)
        lt.debug(f"Server @{self.server_index}: found all files")
        # wait for all servers to see that the files exist
        time.sleep(5)

    @classmethod
    def _get_file_stopped(cls, other_server_index: int):
        """Returns the path_name_ext of the "stopped" indicator file.

        Parameters:
        -----------
            - other_server_index (int): The index of the server to get the indicator file for.
        """
        return f"{cls.path}/stopped_{other_server_index}"

    @classmethod
    def _get_file_error(cls, other_server_index: int):
        """Returns the path_name_ext of the "error" indicator file.

        Parameters:
        -----------
            - other_server_index (int): The index of the server to get the indicator file for.
        """
        return f"{cls.path}/error_{other_server_index}"

    @classmethod
    def _get_file_waiting(cls, other_server_index: int, synchronization_index: int):
        """Returns the path_name_ext of the "wait" indicator file.

        Parameters:
        -----------
            - other_server_index (int): The index of the server to get the indicator file for.
            - synchronization_index (int): Essentially the step number for which wait() call to wait for.
        """
        if synchronization_index == 0:
            # special case for the first index 0, so that when we call ft.delete_files()/wait() in __init__(), we don't accidentally delete id 0 files
            return f"{cls.path}/startup_{synchronization_index}_{other_server_index}"
        return f"{cls.path}/wait_{synchronization_index}_{other_server_index}"

    @classmethod
    def _get_file_value(cls, other_server_index: int, synchronization_index: int):
        """Returns the path_name_ext of the "value" communication file.

        Parameters:
        -----------
            - other_server_index (int): The index of the server to get the indicator file for.
            - synchronization_index (int): Essentially the step number for which wait() call to wait for.
        """
        return f"{cls.path}/value_{synchronization_index}_{other_server_index}.txt"

    def get_stopped_servers(self):
        """Get a list of servers that have stopped (aka have "stopped" indicator files).

        Returns:
        --------
            server_idxs (list[int]): List of all the stopped server indexes."""
        ret: list[int] = []
        all_file_path_name_exts = [(i, self._get_file_stopped(i)) for i in range(self.num_servers)]
        for other_server_index, file_path_name_ext in all_file_path_name_exts:
            if ft.file_exists(file_path_name_ext):
                ret.append(other_server_index)
        return ret

    def get_errored_servers(self):
        """Get a list of servers that have errored (aka have "errored" indicator files).

        Returns:
        --------
            ret (list[int]): List of all the errored server indexes."""
        ret: list[tuple(int, str, str)] = []
        all_file_path_name_exts = [(i, self._get_file_error(i)) for i in range(self.num_servers)]
        for other_server_index, file_path_name_ext in all_file_path_name_exts:
            if ft.file_exists(file_path_name_ext):
                try:
                    lines = ft.read_text_file(file_path_name_ext)
                    err_type, err_msg = lines[0], "\n".join(lines[1:])
                except:
                    err_type, err_msg = "unknown error", "N/A"
                ret.append(tuple([other_server_index, err_type, err_msg]))
        return ret

    def _check_for_other_server_errors(self, method_name):
        """Get a list of all servers that have halted due to an error.

        Returns:
        --------
            error_msg (str|None): None if no errored servers, or a message indicating the type of error for the first errored server.
        """
        errored_servers = self.get_errored_servers()
        errored_servers = list(filter(lambda es: es[0] != self.server_index, errored_servers))
        if len(errored_servers) > 0:
            errored_server, err_type, err_msg = errored_servers[0]
            return f'Error: in ServerSynchronizer.{method_name}(), server {errored_server} encountered a {err_type} with the message "{err_msg}"'
        return None

    def wait(self):
        """Wait for all servers to reach this point."""
        self._wait()

    def _wait(self, check_for_stopped_servers=True):
        """Wait for all servers to reach this point.

        This method uses the "wait" indicator files, as named by the _get_file_waiting() method,
        to wait for all other servers to reach this point. The steps to waiting here are:
            - (1) create this server's wait indicator file
            - (2) look for stopped servers to exclude from the wait check (if check_for_stopped_servers)
            - (3) wait for any still-running server to generate their own wait indicator files
            - (4) look for errored servers, raising a ServerSynchronizerError if there are any
            - (5) increment the wait id, to get ready for the next wait
            - (5) remove this server's wait indicator file"""
        if self.num_servers <= 1:
            return

        # create my file (1)
        my_file_path_name_ext = self._get_file_waiting(self.server_index, self._synchronization_index)
        if not ft.file_exists(my_file_path_name_ext):
            ft.create_file(my_file_path_name_ext)
        else:
            lt.warn(
                f"Warning: in ServerSynchronizer.wait(), current synchronization file {my_file_path_name_ext} "
                + "should not exist yet! This probably indicates a bug in the flow of your synchronized server code!"
            )

        try:
            # Getting ready for the _next_ wait():
            # Make sure the next synchronization files don't exist.
            for i in range(self.num_servers):
                next_file_path_name_ext = self._get_file_waiting(i, self._synchronization_index + 1)
                if ft.file_exists(next_file_path_name_ext):
                    lt.warn(
                        f"Warning: in ServerSynchronizer.wait(), next synchronization file {next_file_path_name_ext} "
                        + "should not exist yet! This probably indicates a bug in the flow of your synchronized server code!"
                    )
                    ft.delete_file(next_file_path_name_ext)

            # wait for all servers (2,3)
            stopped_idxs = []
            stop_file_path_name_exts = None
            if check_for_stopped_servers:
                stopped_idxs = self.get_stopped_servers()
            running_idxs = list(filter(lambda i: i not in stopped_idxs, range(self.num_servers)))
            wait_file_path_name_exts = [self._get_file_waiting(i, self._synchronization_index) for i in running_idxs]
            if check_for_stopped_servers:
                stop_file_path_name_exts = [self._get_file_stopped(i) for i in running_idxs]
            self._wait_on_files(
                wait_file_path_name_exts, stop_file_path_name_exts, msg=f"step {self._synchronization_index}"
            )

            # propagate errors (4)
            err_msg = self._check_for_other_server_errors("wait")
            if err_msg != None and self.propagate_errors:
                lt.error_and_raise(sse.ServerSynchronizerError, err_msg)

        finally:
            # increment my synchronization id (5)
            self._synchronization_index += 1

            # delete my file (6)
            ft.delete_file(my_file_path_name_ext)

    def gather(self, value: str):
        """All servers write the given value to a file and wait. Then
        read all the values from all the servers (in server index order) as the
        return from this function.

        In order for this (or really any ServerSynchronization method) to
        work, all server must call the same method at the same relative
        point in execution.

        Parameters:
        -----------
            value (str): The value this server shares with all other servers.

        Returns:
        --------
            sum (list[str]): All server values, in order.
        """
        value_sync_index = self._synchronization_index
        my_file_path_name_ext = self._get_file_value(self.server_index, value_sync_index)
        my_file_path_name_ext_tmp = my_file_path_name_ext + ".tmp"

        # remove the stale file, if any
        if ft.file_exists(my_file_path_name_ext):
            lt.warn(
                f"Warning: in ServerSynchronizer.gather(), value file {my_file_path_name_ext} "
                + "should not exist yet! This probably indicates a bug in the flow of your synchronized server code!"
            )
            ft.delete_file(my_file_path_name_ext)

        # write my contents to the file
        if ft.file_exists(my_file_path_name_ext_tmp):
            ft.delete_file(my_file_path_name_ext_tmp)
        try:
            with open(my_file_path_name_ext_tmp, "w") as fout:
                fout.write(value)
            ft.rename_file(my_file_path_name_ext_tmp, my_file_path_name_ext)

        except Exception as ex:
            if ft.file_exists(my_file_path_name_ext):
                ft.delete_file(my_file_path_name_ext)
            raise

        finally:
            if ft.file_exists(my_file_path_name_ext_tmp):
                ft.delete_file(my_file_path_name_ext_tmp)

        # wait for all the other servers
        self.wait()
        if self._synchronization_index == value_sync_index:
            if self.num_servers > 1:
                lt.warn("Huh, I expected the _synchronization_index to have incremented...")
            self._synchronization_index += 1

        # gather the results
        ret = []
        for other_idx in range(self.num_servers):
            other_file_path_name_ext = self._get_file_value(other_idx, value_sync_index)
            if ft.file_exists(other_file_path_name_ext):
                with open(other_file_path_name_ext, "r") as fin:
                    ret.append(fin.read())
            else:
                lt.warn(f"Warning: in ServerSynchronizer.gather(), value file {other_file_path_name_ext} is missing!")
        return ret

    def _remove_all_stop_files(self):
        """remove existing "stopped" and "errored" files"""
        for i in range(self.num_servers):
            stop_file_path_name_ext = self._get_file_stopped(i)
            error_file_path_name_ext = self._get_file_error(i)
            for file_path_name_ext in [stop_file_path_name_ext, error_file_path_name_ext]:
                if ft.file_exists(file_path_name_ext):
                    try:
                        os.remove(file_path_name_ext)
                    except Exception as ex:
                        if isinstance(ex, FileNotFoundError) or isinstance(ex, PermissionError):
                            # Probably just attempted to delete a file at the same time as another server.
                            # Randomly backoff to reduce the likelihood of this happening again.
                            time.sleep(random.randint(1, 10) / 10)

    def stop(self, error_to_propagate: Exception = None):
        """Create the wait and stop signal files, wait for other servers to stop, and remove the wait and stop files.

        This works in essentially the same way as the wait() method, except that it is waiting on "stopped" files instead of "wait" files.
        """
        if self.num_servers <= 1:
            return
        if self._stopped:
            return
        self._stopped = True
        lt.info(f"ServerSynchronizer @{self.server_index} stopped")

        # create my "stopped" file
        my_file_path_name_ext = self._get_file_stopped(self.server_index)
        if not ft.file_exists(my_file_path_name_ext):
            ft.create_file(my_file_path_name_ext)
        else:
            lt.warn(
                f"Warning: in ServerSynchronizer.stop(), stop synchronization file {my_file_path_name_ext} "
                + "should not exist yet! This probably indicates a bug in the flow of your synchronized server code!"
            )

        # create my "error" file, if any
        if error_to_propagate != None:
            if not isinstance(error_to_propagate, sse.ServerSynchronizerError):
                my_file_path_name_ext = self._get_file_error(self.server_index)
                if not ft.file_exists(my_file_path_name_ext):
                    with open(my_file_path_name_ext, "w") as err_file:
                        err_file.write(error_to_propagate.__class__.__name__ + "\n")
                        err_file.write(str(error_to_propagate))
                else:
                    lt.warn(
                        f"Warning: in ServerSynchronizer.stop(), error synchronization file {my_file_path_name_ext} "
                        + "should not exist yet! This probably indicates a bug in the flow of your synchronized server code!"
                    )

        # wait for all other servers to stop
        all_stop_files = [self._get_file_stopped(i) for i in range(self.num_servers)]
        self._wait_on_files(all_stop_files, None, msg="end step")

        # Delete the stop files.
        # Note: if self.wait_on_files(all_stop_files) times out, then an
        # exception will be thrown and this code will never be reached.
        # This is intended behavior, since we want the stop file to still
        # be here by the time that the other servers are ready to stop.
        err_msg = self._check_for_other_server_errors("stop")
        self._remove_all_stop_files()

        # propagate error messages
        if err_msg != None and self.propagate_errors:
            if (
                error_to_propagate != None
                and isinstance(error_to_propagate, sse.ServerSynchronizerError)
                and str(error_to_propagate).replace("wait()", "stop()") == err_msg
            ):
                """Avoid double-printing and error encountered first in wait() and now in stop()."""
                pass
            else:
                lt.error_and_raise(sse.ServerSynchronizerError, err_msg)

        # raise the error
        if error_to_propagate != None:
            if not isinstance(error_to_propagate, sse.ServerSynchronizerError):
                raise error_to_propagate
