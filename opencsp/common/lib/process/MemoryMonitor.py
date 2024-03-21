import concurrent.futures
import datetime
import multiprocessing
from multiprocessing.synchronize import Event as multiprocessing_event_type
import time

import opencsp.common.lib.process.ParallelPartitioner as ppart
import opencsp.common.lib.tool.system_tools as st
import opencsp.common.lib.tool.log_tools as lt


class MemoryMonitor:
    def __init__(
        self,
        server_index=0,
        cpu_index=0,
        max_lifetime_hours=2.0,
        print_threshold=0.95,
        print_on_new_min=True,
        always_print=False,
    ):
        """Monitors total system memory usage in a separate subprocess while the rest of the application continues running.

        Example usage::

            monitor = mm.MemoryMonitor()
            try:
                monitor.start()
                # do work
                monitor.stop(wait=True)
                print(monitor.max_usage() - monitor.min_usage())
            finally:
                monitor.stop()

        Args:
            - server_index (int): Used to identify this monitor when running multiple tasks on the same system.
            - cpu_index (int): Used to identify this monitor when running multiple tasks on the same system.
            - max_lifetime_hours (float): Maximum lifetime for the monitor subprocess. Setting this guarantees that the subprocess will at some point stop. Defaults to 2 hours.
            - print_threshold (float): If the memory used/total exceeds this threshold, then print out the usage every second. Defaults to 0.95.
            - print_on_new_min (bool): If true, then for each second, if the memory available has reached a new minimum print out the usage for that second. Defaults to True.
            - always_print (bool): If True, then print out the usage every second. Defaults to False.
        """
        partitioner = ppart.ParallelPartitioner(server_index + 1, server_index, cpu_index + 1, cpu_index)
        self.identifier = partitioner.identifier()
        """ String used to uniquely identify this server/processor core. """
        self._proc: multiprocessing.Process = None
        """ The process that monitors memory """
        self._future: concurrent.futures.Future = None
        """ The thread that is used to start and monitor self._proc. """
        self._stop_sig = multiprocessing.Event()
        """ Assert this to indicate that the sub process should stop. """
        self._log: dict[datetime.datetime, tuple] = {}
        self._max_lifetime_hours = max_lifetime_hours
        """ Maximum amount of time that the monitor will run for. """
        self._max_lifetime_seconds = int(self._max_lifetime_hours * 60 * 60)
        self._print_threshold = print_threshold
        self._print_on_new_min = print_on_new_min
        self._always_print = always_print
        self._start_datetime: datetime.datetime = None
        self._end_datetime: datetime.datetime = None
        self._process_finished = False
        self._min_max_avg_usage = [10e10, 0, 0]
        self._min_max_avg_free = [10e10, 0, 0]

    def start(self):
        """Starts this monitor. Returns True if started, or False if it has been started previously."""
        # check if already running
        if self._start_datetime != None:
            return False

        # start the process
        executor = concurrent.futures.ThreadPoolExecutor(1, "mem_monitor_" + self.identifier)
        self._future = executor.submit(self._run)
        while self._start_datetime == None:
            time.sleep(0.1)
        while self._end_datetime == None:
            if self._proc != None and self._proc.is_alive():
                break
            time.sleep(0.1)

        return True

    def stop(self, wait=False):
        """Attempts to stop the monitoring subprocess.

        Args:
            wait (bool, optional): If True, then wait some time and for the subprocess to stop. Defaults to False.

        Returns:
            stopped (bool): If wait, then return the stopped status of
                the subprocess. Otherwise, return True if the stop signal was
                successfully sent.
        """
        if self._start_datetime == None:
            return False

        self._stop_sig.set()

        # Wait until we know if the monitor process has exited?
        if wait:
            start = time.time()
            while not self._process_finished:
                if time.time() - start >= 1.2:
                    return False
                time.sleep(0.1)
            while self._end_datetime == None:
                time.sleep(0.1)

        return True

    def done(self):
        """Returns true when the return value of the monitoring subprocess has stopped."""
        return self._end_datetime != None

    def _run(self):
        self._stop_sig.clear()
        self._start_datetime = datetime.datetime.now()
        self._end_datetime = None
        start_time = time.time()

        queue = multiprocessing.Queue()
        self._proc = multiprocessing.Process(
            target=_monitor_sys_memory,
            args=[
                self._max_lifetime_seconds,
                self._stop_sig,
                queue,
                self._print_threshold,
                self._print_on_new_min,
                self._always_print,
            ],
        )
        self._proc.start()
        while self._proc.is_alive():
            # print("-", end="")
            elapsed = time.time() - start_time
            if elapsed > self._max_lifetime_seconds + 3:
                self._proc.terminate()
            if self._stop_sig.is_set():
                break
            time.sleep(0.1)
        self._process_finished = True

        while not queue.empty():
            elapsed, sys_tot, sys_used, sys_free = queue.get()
            dt = self._start_datetime + datetime.timedelta(seconds=elapsed)
            self._log[dt] = sys_tot, sys_used, sys_free
            self._min_max_avg_usage[0] = min(self._min_max_avg_usage[0], sys_used)
            self._min_max_avg_usage[1] = max(self._min_max_avg_usage[1], sys_used)
            self._min_max_avg_usage[2] += sys_used
            self._min_max_avg_free[0] = min(self._min_max_avg_free[0], sys_free)
            self._min_max_avg_free[1] = max(self._min_max_avg_free[1], sys_free)
            self._min_max_avg_free[2] += sys_free

        if len(self._log) > 0:
            self._min_max_avg_usage[2] /= len(self._log)
            self._min_max_avg_free[2] /= len(self._log)
        self._end_datetime = datetime.datetime.now()

    def min_usage(self):
        """Returns the minimum memory usage while the monitor was running, in GB."""
        if not self.done():
            return None
        return self._min_max_avg_usage[0]

    def max_usage(self):
        """Returns the maximum memory usage while the monitor was running, in GB."""
        if not self.done():
            return None
        return self._min_max_avg_usage[1]

    def avg_usage(self):
        """Returns the average memory usage while the monitor was running, in GB."""
        if not self.done():
            return None
        return self._min_max_avg_usage[2]

    def min_free(self):
        """Returns the minimum memory free while the monitor was running, in GB."""
        if not self.done():
            return None
        return self._min_max_avg_free[0]

    def max_free(self):
        """Returns the maximum memory free while the monitor was running, in GB."""
        if not self.done():
            return None
        return self._min_max_avg_free[1]

    def avg_free(self):
        """Returns the average memory free while the monitor was running, in GB."""
        if not self.done():
            return None
        return self._min_max_avg_free[2]


def _monitor_sys_memory(
    max_lifetime_secs: int,
    stop_sig: multiprocessing_event_type,
    queue: multiprocessing.Queue,
    print_threshold: float,
    print_on_new_min: bool,
    always_print: bool,
):
    start = time.time()
    # print("monitor started")

    prev_min = 10e10
    while (elapsed := time.time() - start) < max_lifetime_secs:
        sys_tot, sys_used, sys_free = st.mem_status()

        sys_avail_for_min = float("%0.1f" % sys_free)
        is_new_min = False
        if sys_avail_for_min < prev_min:
            prev_min = sys_avail_for_min
            is_new_min = True
        do_print_min = is_new_min and print_on_new_min

        used_ratio = sys_used / sys_tot
        do_print_threshold = used_ratio > print_threshold

        queue.put(tuple([elapsed, sys_tot, sys_used, sys_free]))
        if do_print_min or do_print_threshold or always_print:
            lt.info(
                "MM: %s %s %s %s"
                % (
                    f"{int(elapsed):>3d}",
                    f"{sys_tot:0.1f}".rjust(5),
                    f"{sys_used:0.1f}".rjust(5),
                    f"{sys_free:0.1f}".rjust(5),
                )
            )

        if stop_sig.wait(1):
            break

    # print("monitor stopped")
