import multiprocessing
import time
from typing import Callable, Iterable


class MultiprocessNonDaemonic:
    """
    A class for managing a pool of non-daemonic processes for parallel execution.

    This class is similar to `multiprocessing.Pool`, but it allows for the creation of non-daemonic
    processes, which can spawn child processes. This is useful in scenarios where grandchild processes
    need to be created from the child processes.
    """

    # "ChatGPT 4o" assisted with generating this docstring.
    def __init__(self, num_processes: int):
        """Initializes the MultiprocessNonDaemonic instance.

        Parameters
        ----------
        num_processes : int
            The number of processes in this pool.

        Notes
        -----
        Daemonic processes have certain limitations, such as not being able to create child processes.
        This class allows for the creation of non-daemonic processes to facilitate such use cases.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        self.procs: list[multiprocessing.Process] = []
        self.num_processes = num_processes
        self.queue = multiprocessing.Queue()
        """ Allows us to collect results from processes (from https://stackoverflow.com/questions/10415028/how-to-get-the-return-value-of-a-function-passed-to-multiprocessing-process) """

    def _close_dead_procs(self):
        num_dead = 0
        to_remove = []

        for proc in self.procs:
            proc: multiprocessing.Process = proc
            if not proc.is_alive():
                proc.close()
                to_remove.append(proc)
                num_dead += 1

        for proc in to_remove:
            self.procs.remove(proc)

        return num_dead

    @staticmethod
    def _do_work(func, queue, i, vargs):
        """worker function"""
        ret = func(*vargs)
        queue.put([i, ret])

    def starmap(self, func: Callable, args: Iterable[Iterable]):
        """
        Distributes the execution of a function across multiple processes.

        This method takes a function and a sequence of argument tuples, and executes the function
        in parallel using the specified number of processes.

        Parameters
        ----------
        func : Callable
            The function to execute in parallel.
        args : Iterable[Iterable]
            An iterable of argument tuples to pass to the function.

        Returns
        -------
        list
            A list of results returned by the function, in the order of the input arguments.

        Raises
        ------
        AssertionError
            If the number of processes exceeds the specified limit.
        """
        # "ChatGPT 4o" assisted with generating this docstring.
        for proc_idx, proc_args in enumerate(args):
            # wait for a process slot to become available
            while len(self.procs) >= self.num_processes:
                if self._close_dead_procs() > 0:
                    break
                time.sleep(0.1)

            # create a new process and start it
            args2 = [func, self.queue, proc_idx, proc_args]
            proc = multiprocessing.Process(target=self._do_work, args=args2)
            proc.start()
            self.procs.append(proc)

        # collect processes
        while len(self.procs) > 0:
            self._close_dead_procs()

        # collect results
        rets = []
        while not self.queue.empty():
            idx_ret = self.queue.get()
            idx, ret = idx_ret[0], idx_ret[1]
            rets.append([idx, ret])
        rets = sorted(rets, key=lambda v: v[0])
        rets = [ret[1] for ret in rets]

        return rets
