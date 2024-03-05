import multiprocessing
import time
from typing import Callable, Iterable

class MultiprocessNonDaemonic():
    def __init__(self, num_processes: int):
        """ This class is like multiprocessing.Pool, but the processes it uses aren't daemonic.

        Some properties of daemonic processes include:
            - When a process exits, it attempts to terminate all of its daemonic child processes.
            - Note that a daemonic process is not allowed to create child processes. Otherwise a
              daemonic process would leave its children orphaned if it gets terminated when its
              parent process exits. Additionally, these are not Unix daemons or services, they
              are normal processes that will be terminated (and not joined) if non-daemonic
              processes have exited.

        The second point (not being able to spawn child processes) is why this class exists.
        Sometimes you want to be able to spawn grandchild processes from the child processes
        started with Multiprocessing.starmap(). One example use case is to spawn a grandchild
        process that handles rendering for the child process.

        Args:
        -----
            num_processes (int): The number of processes in this pool.
        """
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
        """ worker function """
        ret = func(*vargs)
        queue.put([i, ret])
    
    def starmap(self, func: Callable, args: Iterable[Iterable]):
        results = []

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