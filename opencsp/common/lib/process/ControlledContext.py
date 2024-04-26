import asyncio
import threading
from typing import Generic, TypeVar

import opencsp.common.lib.tool.log_tools as lt


T = TypeVar('T')


class ControlledContext(Generic[T]):
    """A simple way to protect object access for multithreading contention.

    This class is intended to wrap an object so that it can only be accessed in a controller manner.  It has a built in mutex and __enter__/__exit__ functions.  The object can best be accessed by a "with" statement.  For example::

        def set_to_one(controlled_val: ControlledContext[list[int]]):
            with controlled_val as val:
                for i in range(len(val)):
                    val[i] = 1

        threading_sensitive_value: list[int] = [0]
        controlled_tsv = ControlledContext(threading_sensitive_value)

        thread.start(set_to_one, controlled_tsv)
        print(str(tsv[0])) # will sometimes print '1'

        with controlled_tsv as tsv:
            tsv[0] = 0
            print(str(tsv[0])) # will always print '0'
    """

    def __init__(self, o: T, timeout: float = 1):
        self.o = o
        self.rlock = threading.RLock()
        self.timeout = timeout
        self.timed_out = False

    async def _acquire_with_timeout(self):
        if self.timeout:
            if not self.rlock.acquire(timeout=self.timeout):
                self.timed_out = True
        else:
            self.rlock.acquire()

    def __enter__(self):
        asyncio.run(self._acquire_with_timeout())
        if self.timed_out:
            lt.error_and_raise(
                asyncio.TimeoutError, f"Failed to acquire lock for {self.o} within {self.timeout} seconds"
            )
        return self.o

    def __exit__(self, exc_type, exc_value, traceback):
        self.rlock.release()

        # return False to enable exceptions to be re-raised
        return False
