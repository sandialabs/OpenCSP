import asyncio
from typing import Generic, TypeVar


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

    def __init__(self, o: T):
        self.o = o
        self.mutex = asyncio.Lock()

    def __enter__(self):
        self.mutex.acquire()
        return self.o

    def __exit__(self, exc_type, exc_value, traceback):
        self.mutex.release()
        return False
