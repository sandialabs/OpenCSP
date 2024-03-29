from contextlib import contextmanager


@contextmanager
def ignored(*exceptions):
    """ Easy wrapper to ignore specific kinds of exceptions.

    From https://stackoverflow.com/questions/15572288/general-decorator-to-wrap-try-except-in-python

    Ignore specific exception types. A common use case is to free attributes within the deconstructor without wanting to
    worry if the attribute has already been free'd. In this case you can ignore any non-fatal exceptions with the
    catch-all "Exception". For example::

        def __del__(self):
            with ignored(Exception):
                self.window.close()

    Or a more specific example::

        with ignored(ArithmeticError):
            dbz = 1 / 0
    """
    try:
        yield
    except exceptions:
        pass
