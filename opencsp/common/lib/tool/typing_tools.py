import functools
import inspect
import types
from typing import Callable, Generic, TypeVar

from pandas import Series as pandas_Series

import opencsp.common.lib.tool.log_tools as lt

T = TypeVar("T")


def default(primary: T | Callable[[], T] | None, *default: T | Callable[[], T] | None) -> T | None:
    # TODO should this be in a different module?
    """Get the default value if the primary value is None or if the value is a
    callable that raises an error.

    Examples::

        default(None, "a") # returns "a"
        default(None, None) # returns None
        default(None, None, None, "a") # returns "a"

        lval = [ 1, 2, 3, 4 ]
        dval = { 1: 2, 3: None }
        default(lambda: lval[3], "a") # returns 4
        default(lambda: lval[4], "a") # returns "a"
        default(lambda: dval[1], "a") # returns 2
        default(lambda: dval[2], "a") # returns "a"
        default(lambda: dval[3], "a") # returns "a"

        default(raises_runtime_error, raises_runtime_error) # raises RuntimeError
        default(raises_runtime_error, "a") # returns "a"
        default(raises_runtime_error, raises_value_error) # raises RuntimeError
        default(raises_runtime_error, raises_value_error, None) # returns None
        default("a", raises_value_error) # returns "a"
        default(None, lambda: lval[4], dval[3], lambda: dval[4], "a") # returns "a"

        default("a") # raises ValueError
        default(None) # returns None

    This is especially helpful in cases where a simple check for None would
    cause python to raise an error, as in the following example::

        sval: string = None
        lval: list = None
        dval: dict = { 1: 2, 3: None }
        result = sval or lval or dval[1] or "a" # returns 2
        result = sval or lval or dval[2] or "a" # raises a KeyError
        result = default(sval, lval, dval[2], "a") # returns "a"

    Returns:
    --------
    first_non_none: any | None
        The first value (either primary or one of the defaults) that isn't None
        and doesn't raise an exception.

    Raises:
    -------
    primary_exception: Exception
        If the primary value is a callable, and all default values are
        callables, and all throw an exception when called, then the exception
        from the primary value is raised.
    """
    all_vals = [primary, *default]
    found_only_exceptions = True
    first_exception = None

    # validate the inputs
    if len(all_vals) == 1:
        # We enforce this rule to make code more readable.
        # Basically, we believe that this:
        #
        #     default(dval[3], None)
        #
        # Is more readable than this:
        #
        #     default(dval[3])
        lt.error_and_raise(
            ValueError,
            "Error in typing_tools.default(): " + "at least one alternative for a default value must be provided.",
        )

    for val in all_vals:
        if val is None:
            # val is None, keep looking
            found_only_exceptions = False
            pass
        else:
            if isinstance(val, Callable):
                # attempt to get a value to return from this callable
                try:
                    ret = val()
                    if ret != None:
                        return ret
                    found_only_exceptions = False
                except Exception as e:
                    if first_exception is None:
                        first_exception = e
                    continue
            else:
                # not a callable and not None, return this value
                found_only_exceptions = False
                return val

    # all values were either None or a callable that raised an exception
    if found_only_exceptions:
        raise first_exception
    return None


def strict_types(func):
    '''
    Decoratorates functions to make them strictly typed.

    Takes in keyword arguments with associated types.
    If the argument applied to the function does not pass an
    isinstance test against the defined type, the function will raise a TypeError.
    # TODO tristan: does not apply strictness to return value

    Important
    ---------
    In general it is important to understand that this decorator is not robust, but
    if it fails it should be very clear. In that case do not bother troubleshooting,
    I recommend simply removing the decorator, as it is likely a bug in the decorator.

    Notes
    -----
    * Using this decorator will block some information about the decorated function
    from some python functions that return function information.
    * A type represented with a string of itself (i.e. `'Vxyz'`) will
    not recognize subclasses properly, and in some instaces could break entirely.
    * Beware of using types that contain a secondary type within (e.x. `list[int]`),
    the inner types will be ignored and in some cases could fail entirely.
    * The function does not do any checks on the `return` value. (Might add in the future)

    Example
    -------
    ```python
    @strict_types
    def add(a: int, b: int):
        return a + b

    >>> add(1, 2)
    3

    >>> add(1.0, 2)
    TypeError: incorrect types in 'add'.
        At positional argument 1: type of '1.0' is float, should be int

    >>> add(1, b=2.0)
    TypeError: incorrect types in 'add'.
        At key word argument 'b': type of '2.0' is float, should be int

    >>> add(b=2.0, a=1j)
    TypeError: incorrect types in 'add'.
        At key word argument 'b': type of '2.0' is float, should be int
        At key word argument 'a': type of '1j' is complex, should be int

    ```
    '''

    @functools.wraps(func)
    def wrapper(*posargs, **kwargs):
        argspecs = inspect.getfullargspec(func)
        kwargtypes = argspecs.annotations
        for kw in kwargtypes:
            kwargtypes[kw] = ensure_not_generic(kwargtypes[kw])
        argnames = argspecs.args
        # store every type mismatch to alert user
        positional_type_mismatches = []
        key_word_type_mismatches = []

        # first we look at the positional arguments
        for i, (arg, argname) in enumerate(zip(posargs, argnames)):
            # print(argname in kwargtypes)
            if argname in kwargtypes:
                if (
                    arg != None
                    and type(arg).__name__ != kwargtypes[argname]  # for cases of types represented as strings
                    if type(kwargtypes[argname]) == str
                    else not isinstance(arg, kwargtypes[argname])
                ):
                    positional_type_mismatches.append((i, kwargtypes[argname]))

        # second we look at the key word arguments
        for kw in kwargs:
            if kw in kwargtypes:
                if (
                    kwargs[kw] != None
                    and type(kwargs[kw]).__name__ != kwargtypes[kw]  # for cases of types represented as strings
                    if type(kwargtypes[kw]) == str
                    else not isinstance(kwargs[kw], kwargtypes[kw])
                ):
                    key_word_type_mismatches.append((kw, kwargtypes[kw]))

        # if there are no type mismatches just run the function
        if len(positional_type_mismatches) + len(key_word_type_mismatches) != 0:
            error_info = f"incorrect types in '{func.__name__}'."

            for i, should_be_type in positional_type_mismatches:
                error_info += f"\n\tAt positional argument {i+1}: type of '{posargs[i]}' is {(type(posargs[i])).__name__}, should be {str(should_be_type)}"

            for kw, should_be_type in key_word_type_mismatches:
                error_info += f"\n\tAt key word argument '{kw}': type of '{kwargs[kw]}' is {(type(kwargs[kw])).__name__}, should be {str(should_be_type)}"

            raise TypeError(error_info)
        return func(*posargs, **kwargs)

    if wrapper.__doc__ == None:
        wrapper.__doc__ = ""
    wrapper.__doc__ += "\nThis function uses strictly enforced types, using the @strict_types decorator."
    return wrapper


def ensure_not_generic(t: type | types.GenericAlias) -> type:
    """converts and types.GenericAlias to a regular type"""
    if type(t) == types.GenericAlias:
        t = t.__origin__
    return t


def ensure_not_generic_alias(t: type | types.GenericAlias) -> type:
    """converts and types.GenericAlias to a regular type"""
    if type(t) == types.GenericAlias:
        t = t.__origin__
    return t


def ensure_not_string(t: type | str, class_container: type) -> type:
    """converts a string representation of a type into the actual type"""
    name = class_container.__name__
    if type(t) == str and t == name:
        t = class_container
    if type(t) == str and t != name:
        raise TypeError(
            "@strict_types cannot decorate a function "
            "that uses a string for the type of an argument "
            "that is not the class containing the function"
        )
    return t


# add type hint support for pandas Series
class Series(pandas_Series, Generic[T]):
    pass
