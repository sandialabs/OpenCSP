import os
import pathlib
import sys
import unittest
import numpy as np
import ctypes as cts
from ctypes.util import find_library


@unittest.skipIf(sys.platform != 'win32', "Windows-only: hardcoded cudart64_12.dll and path layout")
class CudaIntegrationCase(unittest.TestCase):
    def setUp(self):
        cwd = pathlib.Path(os.getcwd())
        if 'tests' in cwd.as_posix():
            os.chdir('../src/pyffiam/')

    def tearDown(self):
        pass

    def test_c_ffiam_found(self):
        cwd = pathlib.Path(os.getcwd())
        repo_dir = cwd.parent.parent.parent
        ffiam_dir = repo_dir.joinpath('ffiam')
        self.assertTrue(ffiam_dir.exists())

    def test_dependencies_found(self):
        cuda_lib = find_library('cudart64_12.dll')
        self.assertTrue('cudart64' in cuda_lib)

    def test_dependencies_loaded(self):
        cuda_lib = find_library('cudart64_12.dll')
        val = cts.cdll.LoadLibrary(cuda_lib)
        self.assertEqual(type(val), cts.CDLL)


@unittest.skipIf(sys.platform != 'win32', "Windows-only: hardcoded cudart64_12.dll and ffiam_lib.dll")
class ApiCase(unittest.TestCase):
    def setUp(self):
        _ = cts.cdll.LoadLibrary(find_library('cudart64_12.dll'))
        # Use absolute path to ffiam_lib.dll (relative to pyffiam source directory)
        pyffiam_src = pathlib.Path(__file__).parent.parent / 'src' / 'pyffiam'
        lib_path = pyffiam_src / 'external' / 'ffiam_lib.dll'
        if not lib_path.exists():
            self.skipTest(f"ffiam_lib.dll not found at {lib_path}")
        self._lib = cts.cdll.LoadLibrary(lib_path.as_posix())

    def test_sample_add_func(self):
        """ Tests that sample c function is accessible from python. """
        result = self._lib.Py_TestAdd(3, 5)
        self.assertEqual(type(result), int)
        self.assertEqual(result, 8)

    def test_sample_numpy_func(self):
        """ Verify accessing c array from numpy via pointer """
        arr_size = 10
        libc = cts.CDLL("msvcrt")
        libc.malloc.restype = cts.c_void_p

        # get a pointer to a block of data from malloc
        data_ptr = libc.malloc(arr_size * cts.sizeof(cts.c_int))
        data_ptr = cts.cast(data_ptr, cts.POINTER(cts.c_int))

        arr = np.ctypeslib.as_array(data_ptr, shape=(arr_size,))

        arr[:] = range(arr_size)
        print(f"Numpy array ({arr.shape}, {arr.dtype}):", arr[:arr_size])
        print("Data pointer: ", data_ptr[:arr_size])

        self.assertEqual(arr.shape[0], arr_size)
        self.assertEqual(arr.dtype, np.int32)
        np.testing.assert_array_equal(arr, data_ptr[:arr_size])

        # numpy doesn't own its memory so must explicitly free
        del arr
        libc.free(data_ptr)

