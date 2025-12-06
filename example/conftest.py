"""
Setup pytest for running examples.
"""

import pytest


#
# Add pytest command-line arguments supported by examples.
#
def pytest_addoption(parser):
    parser.addoption('--dir_input', action='store', default='', help='Base directory with data input')
    parser.addoption('--dir_output', action='store', default='', help='Base directory where output will be written')
    parser.addoption(
        '--write_full_data',
        action='store',
        default='False',
        help='If true, write out a directory structure including all input data.  Otherwise only write generated output.',
    )


@pytest.fixture
def dir_input_fixture(request):
    return request.config.getoption('--dir_input')


@pytest.fixture
def dir_output_fixture(request):
    return request.config.getoption('--dir_output')


@pytest.fixture
def write_full_data_fixture(request):
    return request.config.getoption('--write_full_data')
