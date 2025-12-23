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


#    parser.addoption('--verbose', action='store', default='False', help='Output detailed information.')


@pytest.fixture
def dir_input_fixture(request):
    return request.config.getoption('--dir_input')


@pytest.fixture
def dir_output_fixture(request):
    return request.config.getoption('--dir_output')


# @pytest.fixture
# def verbose_fixture(request):
#     return request.config.getoption('--verbose')
