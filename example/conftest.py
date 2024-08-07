import pytest

#
# Ensure pytest adds root directory to the system path.
#
def pytest_addoption(parser):
  parser.addoption(
    '--dir-input', action='store', default='', help='Base directory with data input'
  )
  parser.addoption(
    '--dir-output', action='store', default='', help='Base directory where output will be written'
  )

@pytest.fixture
def dir_input_fixture(request):
    return request.config.getoption('--dir-input')

@pytest.fixture
def dir_output_fixture(request):
    return request.config.getoption('--dir-output')