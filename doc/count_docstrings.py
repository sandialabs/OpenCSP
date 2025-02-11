import pkgutil
import inspect
from sphinx.application import Sphinx

# ChatGPT 4o-mini assisted with generating this code


def count_docstrings(module):
    count = 0
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if obj.__doc__:
                count += 1
    return count


def count_docstrings_in_package(package_name):
    package = __import__(package_name)
    total_count = 0

    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = __import__(module_name, fromlist='dummy')
        total_count += count_docstrings(module)

    return total_count


def print_docstring_summary(app: Sphinx, build_passed: bool):
    package_name = 'opencsp'  # Replace with your package name
    docstring_count = count_docstrings_in_package(package_name)
    print(f'Total number of docstrings in {package_name}: {docstring_count}')


def setup(app: Sphinx):
    app.connect('build-finished', print_docstring_summary)
