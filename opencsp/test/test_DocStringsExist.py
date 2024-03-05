from pathlib import Path

# Assume opencsp is in PYHTONPATH
from opencsp.app.sofast.lib.Sofast import Sofast
# TODO: import all user-facing classes here.

def test_docstrings_exist_for_methods():
    class_list = [
        Sofast
        # TODO: List all user-facing classes here.
    ]

    for class_module in class_list:
        method_list = [
            func for func in dir(class_module)
            if callable(getattr(class_module, func))
            and not func.startswith("__")
            and not func.startswith("_")
        ]

        for method in method_list:
            doc_exists = True
            if getattr(class_module, method).__doc__ is None:
                doc_exists = False

            print(f"doc_exists({class_module.__name__}.{method}): "
                  f"{doc_exists}")
            assert doc_exists
