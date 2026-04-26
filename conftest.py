"""Root conftest for pytest.

Two responsibilities:

1. Tell pytest's collector to never descend into directories that aren't tests
   and never to import the root __init__.py as a test module. The root
   __init__.py uses ComfyUI-style relative imports (`from .nodes import ...`)
   that only resolve when ComfyUI's loader runs the package — pytest can't
   import it standalone.

2. Add the project root to sys.path so test modules can `from lib.x import y`
   and `from nodes.x import y`. This replaces the older `pythonpath = ['.']`
   pyproject setting, which interacted badly with the root __init__.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

collect_ignore = ["__init__.py", "nodes", "lib", "workflows", "docs"]


def pytest_ignore_collect(collection_path, config):
    """Belt-and-suspenders: hard-block any attempt to collect __init__.py at rootdir.

    `collect_ignore` alone hasn't been sufficient in practice — pytest's package
    detection walks up from the test file and tries to import the parent package's
    __init__.py before our conftest.py's collect_ignore is consulted. Returning
    True here aborts that traversal explicitly.
    """
    name = collection_path.name if hasattr(collection_path, "name") else os.path.basename(str(collection_path))
    if name == "__init__.py" and os.path.dirname(str(collection_path)) == _HERE:
        return True
    return None
