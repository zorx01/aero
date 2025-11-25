
import sys
import os
from pathlib import Path
import importlib.util

# Try to import the extension module
# It is located in the same directory as this file
try:
    # Attempt relative import first (works if installed as package)
    from . import pycuvslam
except ImportError:
    # Fallback for direct execution or if relative import fails
    # This might happen if the .so is not treated as a package member correctly
    # or if we are in a weird environment.
    
    # Check if the .so file exists in this directory
    _current_dir = Path(__file__).parent
    _so_path = _current_dir / "pycuvslam.so"
    
    if _so_path.exists():
        spec = importlib.util.spec_from_file_location("pycuvslam", _so_path)
        if spec and spec.loader:
            pycuvslam = importlib.util.module_from_spec(spec)
            sys.modules["pycuvslam"] = pycuvslam
            spec.loader.exec_module(pycuvslam)
        else:
            raise ImportError(f"Could not load extension from {_so_path}")
    else:
        # If the file doesn't exist, we might be in a test environment without the binary.
        # We will re-raise the ImportError unless we want to mock it here.
        # For now, we assume the user has the file or we are mocking it externally.
        raise

# Re-export everything from the module
# We iterate over dir(pycuvslam) and set them as attributes of this module
# This effectively makes this module a proxy for pycuvslam
for _name in dir(pycuvslam):
    if not _name.startswith("__"):
        globals()[_name] = getattr(pycuvslam, _name)

# Explicitly export common classes for static analysis if possible
# (This is dynamic, so static analysis tools might struggle, but runtime will work)
