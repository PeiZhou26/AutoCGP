import os
import sys
from pathlib import Path

_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent

_paths_to_add = [
    str(_project_root),
    str(_project_root / "robosuite"),
    str(_project_root / "robosuite-task-zoo"),
    str(_project_root / "mimicgen_environments"),
]

for _path in _paths_to_add:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

del _current_file, _project_root, _paths_to_add, _path