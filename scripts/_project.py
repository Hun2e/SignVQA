"""Shared paths for repository scripts and generated local notebooks."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

NOTEBOOK_SETUP = """# Resolve shared code and local assets from the repository root.
import os as _project_os
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = next((p for p in [_ProjectPath.cwd(), *_ProjectPath.cwd().parents]
                      if (p / 'src' / 'ocr_engine.py').is_file()), None)
if _project_root is not None:
    _project_os.chdir(_project_root)
    if str(_project_root / 'src') not in _project_sys.path:
        _project_sys.path.insert(0, str(_project_root / 'src'))
# End repository path setup.

"""

def prepare_notebook(nb):
    """Keep cell indexes stable for notebook generation and verification."""
    cell = next(c for c in nb['cells'] if c['cell_type'] == 'code')
    source = ''.join(cell['source'])
    if not source.startswith(NOTEBOOK_SETUP):
        cell['source'] = (NOTEBOOK_SETUP + source).splitlines(keepends=True)
