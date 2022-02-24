"""
console.py

Contains global rich console to use in live displays.
"""
from __future__ import absolute_import, annotations, division, print_function
import os

from rich.console import Console


WIDTH = max(150, int(os.environ.get('COLUMNS', 150)))


def is_interactive():
    from IPython import get_ipython
    return get_ipython() is not None


console = Console(record=True,
                  color_system='truecolor',
                  log_path=False,
                  width=WIDTH)
console.width = WIDTH
