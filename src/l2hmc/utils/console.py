"""
console.py

Contains global rich console to use in live displays.
"""
from __future__ import absolute_import, division, annotations, print_function
from rich.console import Console


console = Console(record=True, color_system='truecolor', log_path=False)
