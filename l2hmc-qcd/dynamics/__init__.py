import os
import sys


HERE = os.path.abspath(os.path.dirname(__file__))
MODULEPATH = os.path.dirname(HERE)
if MODULEPATH not in sys.path:
    sys.path.append(MODULEPATH)

#  modulepath = os.path.join(os.path.dirname(__file__), '..')
#  sys.path.append(modulepath)

