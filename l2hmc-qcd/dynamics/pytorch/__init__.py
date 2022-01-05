import sys
from pathlib import Path

# - modules are here:
#     l2hmc-qcd/
# - but, we're here:
#     l2hmc-qcd/dynamics/pytorch/dynamics.py

MPATHS = Path(__file__).parent.parent.parent
if MPATHS not in sys.path:
    sys.path.append(str(MPATHS))
