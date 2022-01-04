"""
Collect useful software for non-linear dynamics
"""

from typing import Final

import numpy as np

EPS: Final = np.finfo(float).eps  # machine accuracy
SCALE: Final = 1.0e3
# SCALE natural scale of functions.
# we assume that absulate accurancy is better than SCALE*EPS
