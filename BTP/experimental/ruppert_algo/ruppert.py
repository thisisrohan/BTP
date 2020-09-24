import numpy as np
from DT.tools.adaptive_predicates import incircle, orient2d, exactinit2d
from DT.final_2D_robust_multidimarr import _walk, _identify_cavity, \
                                           _make_Delaunay_ball, initialize, \
                                           _cavity_helper

def njit(f):
    return f
# from numba import njit


