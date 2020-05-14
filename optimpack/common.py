#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common types and functions for OptimPack.

Author: Éric Thiébaut (2020)
"""

from __future__ import print_function, division
import numpy as _np
import math

FLOAT32 = _np.dtype('float32')
FLOAT64 = _np.dtype('float64')

class Identifier:
    """
    Each instance of this class is considered as a unique identifier and may
    have associated data.

    Being object instances, identifiers must be tested with the `is` (or
    `is not`) operator, not with `==` (nor `!=`).

    In OptimPack, identifiers are used to indicate at which specific stage is a
    reverse communication algorithm and also to provide additional information.
    In these cases, the associated data is a descriptive message.

    """
    def __init__(self, data = None):
        self.data = data

INITIAL           = Identifier('algorithm not yet started')
SEARCH            = Identifier('search in progress')
COMPUTE_FG        = Identifier('caller shall provide f(x) and g(x)')
NEW_X             = Identifier('a new iterate is available for examination')
FINAL_X           = Identifier('a solution has been found within tolerances')
CONVERGENCE       = Identifier('algorithm has converged')
WARNING           = Identifier('algorithm finished with warnings')
FATOL_HOLDS       = Identifier('absolute function decrease criterion satisfied')
FRTOL_HOLDS       = Identifier('relative function decrease criterion satisfied')
GATOL_HOLDS       = Identifier('absolute gradient criterion satisfied')
GRTOL_HOLDS       = Identifier('relative gradient criterion satisfied')
FTOL_HOLDS        = Identifier('convergence tests on `ftol` satisfied')
GTOL_HOLDS        = Identifier('convergence tests on `gtol` satisfied')
XTOL_HOLDS        = Identifier('convergence tests on `xtol` satisfied')
FIRST_WOLFE_HOLDS = Identifier('first Wolfe condition satisfied')
STRONG_WOLFE_HOLD = Identifier('strong Wolfe conditions both satisfied')
STEP_AT_STEPMIN   = Identifier('step at lower bound')
STEP_AT_STEPMAX   = Identifier('step at upper bound')
ROUNDING_ERRORS   = Identifier('rounding errors prevent progress')

def of_type(arr, val):
    """Convert a scalar value to the same type as the elements of an array."""
    dtype = arr.dtype
    if dtype == FLOAT64:
        return _np.float64(val)
    elif dtype == FLOAT32:
        return _np.float32(val)
    else:
        raise ValueError('expecting array of floating-point values')

def vdot(x, y):
    """
    Yields the inner product of its arguments regardless of their shapes (their
    number of elements must however match).
    """
    return float(_np.inner(x.ravel(),y.ravel()))

def vnorm2(x):
    """Yields the Euclidean norm (L-2) of its argument."""
    return math.sqrt(vdot(x, x))

def vcreate(obj):
    """Create a new array of same shape and data type as given argument."""
    return _np.empty(obj.shape, obj.dtype)

def vscale(arr, val):
    """In-place scaling of the values in an array by a scalar multiplier."""
    if val == 0:
        arr.fill(0)
    elif val != 1:
        arr *= of_type(arr, val)

def vaxpy(dest, alpha, x, y):
    """Store in destination `dest` the result of `alpha*x + y`."""
    if dest is not y:
        _np.copyto(dest, y)
    if alpha == 1:
        dest += x
    elif alpha == -1:
        dest -= x
    elif alpha != 0:
        dest += of_type(x, alpha)*x

def is_real_number(obj):
    """Yield whether an object is a real number"""
    return isinstance(obj, float) or isinstance(obj, int)

# Exported public symbols.
__all__ = [s for s in dir() if not s.startswith('_')]
