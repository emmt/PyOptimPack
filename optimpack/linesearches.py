#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line-search methods for OptimPack.

Author: Éric Thiébaut (2020)
"""

from __future__ import print_function, division

import math
from optimpack.common import *

# A few constants for Moré & Thuente line-search method.
XTRAPL = 1.1
XTRAPU = 4.0
STPMAX = 1e20

class LineSearch:
    """Super-class for line-search methods.

    A line-search method is a component of an algorithm to minimize a
    multi-variate and smooth objective function `func(x)`.  The goal of a
    line-search method is to find the step length `stp` such that
    `func(x0 + stp*d)` is approximately minimized for given initial variables
    `x0` and search direction`d`.

    Instances of derived classes are intended to perfom line-search using
    reverse communication.  Assuming `lnsrch` is a line-search instance, a
    line-search is started by calling the `lnsrch.start(...)` method and,
    until convergence, the next step length `stp` suggested by the
    line-search method is given by:

         stp = lnsrch.step

    the caller shall then compute a new iterate as:

         x = x0 + stp*d

    and call the `lnsrch.iterate(...)` method to check for convergence and
    estimate the next step length to try.  These shall be repeated while

         lnsrch.stage is optimpack.linesearches.SEARCH

    is true.

    Example with Armijo's method:

        import optimpack.linesearches as linesearches
        from optimpack.common import vdot
        lnsrch = linesearches.ArmijoLineSearch()
        x0 = ...          # initial variables
        f0 = func(x0)     # objective function at x0
        g0 = grad(x0)     # gradient of objective function at x0
        d = ...           # search direction
        dtg0 = vdot(d,g0) # directional derivative
        stp = ...         # initial step length
        lnsrch.start(f0, dtg0, stp)
        while lnsrch.stage is linesearches.SEARCH:
            stp = lnsrch.step
            x = x0 + stp*d
            fx = func(x)
            gx = grag(x)
            if lnsrch.use_derivatives:
                # Compute directional derivative.
                dtgx = lnsrch.vdot(d,gx)
            else:
                # Directional derivative is not needed.
                dtgx = None
            lnsrch.iterate(stp, fx, dtgx)

        # Check final stage.
        if lnsrch.stage is not linesearches.CONVERGENCE:
            print('Algorithm finished with warnings: ', lnsrch.reason())

    """

    def __init__(self, stepmin=0.0, stepmax=None):
        stepmin = float(stepmin)
        if stepmin < 0.0:
            raise ValueError('`stepmin` must be nonnegative')
        if stepmax is not None:
            stepmax = float(stepmax)
            if stepmin > stepmax:
                raise ValueError('`stepmax` must be greater than or equal to `stepmin`')
        self.step = 0.0
        self.stepmin = stepmin
        self.stepmax = stepmax
        self._set_stage(INITIAL)

    def _set_stage(self, stage, info = None):
        self.stage = stage
        self.info = info

    def start(self, f0, g0, step=1.0, stepmin=None, stepmax=None):
        """Start a line-search.

        Assuming `lnsrch` is a line-search instance, then:

            lnsrch.start(f0, g0, stp)

        starts the line-search with `f0 = func(x0)` the objective function
        for the variables `x0` at the start the search, `g0 = ⟨d,grad(x0)⟩`
        the directional derivative of the objective function along the
        search direction `d` and `stp` the length of the first step to
        try.  Here `func(x)` and `grad(x)` denote the objective function
        and its gradient for the variables `x` while `⟨d,grad(x0)⟩` denotes
        the inner product of the search direction `d` with the gradient of
        the objective function at `x0`.

        After calling this method, first step length to try is given by:

            stp = lnsrch.step


        """
        # Get and check STEPMIN.
        if stepmin is None:
            stepmin = self.stepmin
        stepmin = float(stepmin)
        if stepmin < 0.0:
            raise ValueError('`stepmin` must be nonnegative')
        # Get and check STEPMAX.
        if stepmax is None:
            stepmax = self.stepmax
        if stepmax is not None:
            stepmax = float(stepmax)
            if stepmin > stepmax:
                raise ValueError('`stepmax` must be greater than or equal to `stepmin`')
        # Get next step.
        step = float(step)
        if step < stepmin:
            raise ValueError('`step` must be greater than or equal to `stepmin`')
        if stepmax is not None and step > stepmax:
            raise ValueError('`step` must be less than or equal to `stepmax`')
        if g0 > 0:
            raise ValueError('not a descent direction')
        self.finit = float(f0)
        self.ginit = float(g0)
        self.step = step
        self.stepmin = stepmin
        self.stepmax = stepmax
        self._set_stage(SEARCH, COMPUTE_FG)

    def iterate(self, f, g, step=1.0, stepmin=None, stepmax=None):
        """Compute next line-search step.

        The call:

            lnsrch.iterate(stp, f, g = None)

        updates the line-search instance `lnsrch` by providing the current
        step length `stp` and `f = func(x0 + stp*d)`, the corresponding
        value of the objective function `func(x)` --- `x0` denotes the
        variables at the start of the line search and `d` denotes the
        search direction.  If the line-search method implemented by
        `lnsrch` requires it, argument `g =  ⟨d,grad(x0 + stp*d)⟩` is
        the directional derivative at this point.  The statement:

            lnsrch.use_derivatives

        yields whether the line-search method implemented by `lnsrch`
        requires derivatives.

        After calling `lnsrch.iterate`, the next step length to try is given by:

            lnsrch.step

        In principle, the value of `stp` should be equal to that given by
        `lnsrch.step` before calling `lnsrch.iterate`.  Some line-search
        methods may however be able to accommodate from a different step
        length than the suggested one.

        """
        raise Exception('the `iterate` method shal be extended by sub-classes')

    def reason(self):
        """Yields a textual description of the current state of a line_search
        instance.

        """
        if self.info is not None:
            return self.info.data
        else:
            return self.stage.data


class ArmijoLineSearch(LineSearch):
    """This class implements Armijo backtracking line search method.

    See documentation about `LineSearch` for a general usage of an instance
    of this class.

    Armijo backtracking line search method is designed to find a step
    `stp` that satisfies the sufficient decrease condition (a.k.a. first
    Wolfe condition):

        f(stp) ≤ f(0) + ftol⋅stp⋅f'(0)

    where `f(stp)` is the value of the objective function for a step `stp`
    along the search direction while `f'(stp)` is the derivative of this
    function.  Starting with an initial estimate of `stp`, given by calling
    the `start` method, the method proceeds by dividing the step by some
    factor greater than one (typically 2) until the above condition holds.

    This line-search method has the following specific parameter:

    * `ftol` specifies the nonnegative tolerance for the sufficient
      decrease condition.

    Default settings are suitable for quasi-Newton optimization methods.

    The algorithm is described in:

    * L. Armijo, "Minimization of functions having Lipschitz continuous
      first partial derivatives" in Pacific Journal of Mathematics,
      vol. 16, pp. 1–3 (1966).

    """
    def __init__(self, ftol=1e-4, **kwds):
        LineSearch.__init__(self, **kwds)
        if ftol <= 0:
            raise ValueError("`ftol` must be greater than 0")
        if ftol > 0.5:
            raise ValueError("`ftol` must be less than or equal to 0.5")
        self.ftol = float(ftol)

        # Armijo's line search does not use the directional derivative to
        # refine the step.
        self.use_derivatives = False

    def iterate(self, step, f, g = None):
        if self.stage is not SEARCH:
            raise ValueError('stage should be `SEARCH`, call `start` method first')
        if step != self.step:
            raise ValueError('step length has changed')
        if step < self.stepmin or (self.stepmax is not None and step > self.stepmax):
            raise ValueError('out of range step length')
        if f <= self.finit + self.ftol*self.ginit*step:
            self._set_stage(CONVERGENCE, FIRST_WOLFE_HOLDS)
        elif step > self.stepmin:
            self.step = max(step/2.0, self.stepmin)
            self._set_stage(SEARCH, COMPUTE_FG)
        else:
            self.step = self.stepmin
            return self._set_stage(WARNING, STEP_AT_STEPMIN)

#------------------------------------------------------------------------------
class MoreThuenteLineSearch(LineSearch):
    """This class implements Moré & Thuente cubic line search method.

    See documentation about `LineSearch` for a general usage of an instance
    of this class.

    Moré & Thuente cubic line search method is designed to find a step
    `stp` that satisfies the sufficient decrease condition (a.k.a. first
    Wolfe condition):

        f(stp) ≤ f(0) + ftol⋅stp⋅f'(0)

    and the curvature condition (a.k.a. second strong Wolfe condition):

        abs(f'(stp)) ≤ gtol⋅abs(f'(0))

    where `f(stp)` is the value of the objective function for a step `stp`
    along the search direction while `f'(stp)` is the derivative of this
    function.

    This line-search method has the following specific parameters:

    * `ftol` specifies the nonnegative tolerance for the sufficient
      decrease condition.

    * `gtol` specifies the nonnegative tolerance for the curvature
      condition.

    * `xtol` specifies a nonnegative relative tolerance for an acceptable
       step.  The method exits with a warning if the relative size of the
       bracketting interval is less than `xtol`.

    Default settings are suitable for quasi-Newton optimization methods.

    The algorithm is described in:

    * J.J. Moré and D.J. Thuente, "Line search algorithms with guaranteed
      sufficient decrease" in ACM Transactions on Mathematical Software,
      vol. 20, pp. 286–307 (1994).

    """

    def __init__(self, ftol=0.001, gtol=0.9, xtol=0.1, **kwds):
        LineSearch.__init__(self, **kwds)
        if not is_real_number(ftol) or ftol <= 0.0:
             raise ValueError('`ftol` must be a positive value')
        if not is_real_number(gtol) or gtol <= 0.0:
             raise ValueError('`gtol` must be a positive value')
        if not is_real_number(xtol) or xtol <= 0.0:
             raise ValueError('`xtol` must be a positive value')
        self.ftol     = float(ftol)
        self.gtol     = float(gtol)
        self.xtol     = float(xtol)
        self.finit    = 0.0
        self.ginit    = 0.0
        self.stx      = 0.0
        self.fx       = 0.0
        self.gx       = 0.0
        self.sty      = 0.0
        self.fy       = 0.0
        self.gy       = 0.0
        self.lower    = 0.0
        self.upper    = 0.0
        self.width    = 0.0
        self.oldwidth = 0.0
        self.initial  = True
        self.brackt   = False

        # Moré & Thuente line search does use the directional derivative to
        # refine the step.
        self.use_derivatives = True

    def start(self, f0, g0, step=1.0, stepmin=None, stepmax=None):
        # Call super class version of the method and set a defalut value
        # for the upper bound of the step.
        LineSearch.start(self, f0, g0, step, stepmin, stepmax)
        if self.stepmax is None:
            self.stepmax = STPMAX

        # Initialize attributes.
        # Attributes STX, FX, GX contain the values of the step,
        # function, and derivative at the best step.
        # Attributes STY, FY, GY contain the value of the step,
        # function, and derivative at STY.
        # Attributes STEP, F, G contain the values of the step,
        # function, and derivative at STEP.
        self.stx       = 0.0
        self.fx        = f0
        self.gx        = g0
        self.sty       = 0.0
        self.fy        = f0
        self.gy        = g0
        self.lower     = 0.0
        self.upper     = (1.0 + XTRAPU)*self.step
        self.width     = self.stepmax - self.stepmin
        self.oldwidth  = 2.0*self.width
        self.initial   = True # The algorithm has two different stages.
        self.brackt    = False

    def iterate(self, step, f, g):
        if self.stage is not SEARCH:
            raise ValueError('stage should be `SEARCH`, call `start` method first')
        if step != self.step:
            raise ValueError('step length has changed')
        if step < self.stepmin or (self.stepmax is not None and step > self.stepmax):
            raise ValueError('out of range step length')
        step = float(step)

        # If psi(step) ≤ 0 and f'(step) ≥ 0 for some step, then the algorithm
        # enters the second stage.
        gtest = self.ftol*self.ginit
        ftest = self.finit + step*gtest
        if self.initial and f <= ftest and g >= 0.0:
            self.initial = False

        # Test for termination: convergence or warnings.
        if f <= ftest and abs(g) <= -self.gtol*self.ginit:
            return self._set_stage(CONVERGENCE, STRONG_WOLFE_HOLD)
        if step == self.stepmin and (f > ftest or g >= gtest):
            return self._set_stage(WARNING, STEP_AT_STEPMIN)
        if step == self.stepmax and f <= ftest and g <= gtest:
            return self._set_stage(WARNING, STEP_AT_STEPMAX)
        if self.brackt:
            if self.upper - self.lower <= self.xtol*self.upper:
                return self._set_stage(WARNING, XTOL_HOLDS)
            if step <= self.lower or step >= self.upper:
                return self._set_stage(WARNING, ROUNDING_ERRORS)

        # A modified function is used to predict the step during the first stage if
        # a lower function value has been obtained but the decrease is not
        # sufficient.

        if self.initial and f <= self.fx and f > ftest:

            # Call CSTEP to update STX, STY, and to compute the new step for the
            # modified function and its derivatives.
            (self.stx, fxm, gxm,
             self.sty, fym, gym,
             step, self.brackt,
             info) = cstep(self.stx, self.fx - self.stx*gtest, self.gx - gtest,
                           self.sty, self.fy - self.sty*gtest, self.gy - gtest,
                           step, f - step*gtest, g - gtest,
                           self.brackt, self.lower, self.upper)

            # Reset the function and derivative values for F.
            self.fx = fxm + self.stx*gtest
            self.fy = fym + self.sty*gtest
            self.gx = gxm + gtest
            self.gy = gym + gtest

        else:

            # Call CSTEP to update STX, STY, and to compute the new step.
            (self.stx, self.fx, self.gx,
             self.sty, self.fy, self.gy,
             step, self.brackt,
             info) = cstep(self.stx, self.fx, self.gx,
                           self.sty, self.fy, self.gy,
                           step, f, g,
                           self.brackt, self.lower, self.upper)

        # Decide if a bisection step is needed.
        if self.brackt:
            wcur = abs(self.sty - self.stx)
            if wcur >= 0.66*self.oldwidth:
                step = self.stx + 0.5*(self.sty - self.stx)
            self.oldwidth = self.width
            self.width = wcur

        # Set the minimum and maximum steps allowed for STP.
        if self.brackt:
            if self.stx <= self.sty:
                self.lower = self.stx
                self.upper = self.sty
            else:
                self.lower = self.sty
                self.upper = self.stx
        else:
            self.lower = step + XTRAPL*(step - self.stx)
            self.upper = step + XTRAPU*(step - self.stx)

        # Force the step to be within the bounds STPMAX and STPMIN.
        step = min(max(step, self.stepmin), self.stepmax)

        # If further progress is not possible, let STP be the best point
        # obtained during the search.
        if self.brackt and (step <= self.lower or step >= self.upper or
                            self.upper - self.lower <= self.xtol*self.upper):
            step = self.stx

        # Save next step to try.
        self.step = step

        # Obtain another function and derivative.
        self._set_stage(SEARCH, COMPUTE_FG)

#------------------------------------------------------------------------------
#
# History:
#
# - MINPACK-1 Project. June 1983
#   Argonne National Laboratory.
#   Jorge J. Moré and David J. Thuente.
#
# - MINPACK-2 Project. November 1993.
#   Argonne National Laboratory and University of Minnesota.
#   Brett M. Averick and Jorge J. Moré.
#
# - Python version.  May 2020.
#   Centre de Recherche Astrophysique de Lyon.
#   Éric Thiébaut.
#
def cstep(stx, fx, dx,
          sty, fy, dy,
          stp, fp, dp,
          brackt, stpmin, stpmax):
    """Compute a safeguarded cubic line-search step.

    The function `cstep` computes a safeguarded cubic step for a search
    procedure and updates an interval that contains a step that satisfies a
    sufficient decrease and a curvature condition.  The algorithm is
    described in:

    * J.J. Moré and D.J. Thuente, "Line search algorithms with guaranteed
      sufficient decrease" in ACM Transactions on Mathematical Software,
      vol. 20, pp. 286–307 (1994).

    Called as:

        cstep(stx, fx, dx,
              sty, fy, dy,
              stp, fp, dp,
              brackt, stpmin, stpmax) ->  (stx, fx, dx,
                                           sty, fy, dy,
                                           stp, brackt, info)

    The parameter `stx` contains the step with the least function value.
    The parameter `stp` contains the current step.  If `brackt` is set to
    `True`, then a minimizer has been bracketed in an interval with
    endpoints `stx` and `sty` and it is assumed that:

         min(stx,sty) < stp < max(stx,sty),

    and that the derivative at `stx` is negative in the direction of the
    step.

    On output, the updated parameters are returned.

    Parameter `brackt` specifies if a minimizer has been bracketed.
    Initially `brackt` must be set to `False`.  The returned value of
    `brackt` indicates if a minimizer has been bracketed.

    Parameters `stpmin` and `stpmax` specify the lower and the upper bounds
    for the step.

    Parameters `stx`, `fx` and `dx` specify the step, the function and the
    derivative at the best step obtained so far.  The derivative must be
    negative in the direction of the step, that is, `dx` and `stp - stx`
    must have opposite signs.  On return, these parameters are updated
    appropriately.

    Parameters `sty`, `fy` and `dy` specify the step, the function and the
    derivative at the other endpoint of the interval of uncertainty.  On
    return, these parameters are updated appropriately.

    Parameters `stp`, `fp` and `dp` specify the step, the function and the
    derivative at the current step.  If `brackt` is true, then `stp` must
    be between `stx` and `sty`.  On return, these parameters are updated
    appropriately.  The returned value of `stp` is the next trial step.

    The returned value `info` indicates which case occured for computing
    the new step.

    """

    # Check the input parameters for errors.
    if brackt and (stp <= min(stx, sty) or stp >= max(stx, sty)):
        raise ValueError('`stp` outside bracket `(stx,sty)`')
    if dx*(stp - stx) >= 0.0:
        raise ValueError('descent condition violated')
    if stpmax < stpmin:
        raise ValueError('invalid bounds (`stpmax < stpmin`)')

    # Determine if the derivatives have opposite sign.
    opposite = ((dx < 0.0 and dp > 0.0) or
                (dx > 0.0 and dp < 0.0))

    if fp > fx:
        # First case.  A higher function value.  The minimum is bracketed.
        # If the cubic step is closer to STX than the quadratic step, the
        # cubic step is taken, otherwise the average of the cubic and
        # quadratic steps is taken.
        info = 1
        theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*math.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpc = stx + r*(stp - stx)
        stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/2.0)*(stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc)/2.0
        brackt = True
    elif opposite:
        # Second case.  A lower function value and derivatives of opposite
        # sign.  The minimum is bracketed.  If the cubic step is farther
        # from STP than the secant (quadratic) step, the cubic step is
        # taken, otherwise the secant step is taken.
        info = 2
        theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*math.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p/q
        stpc = stp + r*(stx - stp)
        stpq = stp + (dp/(dp - dx))*(stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # Third case.  A lower function value, derivatives of the same
        # sign, and the magnitude of the derivative decreases.  The cubic
        # step is computed only if the cubic tends to infinity in the
        # direction of the step or if the minimum of the cubic is beyond
        # STP.  Otherwise the cubic step is defined to be the secant step.
        info = 3
        theta = 3.0*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        # The case GAMMA = 0 only arises if the cubic does not tend to
        # infinity in the direction of the step.
        t = (theta/s)**2 - (dx/s)*(dp/s)
        if t > 0.0:
            gamma = s*math.sqrt(t)
            if stp > stx:
                gamma = -gamma
        else:
            gamma = 0.0
        p = (gamma - dp) + theta
        #q = ((gamma - dp) + gamma) + dx
        q = (gamma + (dx - dp)) + gamma
        r = p/q
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r*(stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp/(dp - dx))*(stx - stp)
        if brackt:
            # A minimizer has been bracketed.  If the cubic step is closer
            # to STP than the secant step, the cubic step is taken,
            # otherwise the secant step is taken.
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            t = stp + 0.66*(sty - stp)
            if stp > stx:
               stpf = min(t, stpf)
            else:
               stpf = max(t, stpf)
        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = max(stpmin, min(stpf, stpmax))
    else:
        # Fourth case.  A lower function value, derivatives of the same
        # sign, and the magnitude of the derivative does not decrease.  If
        # the minimum is not bracketed, the step is either STPMIN or
        # STPMAX, otherwise the cubic step is taken.
        info = 4
        if brackt:
            theta = 3.0*(fp - fy)/(sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            t = theta/s
            gamma = s*math.sqrt((theta/s)**2 - (dy/s)*(dp/s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p/q
            stpc = stp + r*(sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer and guess for next
    # step.
    if fp > fx:
        sty, fy, dy = stp, fp, dp
    else:
        if opposite:
            sty, fy, dy = stx, fx, dx
        stx, fx, dx = stp, fp, dp

    return (stx,fx,dx, sty,fy,dy, stpf, brackt, info)

#------------------------------------------------------------------------------
