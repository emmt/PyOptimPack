#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conjugate gradient methods for OptimPack.

Author: Éric Thiébaut (2020)
"""

from __future__ import print_function, division

#from optimpack import vdot, vnorm2
from . import linesearches as linesearches
from optimpack.common import *
import numpy as _np
import math
import sys
import time

__all__ = ['nlcg', 'NonlinearConjugateGradient']

def default_printer(opt, x, f, g):
    elapsed = opt.elapsed_time()*1E3 # elapsed time in milliseconds
    gnorm = vnorm2(g)       # gradient norm FIXME: opt.gnorm?
    step  = opt.alpha       # step length
    iters = opt.iterations  # number of algorithm iterations
    evals = opt.evaluations # number of function evaluations
    rests = opt.restarts    # number of algorithm restarts
    if iters == 0:
        print(u' ITER  EVAL  REST     CPU (ms)          FUNC          ',
              u'   ‖GRAD‖    STEP', sep='')
        print(u'------------------------------------------------------',
              u'------------------', sep='')
    print('{:5d} {:5d} {:5d} {:12.3f} {:< 24.15e} {:<8.1e} {:<8.1e}'.format(
        iters, evals, rests, elapsed, f, gnorm, step))

def nlcg(fg, x, inplace = True, verb = 0, printer = default_printer,
         maxiter = None, maxeval = None, **kwds):
    opt = NonlinearConjugateGradient(x, **kwds)
    if inplace:
        g = vcreate(x)
    prnt = False
    stop = False
    mesg = ''
    while stop == False:
        stage = opt.stage
        if stage is COMPUTE_FG:
            # Compute the objective function and its gradient at the
            # current iterate unless the number of evaluations would exceed
            # the limit.
            if maxeval is not None and opt.evaluations >= maxeval:
                # FIXME: find a way to recover best solution so far
                stop = True
                mesg = 'too many evaluations'
            elif inplace:
                f = fg(x, g)
            else:
                f, g = fg(x)
        elif stage is NEW_X:
            iters = opt.iterations
            if maxiter is not None and iters >= maxiter:
                stop = True
                mesg = 'too many iterations'
            prnt = (verb > 0 and (stop or iters % verb == 0))
        elif stage is FINAL_X:
            stop = True
            mesg = 'algorithm has converged'
            prnt = (verb > 0)
        if prnt:
            printer(opt, x, f, g)
            prnt = False
        opt.iterate(x, f, g)

    # Return solution.
    if verb > 0 and len(mesg) > 0:
        print(mesg)
    return x

class NonlinearConjugateGradient:
    """
    Implement reverse communication version of non-linear conjugate gradient
    method.
    Example:
        import optimpack.conjugategradient as methods
        verb = True
        opt = methods.NonlinearConjugateGradient(x)
        while True:
            stage = opt.stage
            if stage is methods.COMPUTE_FG:
                # Compute objective function and gradient at x.
                f = func(x)
                g = grad(x)
            elif stage is methods.NEW_X or stage is methods.FINAL_X:
                # New iterate is available.
                if verb:
                    print('iter: ', opt.iterations, ' f: ', f)
                if stage is methods.FINAL_X:
                    break
            opt.iterate(x, f, g)
    """
    def __init__(self, x, method = 'Hestenes-Stiefel', lnsrch = None,
                 Powell = True, ShannoPhua = True,
                 stepmin = 0.0, stepmax = None,
                 fmin = None, epsilon = 0.0, delta = 1e-5,
                 gtol = (0, 1e-5)):
        # Definitions of the variables of the problem.
        if not isinstance(x, _np.ndarray):
            raise ValueError('variables must be stored in a NumPy array')
        if x.dtype != FLOAT64 and x.dtype != FLOAT32:
            raise ValueError('variables must be stored as floating-point values')
        self.shape = x.shape
        self.dtype = x.dtype

        # Function value at the start of the line search.
        self.f0 = float(0.0)

        # Euclidean norm of G0, the gradient at the start of the line
        # search.
        self.g0norm = float(0.0)

        # Euclidean norm of G, the gradient of the last accepted point.
        self.gnorm = float(0.0)

        # Directional derivative at the start of the line search; given by
        # the inner product: -<d,g0>
        self.dtg0 = float(0.0)

        # Directional derivative at the last trial point; given by the
        # inner product: -<d,g>
        self.dtg = float(0.0)

        # Gradient tolerances for the convergence.
        if (isinstance(gtol, tuple) or isinstance(gtol, list)) and len(gtol) == 2:
            gatol, grtol = gtol
        else:
            raise ValueError('`gtol` must be a 2-list or a 2-tuple of values')

        # Relative threshold for the norm or the gradient (relative to the
        # initial gradient) for convergence.
        if is_real_number(grtol) and 0 <= grtol < 1:
            self.grtol = float(grtol)
        else:
            raise ValueError('`grtol` must a nonnegative real number less than one')

        # Absolute threshold for the norm or the gradient for convergence.
        if is_real_number(gatol) and gatol >= 0:
            self.gatol = float(gatol)
        else:
            raise ValueError('`gatol` must a nonnegative real number')

        # Euclidean norm of the initial gradient.
        self.ginit = None

        # Minimal function value if provided.
        if fmin is None:
            self.fmin = None
        elif is_real_number(fmin):
            self.fmin = float(fmin)
        else:
            raise ValueError('`fmin` must be `None` or a real number')

        # Relative size for a small step.
        if is_real_number(delta) and 0 <= delta < 1:
            self.delta = float(delta)
        else:
            raise ValueError('`delta` must be a nonnegative real number less than one')

        # Threshold to accept descent direction.
        if is_real_number(epsilon) and 0 <= epsilon < 1:
            self.epsilon = float(epsilon)
        else:
            raise ValueError('`epsilon` must be a nonnegative real number less than one')

        # Current step length.
        self.alpha = float(0.0)

        # Current parameter in conjugate gradient update rule (for
        # information).
        self.beta = float(0.0)

        # Relative lower bound for the step length.
        if is_real_number(stepmin) and stepmin >= 0:
            self.stepmin = float(stepmin)
        else:
            raise ValueError('`stepmin` must be a nonnegative real number')

        # Relative upper bound for the step length.
        if stepmax is None:
            self.stepmax = None
        elif is_real_number(stepmax) and stepmax > stepmin:
            self.stepmax = float(stepmax)
        else:
            raise ValueError('`stepmax` must be `None` or a real number greater than `stepmin`')

        # The update method is called to update the search direction.  The
        # returned value indicates whether the updating rule has been
        # successful, otherwise a restart is needed.
        if method == 'Polak-Ribiere-Polyak':
            self._update = _update_Polak_Ribiere_Polyak
            g0_needed = True
            y_needed = True
        elif method == 'Fletcher-Reeves':
            self._update = _update_Fletcher_Reeves
            g0_needed = False
            y_needed = False
        elif method == 'Hestenes-Stiefel':
            self._update = _update_Hestenes_Stiefel
            g0_needed = True
            y_needed = True
        elif method == 'Fletcher':
            self._update = _update_Fletcher
            g0_needed = False
            y_needed = False
        elif method == 'Liu-Storey':
            self._update = _update_Liu_Storey
            g0_needed = True
            y_needed = True
        elif method == 'Dai-Yuan':
            self._update = _update_Dai_Yuan
            g0_needed = True
            y_needed = True
        elif method == 'Perry-Shanno':
            self._update = _update_Perry_Shanno
            g0_needed = True
            y_needed = True
        elif method == 'Hager-Zhang':
            self._update = _update_Hager_Zhang
            self.strictHagerZhang = False
            g0_needed = True
            y_needed = True
        elif method == 'Hager-Zhang (strict)':
            # Idem but conform to Hager & Zhang original method.
            self._update = _update_Hager_Zhang
            self.strictHagerZhang = True
            g0_needed = True
            y_needed = True
        else:
            raise ValueError('unknown method')

        # Force beta >= 0 according to Powell's prescription?
        if Powell is None or Powell is True:
            self.Powell = True
        elif Powell is False:
            self.Powell = False
        else:
            raise ValueError('parameter `Powell` must be `True`, `False` or `None`')

        # Compute the initial step size from the previous iteration
        # according to Shanno & Phua?
        if  ShannoPhua is None or ShannoPhua is True:
            self.ShannoPhua = True
        elif ShannoPhua is False:
            self.ShannoPhua = False
        else:
            raise ValueError('parameter `ShannoPhua` must be `True`, `False` or `None`')

        # Line search method.
        if lnsrch is None:
            #self.lnsrch = linesearches.ArmijoLineSearch()
            self.lnsrch = linesearches.MoreThuenteLineSearch(ftol=0.001, gtol=0.1)
        elif isinstance(lnsrch, linesearches.LineSearch):
            self.lnsrch = lnsrch
        else:
            raise ValueError('parameter `lnsrch` must be an instance of `LineSearch`')

        # Variables at start of line search.
        self.x0 = vcreate(self)

        # Gradient at start of line search.
        if g0_needed:
            self.g0 = vcreate(self)
        else:
            self.g0 = None

        # (Anti-)search direction, new iterate is searched
        # as: x = x0 - alpha*d, for alpha >= 0.
        self.d = vcreate(self)

        # Work vector (e.g., to store the gradient
        # difference: Y = G - G0).
        if y_needed:
            self.y = vcreate(self)
        else:
            self.y = None

        # FIXME: do allocation stuff in the start() method.
        self.start(x)


    def start(self, x):
        """(Re)start optimizer with same parameters but new initial variables."""
        # Number of iterations.
        self.iterations = 0

        # Number of algorithm restarts.
        self.restarts = 0

        # Number of function and gradient evaluations.
        self.evaluations = 0

        # Current stage.
        self.stage = COMPUTE_FG

        # Starting and elapsed times (in seconds).
        self.start_time = time.clock()

    def elapsed_time(self):
        """Yield elapsed time (in seconds) since algorithm started."""
        return time.clock() - self.start_time

    def iterate(self, x, f, g):
        #
        # The new iterate is:
        #    x_{k+1} = x_{k} - \alpha_{k} d_{k}
        # as we consider the anti-search direction here.
        #
        stage = self.stage
        lnsrch = self.lnsrch
        if stage is COMPUTE_FG:
            # Objective function has been evaluated one more time.
            self.evaluations += 1
            if self.evaluations > 1:
                # Line search in progress. Compute directional derivative
                # and check whether line search has converged.
                self.dtg = -vdot(self.d, g)
                lnsrch.iterate(self.alpha, f, self.dtg)
                if lnsrch.stage is SEARCH:
                    # Line search has not converged.  Get step length,
                    # compute a new trial point along the search direction
                    # and retrun to caller to compute the objective
                    # function and its gradient.
                    self.alpha = lnsrch.step
                    vaxpy(x, -self.alpha, self.d, self.x0)
                    return
                elif (lnsrch.stage is CONVERGENCE
                      or (lnsrch.stage is WARNING
                          and lnsrch.info is ROUNDING_ERRORS)):
                    # Line search has converged.  Increment number of
                    # iterations.
                    self.iterations += 1
                else:
                    raise Exception(lnsrch.reason())

            # The current step is acceptable.  Check for global
            # convergence.
            self.gnorm = vnorm2(g)
            if self.evaluations <= 1:
                self.ginit = self.gnorm
            if self.gnorm <= max(0.0, self.gatol, self.grtol*self.ginit):
                self.stage = FINAL_X
            else:
                self.stage = NEW_X
            return

        elif stage is NEW_X or stage is FINAL_X:

            # Compute search direction and initial step size.
            if self.evaluations <= 1 or self._update(self, x, g) == False:
                # First evaluation or update failed, set DTG to zero to use
                # the steepest descent direction.
                dtg = 0.0
            else:
                dtg = -vdot(self.d, g)
                if self.epsilon > 0 and dtg > -self.epsilon*vnorm2(self.d)*self.gnorm:
                    # Set DTG to zero to indicate that we do not have a
                    # sufficient descent direction.
                    dtg = 0
            self.dtg = float(dtg)
            if self.dtg < 0:
                # The recursion yields a sufficient descent direction (not
                # all methods warrant that).  Compute an initial step size
                # ALPHA along the new direction.
                if self.ShannoPhua:
                    # Initial step size is such that:
                    # <alpha_{k+1}*d_{k+1},g_{k+1}> = <alpha_{k}*d_{k},g_{k}>
                    self.alpha *= (self.dtg0/self.dtg)
            else:
                # Initial search direction or recurrence has been restarted.  FIXME:
                # other possibility is to use Fletcher's formula, see BGLS p. 39)
                if self.evaluations > 1:
                    self.restarts += 1
                _np.copyto(self.d, g)
                self.dtg = -self.gnorm*self.gnorm
                if self.fmin is not None and self.fmin < f:
                    alpha = 2*(self.fmin - f)/self.dtg
                elif f != 0:
                    alpha = 2*abs(f/self.dtg)
                else:
                    dnorm = self.gnorm
                    xnorm = vnorm2(x)
                    if xnorm > 0:
                        alpha = self.delta*xnorm/dnorm
                    else:
                        alpha = self.delta/dnorm
                self.alpha = float(alpha)
                self.beta = float(0.0)

            # Store current position as X0, f0, etc.
            _np.copyto(self.x0, x)
            self.f0 = float(f)
            if self.g0 is not None:
                _np.copyto(self.g0, g)
            self.g0norm = self.gnorm
            self.dtg0 = self.dtg

            # Start the line search and break to compute the first trial point
            # along the line search.
            stepmin = self.stepmin*self.alpha
            stepmax = None if self.stepmax is None else self.stepmax*self.alpha
            lnsrch.start(self.f0, self.dtg0, self.alpha,
                         stepmin = stepmin, stepmax = stepmax)
            if lnsrch.stage is SEARCH:
                self.stage = COMPUTE_FG
            else:
                raise Exception(lnsrch.reason())

            # Compute a trial point along the line search.
            vaxpy(x, -self.alpha, self.d, self.x0)
            self.stage = COMPUTE_FG

        else:
            # There must be something wrong.
            raise ValueError('unexpected stage')


# Form: Y = G - G0
def _form_y(obj, g):
    y = obj.y
    vaxpy(y, -1, obj.g0, g)
    return y

#
# Most non-linear conjugate gradient methods, update the new search direction
# by the following rule:
#
#     d' = -g + beta*d
#
# with d' the new search direction, g the current gradient, d the previous
# search direction and beta a parameter which depends on the method.
#
# Some methods (e.g., Perry & Shanno) implement the following rule:
#
#     d' = (-g + beta*d + gamma*y)*delta
#
# with y = g - g0.
#
# For us, the anti-search direction is used instead, thus:
#
#     d' = g + beta*d
#
def _update_common(self, g, beta):
    """
    Helper function to compute search direction as: `d = g + beta*d` or as
    `d = g + max(beta,0)*d` if Powell's prescription is applied.  Returns
    whether previous direction was taken into account; otherwise the new
    search direction is just the steepest ascent.
    """
    if self.Powell:
        beta = max(beta, 0.0)
    flag = (beta != 0)
    if flag:
        beta = of_type(self.d, beta)
        _np.copyto(self.d, g + beta*self.d)
    else:
        _np.copyto(self.d, g)
    self.beta = float(beta)
    return flag
#
# For Hestenes & Stiefel method:
#
#     beta = <g,y>/<d,y>
#
# with y = g - g0.
#
def _update_Hestenes_Stiefel(obj, x, g):
    y = _form_y(obj, g)
    d = obj.d
    gty =  vdot(g, y)
    dty = -vdot(d, y)
    beta = gty/dty if dty != 0 else 0.0
    return _update_common(obj, g, beta)

#
# For Fletcher & Reeves method:
#
#     beta = <g,g>/<g0,g0>
#
# (this value is always >= 0 and can only be zero at a stationary point).
#
def _update_Fletcher_Reeves(obj, x, g):
    r = obj.gnorm/obj.g0norm
    return _update_common(obj, g, r*r)

#
# For Polak-Ribière-Polyak method:
#
#     beta = <g,y>/<g0,g0>
#
def _update_Polak_Ribiere_Polyak(obj, x, g):
    y = _form_y(obj, g)
    beta = (vdot(g, y)/obj.g0norm)/obj.g0norm
    return _update_common(obj, g, beta)

#
# For Fletcher "Conjugate Descent" method:
#
#     beta = -<g,g>/<d,g0>
#
# (this value is always >= 0 and can only be zero at a stationnary point).
#
def _update_Fletcher(obj, x, g):
    beta = -obj.gnorm*(obj.gnorm/obj.dtg0)
    return _update_common(obj, g, beta)

#
# For Liu & Storey method:
#
#     beta = -<g,y>/<d,g0>
#
def _update_Liu_Storey(obj, x, g):
    y = _form_y(obj, g)
    dtg0 = obj.dtg0
    gty =  vdot(g, y)
    beta = -gty/dtg0
    return _update_common(obj, g, beta)

#
# For Dai & Yuan method:
#
#     beta = <g,g>/<d,y>
#
def _update_Dai_Yuan(obj, x, g):
    y = _form_y(obj, g)
    d = obj.d
    dty = -vdot(d, y)
    if dty != 0:
        gnorm = obj.gnorm
        beta = gnorm*(gnorm/dty)
    else:
        beta = 0.0
    return _update_common(obj, g, beta)

#
# For Hager & Zhang method:
#
#     beta = <y - (2*<y,y>/<d,y>)*d,g>/<d,y>
#          = (<g,y> - 2*<y,y>*<d,g>/<d,y>)/<d,y>
#
def _update_Hager_Zhang(obj, x, g):
    y = _form_y(obj, g)
    d = obj.d
    dty = -vdot(d, y)
    if dty == 0:
        beta = 0.0
    elif obj.strictHagerZhang:
        # Original formulation, using Y as a scratch vector.
        q = 1.0/dty
        r = 2.0*(q*vnorm2(y))**2
        _np.copy(y,
                 of_type(y, q)*y +
                 of_type(d, r)*d)
        beta = vdot(y, g)
    else:
        # Improved formulation which spares one linear combination and
        # thus has less overhead (only 3 scalar products plus 2 linear
        # combinations instead of 3 scalar products and 3 linear
        # combinations).  The rounding errors are however different, so
        # one or the other formulation can be, by chance, more
        # efficient.  Though there is no systematic trend.
        ytg = vdot(y, g)
        dtg = obj.dtg
        ynorm = vnorm2(y)
        beta = (ytg - 2.0*(ynorm/dty)*ynorm*dtg)/dty
    return _update_common(obj, g, beta)

# Perry & Shanno, update rule (used in CONMIN and see Eq. (1.4) in [3])
# writes:
#
#     d' = alpha*(-c1*g + c2*d - c3*y)  ==>   d' = c1*g + c2*d + c3*y
#
#     c1 = (1/alpha)*<s,y>/<y,y>
#        =  <d,y>/<y,y>
#        = -<d,y>/<y,y>
#
#     c2 = <g,y>/<y,y> - 2*<s,g>/<s,y>
#        = <g,y>/<y,y> - 2*<d,g>/<d,y>
#        = <g,y>/<y,y> - 2*<d,g>/<d,y>
#
#     c3 = -(1/alpha)*<s,g>/<y,y>
#        = -<d,g>/<y,y>
#        =  <d,g>/<y,y>
#
# with alpha the step length, s = x - x0 = alpha*d = -alpha*d.  For this
# method, beta = c2/c1.
#
def _update_Perry_Shanno(obj, x, g):
    y = _form_y(obj, g)
    d = obj.d
    yty = vdot(y, y)
    dty = -vdot(d, y) if yty > 0 else 0.0
    if dty == 0:
        _np.copy(d, g)
        obj.beta = float(0.0)
        return False
    else:
        dtg = obj.dtg
        gty = vdot(g, y)
        c1 = dty/yty
        c2 = gty/yty - 2.0*dtg/dty
        c3 = -dtg/yty
        _np.copy(d,
                 of_type(g, c1)*g +
                 of_type(d, c2)*d +
                 of_type(y, c3)*y)
        obj.beta = float(c2/c1)
        return True
