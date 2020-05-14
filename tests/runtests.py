#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

def rosenbrock_init(x):
    x[0:len(x):2] = -1.2
    x[1:len(x):2] =  1.0
    return x

def rosenbrock_fg(x, g):
    x1 = x[0:len(x):2]
    x2 = x[1:len(x):2]
    c = 100
    t1 = 1 - x1
    t2 = x2 - x1*x1
    g[0:len(g):2] = -2*t1 - (4*c)*t2*x1
    g[1:len(g):2] = (2*c)*t2
    return sum(t1*t1) + c*sum(t2*t2)

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import numpy as np
    from optimpack.conjugategradient import nlcg
    x = rosenbrock_init(np.empty(200, 'float64'))
    x = nlcg(rosenbrock_fg, x, method='Hestenes-Stiefel', gtol=(0,1e-6),
             fmin=0.0, verb=1, maxiter=1000, maxeval=100)
    #print(x)
