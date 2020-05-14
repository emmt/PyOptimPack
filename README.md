# OptimPack for Python

This package provides optimization methods for multi-variate problems whose
objective function is differentiable.

OptimPack for Python implements line-search methods and two families of
methods to solve multi-variate optimization problems: non-linear conjugate
gradient (`nlcg`) and limited memory quasi-Newton method (`vmlmb`).  The
latter method can take into account simple bounds on the variables.  These
algorithms are a pure Python + NumPy version of the ones implemented in
[OptimPack](https://github.com/emmt/OptimPack), a C library for large
optimization problems.

This is a **WORK IN PROGRESS**, not all methods have been implemented yet.
See [tests/runtests.py](./tests/runtests.py) for examples of usage of
available methods.
