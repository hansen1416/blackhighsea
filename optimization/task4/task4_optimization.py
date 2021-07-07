from collections import defaultdict
from time import time

import numpy as np
import scipy
from scipy.special import expit


#######################################################
#                                                     #
#                   OPTIMIZATION                      #
#                                                     #
#######################################################


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).astype(np.float64)
    f_k = oracle.func(x_k)

    start_time = time()
    converge = False
    L_k = L_0
    last_nesterov_num_iterations = 0

    for num_iter in range(max_iter + 1):
        duality_gap = oracle.duality_gap(x_k) if hasattr(oracle, 'duality_gap') else None

        if duality_gap is None:
            converge = True

        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(f_k)
            history['duality_gap'].append(duality_gap)
            history['nesterov_num_iterations'].append(last_nesterov_num_iterations)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step', history['time'][-1] if history else '')

        if duality_gap is not None and duality_gap <= tolerance:
            converge = True
            break

        if num_iter == max_iter: break

        _f_k = oracle._f.func(x_k)
        grad_k = oracle.grad(x_k)

        nesterov_converge = False
        last_nesterov_num_iterations = 0
        while not nesterov_converge:
            last_nesterov_num_iterations += 1

            def m(y, L, _h_y):
                if _h_y is None: _h_y = oracle._h.func(y)
                return _f_k + np.dot(grad_k, y - x_k) + L / 2.0 * np.dot(y - x_k, y - x_k) + _h_y

            alpha = 1.0 / L_k
            y = oracle.prox(x_k - alpha * grad_k, alpha)
            _f_y = oracle._f.func(y)
            _h_y = oracle._h.func(y)
            f_y = _f_y + _h_y

            if f_y <= m(y, L_k, _h_y):
                nesterov_converge = True
            else:
                L_k *= 2.0

        x_k, f_k = y, f_y
        L_k = max(L_0, L_k / 2.0)

    return x_k, 'success' if converge else 'iterations_exceeded', history


#######################################################
#                                                     #
#                     ORACLES                         #
#                                                     #
#######################################################


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')

class BaseNonsmoothConvexOracle(object):
    """
    Base class for implementation of oracle for nonsmooth convex function.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def subgrad(self, x):
        """
        Computes arbitrary subgradient vector at point x.
        """
        raise NotImplementedError('Subgrad is not implemented.')

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        return None


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        raise NotImplementedError('Duality gap is not implemented.')


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    # TODO: implement.
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        Ax_b = self.matvec_Ax(x) - self.b
        return 0.5 * np.dot(Ax_b, Ax_b)

    def grad(self, x):
        return self.matvec_ATx(self.matvec_Ax(x) - self.b)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    # TODO: implement.
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * scipy.linalg.norm(x, ord=1)

    def subgrad(self, x):
        return self.regcoef * np.sign(x)

    def prox(self, x, alpha):
        alpha_regcoef = alpha * self.regcoef

        @np.vectorize
        def prox(x_i):
            if x_i < -alpha_regcoef:
                return x_i + alpha_regcoef
            elif x_i > alpha_regcoef:
                return x_i - alpha_regcoef
            else:
                return 0.0

        return prox(x)

class LassoNonsmoothOracle(BaseNonsmoothConvexOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    # TODO: implement.
    def __init__(self, matvec_Ax, matvec_ATx, b, regcoef):
        self._f = LeastSquaresOracle(matvec_Ax, matvec_ATx, b)
        self._h = L1RegOracle(regcoef)
        self.regcoef = regcoef

    def func(self, x):
        return self._f.func(x) + self._h.func(x)

    def subgrad(self, x):
        return self._f.grad(x) + self._h.subgrad(x)

    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        ATAx_b = self._f.matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._f.b, self._h.regcoef)


class LassoProxOracle(BaseCompositeOracle, LassoNonsmoothOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """

    def __init__(self, f, h):
        super(LassoProxOracle, self).__init__(f, h)
        
    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        ATAx_b = self._f.matvec_ATx(Ax_b)
        return lasso_duality_gap(x, Ax_b, ATAx_b, self._f.b, self._h.regcoef)    


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """

    norm = scipy.linalg.norm(ATAx_b, np.inf)
    coef = min(1.0, regcoef / scipy.linalg.norm(ATAx_b, np.inf)) if norm else 1.0
    mu = coef * Ax_b
    return 0.5 * np.dot(Ax_b, Ax_b) + regcoef * scipy.linalg.norm(x, ord=1) + 0.5 * np.dot(mu, mu) + np.dot(b, mu)

def create_lasso_prox_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))