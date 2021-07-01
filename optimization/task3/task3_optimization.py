from collections import defaultdict
import datetime
import sys
from time import time

import numpy as np
import scipy
from numpy.linalg import norm, solve
from scipy.special import expit
from scipy.optimize.linesearch import scalar_search_wolfe2


assert sys.version_info >= (3, 6), (
    "Please use Python3.6+ to make this assignment"
)

class LineSearchTool:
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        dphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)
        phi0, dphi0 = phi(0), dphi(0)
        
        if self._method == 'Wolfe':
            # \phi^{'}(\alpha_k) \ge c_2 \phi^{'}(0); c_2 \in (0,1)
            alpha_k, *_ = scalar_search_wolfe2(phi, dphi, phi0, None, dphi0, c1=self.c1, c2=self.c2)
            if alpha_k is not None:
                return alpha_k
            else:
                self._method = 'Armijo'

        if self._method == 'Armijo':
            # \phi(\alpha_k) \le \phi(0) + c_1 \alpha_k \phi^{'}(0); c_1 \in (0,1)
            alpha_k = previous_alpha if previous_alpha is not None else self.alpha_0

            while phi(alpha_k) > (phi0 + alpha_k * dphi0 * self.c1):
                alpha_k /= 2

            return alpha_k

        elif self._method == 'Constant':
            return self.c


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()

def matvec_Ax(A, x):
    return A.dot(x)

def matvec_ATx(A, x):
    return A.T.dot(x)

def eval_duality_gap(A, x, b, reg_coef, lasso_duality_gap):
    if lasso_duality_gap is None:
        return None
    Ax_b = matvec_Ax(A, x) - b
    ATAx_b = matvec_ATx(A, Ax_b)
    return lasso_duality_gap(x, Ax_b, ATAx_b, b, reg_coef)

class BaseSmoothOracle:
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

class BarrierOracle(BaseSmoothOracle):
    def __init__(self, t, n, A, b, reg_coef):
        self.t = t
        self.n = n
        self.A = A
        self.b = b
        self.reg_coef = reg_coef

    def func(self, x):
        Ax_b = matvec_Ax(self.A, x[:self.n]) - self.b
        regression = 0.5 * np.dot(Ax_b, Ax_b)
        regularization = self.reg_coef * scipy.linalg.norm(x[self.n:], ord=1)

        @np.vectorize
        def fixed_log(x):
            return np.log(x) if x > 0 else np.inf

        return self.t * (regression + regularization) - \
            np.sum(fixed_log(x[self.n:] + x[:self.n]) + fixed_log(x[self.n:] - x[:self.n]))

    def grad(self, x):
        regression = matvec_ATx(self.A, matvec_Ax(self.A, x[:self.n]) - self.b)
        regularization = np.full(self.n, self.reg_coef)

        left = 1.0 / (self.t * x[self.n:] + self.t * x[:self.n])
        right = 1.0 / (self.t * x[self.n:] - self.t * x[:self.n])

        return self.t * np.concatenate((regression - left + right, regularization - left - right))


def newton_barrier_lasso(oracle, tATA, x_0, u_0, tolerance=1e-5, max_iter=100, theta=0.99, line_search_options=None):
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)
    u_k = np.copy(u_0).astype(np.float64)
    n = x_k.size

    converge = False

    for num_iter in range(max_iter + 1):
        x = np.concatenate((x_k, u_k))

        if x.dtype != float or np.isinf(x).any() or np.isnan(x).any():
            return (x_k, u_k), 'computational_error'

        try:
            grad_k = oracle.grad(x)
        except Exception:
            return (x_k, u_k), 'computational_error'

        grad_x = grad_k[:n]
        grad_u = grad_k[n:]

        if grad_k.dtype != float or np.isinf(grad_k).any() or np.isnan(grad_k).any():
            return (x_k, u_k), 'computational_error'

        grad_norm_k = scipy.linalg.norm(grad_k)

        if num_iter == 0:
            eps_grad_norm_0 = np.sqrt(tolerance) * grad_norm_k
        if grad_norm_k <= eps_grad_norm_0:
            converge = True
            break

        if num_iter == max_iter: break

        alpha = 1.0 / (u_k + x_k)**2
        beta = 1.0 / (u_k - x_k)**2

        A = tATA + scipy.sparse.diags(((alpha + beta)**2 - (alpha - beta)**2) / (alpha + beta))
        b = grad_u * (alpha - beta) / (alpha + beta) - grad_x

        try:
            if scipy.sparse.issparse(A):
                d_x = scipy.sparse.linalg.spsolve(A, b)
            else:
                c, lower = scipy.linalg.cho_factor(A, overwrite_a=True)
                d_x = scipy.linalg.cho_solve((c, lower), b, overwrite_b=True)
        except Exception:
            return (x_k, u_k), 'newton_direction_error'

        d_u = -(grad_u + d_x * (alpha - beta)) / (alpha + beta)

        d_1 = d_x - d_u
        idxs_1 = d_1 > 0.0
        d_2 = -d_x - d_u
        idxs_2 = d_2 > 0.0

        alpha_0 = np.concatenate([
            [1.0],
            theta * (u_k - x_k)[idxs_1] / d_1[idxs_1],
            theta * (u_k + x_k)[idxs_2] / d_2[idxs_2]
        ]).min()

        try:
            alpha_k = line_search_tool.line_search(oracle, x, np.concatenate((d_x, d_u)), alpha_0)
        except Exception:
            return (x_k, u_k), 'computational_error'

        x_k = x_k + alpha_k * d_x
        u_k = u_k + alpha_k * d_u

    return (x_k, u_k), 'success' if converge else 'iterations_exceeded'

def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).astype(np.float64)
    u_k = np.copy(u_0).astype(np.float64)
    t_k = np.float64(t_0)

    AT = A.T
    ATA = AT.dot(A)

    start_time = time()
    converge = False
    n = x_k.size

    for num_iter in range(max_iter + 1):
        duality_gap = eval_duality_gap(A, x_k, b, reg_coef, lasso_duality_gap)

        if duality_gap is None:
            converge = True

        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(0.5 * matvec_Ax(A, x_k) - b + reg_coef * scipy.linalg.norm(x_k, ord=1))
            history['duality_gap'].append(duality_gap)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: print('step: {}, time consumed {}'.\
            format(num_iter, history['time'][-1] if history else ''))

        if duality_gap is not None and duality_gap <= tolerance:
            converge = True
            break

        if t_k > np.finfo(np.float64).max:
            return (x_k, u_k), 'computational_error', history

        if num_iter == max_iter: break

        (x_k, u_k), newton_message = newton_barrier_lasso(
            BarrierOracle(t_k, n, A, b, reg_coef), t_k * ATA, x_k, u_k,
            theta=0.99, line_search_options={'method': 'Armijo', 'c1': c1},
            tolerance=tolerance_inner,
            max_iter=max_iter_inner,
        )

        t_k *= gamma

    return (x_k, u_k), 'success' if converge else 'iterations_exceeded', history


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    norm = scipy.linalg.norm(ATAx_b, np.inf)
    # µ(x) := min {1, λ / ||AT (Ax − b)||∞ } (Ax − b).
    coef = min(1.0, regcoef / scipy.linalg.norm(ATAx_b, np.inf)) if norm else 1.0
    mu = coef * Ax_b
    return 0.5 * np.dot(Ax_b, Ax_b) + regcoef * scipy.linalg.norm(x, ord=1) + 0.5 * np.dot(mu, mu) + np.dot(b, mu)
