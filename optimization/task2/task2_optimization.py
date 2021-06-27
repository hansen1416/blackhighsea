from datetime import datetime
import os
import sys
import time

from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS

import numpy as np
import scipy.sparse
from numpy.linalg import LinAlgError
from scipy.special import expit
from scipy.optimize.linesearch import scalar_search_wolfe2


#######################################################
#                                                     #
#                   OPTIMIZATION                      #
#                                                     #
#######################################################


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


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient
                on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start_time = time.time()
    converge = False
    alpha_k = None

    s_trace, y_trace = deque(), deque()
    grad_k = oracle.grad(x_k)

    for num_iter in range(max_iter + 1):

        func_k = oracle.func(x_k)

        grad_norm_k = scipy.linalg.norm(grad_k)

        if trace:
            history['time'].append(time.time() - start_time)
            history['func'].append(np.copy(func_k))
            history['grad_norm'].append(np.copy(grad_norm_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display: 
            print('step', num_iter, history['time'][-1] if history else '')

        if num_iter == 0:
            eps_grad_norm_0 = np.sqrt(tolerance) * grad_norm_k
        if grad_norm_k <= eps_grad_norm_0:
            converge = True
            break

        if num_iter == max_iter: 
            break

        d_k = lbfgs_direction(grad_k, s_trace, y_trace)
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, \
            2.0 * alpha_k if alpha_k else None)

        x_k = x_k + alpha_k * d_k
        grad_k_1 = np.copy(grad_k)
        grad_k = oracle.grad(x_k)

        if memory_size > 0:
            # limited memory, save storage
            if len(s_trace) == memory_size:
                s_trace.popleft()
                y_trace.popleft()
            s_trace.append(alpha_k * d_k)
            y_trace.append(grad_k - grad_k_1)


    return x_k, 'success' if converge else 'iterations_exceeded', history

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, 'success', history

def lbfgs_direction(grad, s_trace, y_trace):
    """
    H_{k+1} = V^T_k H_kV_k + ρ_k s_k s^T_k
    ρ_k = \frac{1}{y^T_k s_k}, V_k = I − ρ_k y_k s^T_k,
    s_k = x_{k+1} − x_k , y_k =  ∇ f_{k+1} − ∇ f_k .
    """
    d = -grad

    if not s_trace:
        return d

    mus = []
    for s, y in zip(reversed(s_trace), reversed(y_trace)):
        mu = np.dot(s, d) / np.dot(s, y)
        mus.append(mu)
        d -= mu * y

    d *= np.dot(s_trace[-1], y_trace[-1]) / np.dot(y_trace[-1], y_trace[-1])

    for s, y, mu in zip(s_trace, y_trace, reversed(mus)):
        beta = np.dot(y, d) / np.dot(s, y)
        d += (mu - beta) * s

    return d
#######################################################
#                                                     #
#                     ORACLES                         #
#                                                     #
#######################################################


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


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        logadd = np.logaddexp(0, - self.b * self.matvec_Ax(x))
        res = np.linalg.norm(logadd, 1) / self.b.size +\
              np.linalg.norm(x, 2) ** 2 * self.regcoef / 2
        return res

    def grad(self, x):
        # TODO: Implement
        return self.regcoef * x - self.matvec_ATx(self.b * (expit(-self.b * self.matvec_Ax(x)))) \
            / self.b.size

    def hess(self, x):
        # TODO: Implement
        tmp = expit(self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(tmp * (1 - tmp)) / self.b.size + self.regcoef * np.identity(x.size)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    def matvec_Ax(x):
        # TODO: implement proper matrix-vector multiplication
        return A.dot(x)

    def matvec_ATx(x):
        # TODO: implement proper martix-vector multiplication
        return A.T.dot(x)

    def matmat_ATsA(s):
        # TODO: Implement
        if scipy.sparse.issparse(A):
            return matvec_ATx(matvec_ATx(scipy.sparse.diags(s)).T)
        return np.dot(matvec_ATx(np.diag(s)), A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    x = np.array(x)
    e = np.identity(x.shape[0])
    res = np.zeros(x.shape)

    f = func(x)

    for i in range(x.shape[0]):
        res[i] = func(x + e[i] * eps) - f

    return res / eps


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    n = x.shape[0]

    output = np.matrix(np.zeros(n*n))
    output = output.reshape(n,n)

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            ej = np.zeros(n)
            ej[j] = 1
            f1 = func(x + eps * ei + eps * ej)
            f2 = func(x + eps * ei)
            f3 = func(x + eps * ej)
            f4 = func(x)
            numdiff = (f1-f2-f3+f4)/(eps*eps)
            output[i,j] = numdiff

    return output