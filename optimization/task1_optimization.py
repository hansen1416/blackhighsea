from datetime import datetime
from collections import defaultdict
import time
import math

import numpy as np
import scipy.sparse
from numpy.linalg import LinAlgError, norm
from scipy.special import expit
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.linalg import cho_factor, cho_solve


#######################################################
#                                                     #
#                   OPTIMIZATION                      #
#                                                     #
#######################################################


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


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient
                on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> line_search_options = {'method': 'Armijo', 'c1': 1e-4}
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options=line_search_options)
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    stop_criterion = tolerance * (norm(oracle.grad(x_0))**2)
    
    alpha_p = None

    start_time = time.time()

    for i in range(max_iter):
        # (Oracle Call) Calculate f(x_k), ∇f(x_k), etc
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        grad_norm = norm(grad_k)

        # (Calculating the direction) Calculate the direction of descent d_k.
        d_k = - grad_k

        # (Linear search) Find the appropriate step length α_k.
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, 2 * alpha_p if alpha_p else None)

        if display:
            print("iteration {}, func_k {}, grad_k {}, grad_norm {}, x_k: {}, d_k {}, alpha {}"\
                .format(i, func_k, grad_k, grad_norm, x_k, d_k, alpha_p))

        # (Update) xk+1 ← xk + αkdk.
        x_k = x_k + d_k * alpha_k

        alpha_p = alpha_k

        if trace:
            history['x'].append(x_k)
            history['func'].append(func_k)
            history['grad_norm'].append(grad_norm)
            history['time'].append(time.time() - start_time)

        if x_k is None or alpha_k is None:
            return x_k, 'computational_error', history

        # (Stop criterion) If the stop criterion is met, then exit.
        if grad_norm**2 <= stop_criterion:
            return x_k, 'success', history

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix
                (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient
                on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    grad_0_norm = norm(oracle.grad(x_0))
    stop_criterion = tolerance * grad_0_norm**2
    
    alpha_p = None

    start_time = time.time()

    for i in range(max_iter):
        # (Oracle Call) Calculate f(x_k), ∇f(x_k), etc
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        hess_k = oracle.hess(x_k)
        grad_norm = norm(grad_k)

        # (Calculating the direction) Calculate the direction of descent d_k.
        try:
            c, low = cho_factor(hess_k)
            d_k = cho_solve((c, low), grad_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        # (Linear search) Find the appropriate step length α_k.
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)

        if display:
            print("iteration {}, func_k {}, grad_k {}, grad_norm {}, x_k: {}, d_k {}, alpha {}"\
                .format(i, func_k, grad_k, grad_norm, x_k, d_k, alpha_p))

        # (Update) xk+1 ← xk + αkdk.
        x_k = x_k + alpha_k * d_k

        alpha_p = alpha_k

        if trace:
            history['x'].append(x_k)
            history['func'].append(func_k)
            history['grad_norm'].append(grad_norm)
            history['time'].append(time.time() - start_time)

        if x_k is None or alpha_k is None or math.isinf(x_k) or math.isinf(alpha_k):
            return x_k, 'computational_error', history

        # (Stop criterion) If the stop criterion is met, then exit.
        if grad_norm**2 <= stop_criterion:
            return x_k, 'success', history

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, 'iterations_exceeded', history


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
        # return 1/self.b.shape[0] * \
        #     (np.sum(np.log(1+np.exp(-1 * self.matvec_Ax(x)))) + \
        #         self.regcoef / 2 * norm(x)**2)
        logadd = np.logaddexp(0, - self.b * self.matvec_Ax(x))
        res = np.linalg.norm(logadd, 1) / self.b.size +\
              np.linalg.norm(x, 2) ** 2 * self.regcoef / 2
        return res

    def grad(self, x):
        return self.regcoef * x - self.matvec_ATx(self.b * (expit(-self.b * self.matvec_Ax(x)))) / self.b.size

    def hess(self, x):
        tmp = expit(self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(tmp * (1 - tmp)) / self.b.size + self.regcoef * np.identity(x.size)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    if scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)

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

if __name__ == '__main__':

    def test_grad_finite_diff_1(test='logreg', A = np.diag([1,1,1]), b = np.array([1, 1, 1]), x = np.zeros(3)):
        # Quadratic function.
        if test == 'quadratic' or test == 'all':

            quadratic = QuadraticOracle(A, b)
            gfd = grad_finite_diff(quadratic.func, x)

            gd = quadratic.grad(x)

            print("A:\n", A, "\nb:\n", b, "\nx:\n", \
                x, "\ngradient result:\n", gd, "\ngradient finite difference\n", gfd)

            if not np.allclose(gd, gfd):
                print("====Wrong answer====")

        if test == 'logreg' or test == 'all':

            regcoef = 0.5

            logreg = create_log_reg_oracle(A, b, regcoef)

            gfd_lr = grad_finite_diff(logreg.func, x)

            gd_lr = logreg.grad(x)

            print("A:\n", A, "\nb:\n", b, "\nx:\n", \
                x, "\ngradient result:\n", gd_lr, "\ngradient finite difference\n", gfd_lr)

            if not np.allclose(gd_lr, gfd_lr):
                print("====Wrong answer====")

    def test_hess_finite_diff_1(test='logreg', A = np.diag([1,2,1]), b = np.array([1, 0, 1]), \
        x = np.array([2,2,3])):

        if test == 'quadratic' or test == 'all':

            # Quadratic function.
            quadratic = QuadraticOracle(A, b)

            hfd = hess_finite_diff(quadratic.func, x)
            hs = quadratic.hess(x)
            
            print("A:\n", A, "\nb:\n", b, "\nx:\n", \
                x, "\nhessian:\n", hs, "\nhessian finite difference\n", hfd)
            
            if not np.allclose(hs, hfd):
                print("====Wrong answer====")

        if test == 'logreg' or test == 'all':

            regcoef = 0.5

            logreg = create_log_reg_oracle(A, b, regcoef)

            hfd_lr = hess_finite_diff(logreg.func, x)

            hs_lr = logreg.hess(x)

            print("A:\n", A, "\nb:\n", b, "\nx:\n", \
                x, "\nhessian:\n", hs_lr, "\nhessian finite difference\n", hfd_lr)

            if not np.allclose(hs_lr, hfd_lr):
                print("====Wrong answer====")

    test_grad_finite_diff_1()

    test_hess_finite_diff_1()