from scipy.sparse.linalg import LinearOperator, splu


class GeneralisedEigenvalueProblemOperator(LinearOperator):
    """Matrix-vector product operator for shifted generalised eigenvalue problem

    For a generalised eigenvalue problem of the form,

        A x = lambda B x

    The shift and invert methodology involves introducing a shift sigma such
    that the problem

        (A - sigma B)^(-1) B x = mu x

    where mu = 1 / (lambda - sigma) is considered. This transformation is
    effective for finding eigenvalues near sigma since the eigenvalues mu 
    which are of largest magnitude correspond to eigenvalues lambda of the
    original problem that are nearest to the shift sigma in absolute value.

    For Arnoldi methods, it is sufficient to provide a methodology to compute
    the matrix vector product w <- A v of an eigenvalue problem A x = lambda x
    in order to build the Krylov subspace.

    In the case of a generalised eigenvalue problem with shift sigma, this can
    be achieved through two steps,

        1. Performing the matrix vector multiplication z <- B w
        2. Solving the linear system (A - sigma B) v = z

    For efficiency, the sparse LU factorisation of (A - sigma B) is performed
    at instantiation such that step 2 is trivial when repeated calls are made.

    Parameters
    ----------
    LinearOperator : scipy.sparse.linalg.LinearOperator
    """

    def __init__(self, A, B, sigma):
        self.A = A
        self.B = B
        self.sigma = sigma
        self.shape = A.shape
        self.dtype = A.dtype

        # Perform sparse LU factorisation
        self.lu = splu(self.A - self.sigma*self.B)

    def _matvec(self, x):
        return self.lu.solve(self.B @ x)

    def _rmatvec(self, x):
        return self.lu.solve(self.B @ x)
