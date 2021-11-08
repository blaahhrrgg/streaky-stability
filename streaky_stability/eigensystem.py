import numpy

from scipy import sparse
from scipy.sparse.linalg import eigs

from . import eig_operator
from . import eig_solution


class EigenvalueSystem(object):
    """Solver for the generalised eigenvalue system which describes the linear
    stability of a streaky base flow.

    TODO: Add more details on the discretisation scheme and matrix construction
    """

    def __init__(self, alpha, reynolds, wavelength, streak):
        self.alpha = alpha
        self.reynolds = reynolds
        self.wavelength = wavelength
        self.streak = streak

        self.u = self.streak.u_grid
        self.uy = self.streak.uy_grid
        self.uyy = self.streak.uyy_grid

        self.block_size = 6
        self.system_size = self.block_size*self.streak.domain.ny

        self.A, self.B = self._generate_system()
        self.solutions = {}

    def _generate_first_order_system(self, y_step):
        """Generates the first order system at the given y_step

        Parameters
        ----------
        y_step : int
            The index of the grid point in y

        Returns
        -------
        A : numpy.array
            The first order system for matrix A
        B : numpy.array
            The first order system for matrix B
        """
        shape = (self.block_size, self.block_size)

        A = numpy.zeros(shape=shape, dtype='complex')
        B = numpy.zeros(shape=shape, dtype='complex')

        # Some common terms
        alpha_sq = self.alpha**2
        ialphar = 1j*self.alpha*self.reynolds

        # Matrix A
        A[0, 1] = 1

        A[1, 0] = ialphar*self.u[0, y_step] + alpha_sq
        A[1, 2] = self.reynolds*self.uy[0, y_step]
        A[1, 5] = ialphar

        A[2, 0] = -1j*self.alpha

        A[3, 4] = 1

        A[4, 3] = ialphar*self.u[0, y_step] + alpha_sq

        A[5, 1] = -1j*self.alpha/self.reynolds
        A[5, 2] = -(
            1j*self.alpha*self.u[0, y_step]
            + alpha_sq/self.reynolds)

        # Matrix B
        B[1, 0] = -ialphar
        B[4, 3] = -ialphar
        B[5, 2] = 1j*self.alpha

        return A, B

    def _generate_second_order_system(self, y_step):
        """Generates the second order system at the given y_step

        Note that the second order system can be found by differentiating the
        first order system with respect to y to find that,

        b_{ij} = d/dy (a_{ij}) + sum_{l} a_{il} a_{lj}

        Parameters
        ----------
        y_step : int
            The index of the grid point in y

        Returns
        -------
        A : numpy.array
            The second order system for matrix A
        B : numpy.array
            The second order system for matrix B
        """
        A_f, B_f = self._generate_first_order_system(y_step)

        shape = self.block_size, self.block_size
        A_y = numpy.zeros(shape=shape, dtype='complex')

        A_y[1, 0] = 1j*self.alpha*self.reynolds*self.uy[0, y_step]
        A_y[1, 2] = self.reynolds*self.uyy[0, y_step]
        A_y[4, 3] = 1j*self.alpha*self.reynolds*self.uy[0, y_step]
        A_y[5, 2] = -1j*self.alpha*self.uy[0, y_step]

        A = A_y + A_f @ A_f
        B = A_f @ B_f + B_f + A_f

        return A, B

    def _generate_discretisation_step(self, y_step):
        """Generates the discretisation step by an Euler-Maclaurin scheme

        Parameters
        ----------
        y_step : int
            The index of the grid point in y

        Returns
        -------
        A_forward : numpy.array
            The forward discretisation step for matrix A
        A_backward : numpy.array
            The backward discretisation step for matrix A
        B_forward : numpy.array
            The forward discretisation step for matrix B
        B_backward : numpy.array
            The backward discretisation step for matrix B
        """
        A_first_forward, B_first_forward = \
            self._generate_first_order_system(y_step)

        A_second_forward, B_second_forward = \
            self._generate_first_order_system(y_step)

        A_first_backward, B_first_backward = \
            self._generate_first_order_system(y_step - 1)

        A_second_backward, B_second_backward = \
            self._generate_second_order_system(y_step - 1)

        # Identity
        identity = numpy.identity(n=self.block_size)

        y_step_size = self.streak.domain.y_step_size

        h1 = y_step_size / 2.0
        h2 = h1 * (y_step_size / 6.0)

        # Matrix A
        A_forward = identity - h1*A_first_forward + h2*A_second_forward
        A_backward = -identity - h1*A_first_backward - h2*A_second_backward

        # Matrix B
        B_forward = -h1*B_first_forward + h2*B_second_forward
        B_backward = -h1*B_first_backward - h2*B_second_backward

        return A_forward, A_backward, B_forward, B_backward

    def _generate_system(self):
        """Generates the matrices for the generalised eigenvalue problem

        For each y_step, the forward and backward steps of Euler-Maclaurin
        scheme are calculated and inserted into the system matrices A and B to
        form a tridigonal-block system.

        Returns
        -------
        scipy.sparse.csc_matrix : A
            The matrix A of the generalised eigenvalue problem
        scipy.sparse.csc_matrix : B
            The matrix B of the generalised eigenvalue problem
        """
        system_shape = (self.system_size, self.system_size)

        # Initialise two dictionary of keys sparse matrices for construction
        A = sparse.dok_matrix(system_shape, dtype='complex')
        B = sparse.dok_matrix(system_shape, dtype='complex')

        # Iterate through each y_step and set values into system matrices
        for yi in range(1, self.streak.domain.ny):

            A_forward, A_backward, B_forward, B_backward = \
                self._generate_discretisation_step(yi)

            # Matrix A
            A[
                (yi - 1)*self.block_size + 3: (yi - 1)*self.block_size + 6,
                (yi - 1)*self.block_size: yi*self.block_size
            ] = A_backward[3:, :]

            A[
                (yi - 1)*self.block_size + 3: (yi - 1)*self.block_size + 6,
                yi*self.block_size: (yi + 1)*self.block_size
            ] = A_forward[3:, :]

            A[
                yi*self.block_size: yi*self.block_size + 3,
                (yi - 1)*self.block_size: yi*self.block_size
            ] = A_backward[:3, :]

            A[
                yi*self.block_size: yi*self.block_size + 3,
                yi*self.block_size: (yi + 1)*self.block_size
            ] = A_forward[:3, :]

            if yi == 1:
                # Lower boundary conditions
                A[(yi-1)*self.block_size + 0, (yi-1)*self.block_size + 0] = 1
                A[(yi-1)*self.block_size + 1, (yi-1)*self.block_size + 2] = 1
                A[(yi-1)*self.block_size + 2, (yi-1)*self.block_size + 3] = 1

            if yi == self.streak.domain.ny - 1:
                # Upper boundary conditions
                A[yi*self.block_size + 3, yi*self.block_size + 0] = 1
                A[yi*self.block_size + 4, yi*self.block_size + 2] = 1
                A[yi*self.block_size + 5, yi*self.block_size + 3] = 1

            # Matrix B
            B[
                (yi - 1)*self.block_size + 3: (yi - 1)*self.block_size + 6,
                (yi - 1)*self.block_size: yi*self.block_size
            ] = B_backward[3:, :]

            B[
                (yi - 1)*self.block_size + 3: (yi - 1)*self.block_size + 6,
                yi*self.block_size: (yi + 1)*self.block_size
            ] = B_forward[3:, :]

            B[
                yi*self.block_size: yi*self.block_size + 3,
                (yi - 1)*self.block_size: yi*self.block_size
            ] = B_backward[:3, :]

            B[
                yi*self.block_size: yi*self.block_size + 3,
                yi*self.block_size: (yi + 1)*self.block_size
            ] = B_forward[:3, :]

        # Return in compressed sparse column format
        return A.tocsc(), B.tocsc()

    def solve(self, sigma=1j, n_eigs=50):
        """Solve the generalised eigenvalue system using Arnoldi methods

        Returns
        -------
        dict
            A dictionary of eigenmode solutions
        """
        # Define the linear operator to perform matrix-vector products of the
        # shifted generalised eigenvalue problem
        op = eig_operator.GeneralisedEigenvalueProblemOperator(
            self.A, -self.B, sigma)

        # Call Arnoldi methods from scipy
        evals, evecs = eigs(A=op, k=n_eigs, which='LI')

        # Shift eigenvalues back
        evals = sigma + 1. / evals

        # Unpack solutions and store by growth rate in descending order
        for i, (evalue, evec) in enumerate(sorted(
            zip(evals, evecs.T), key=lambda x: x[0].imag, reverse=True)):

            self.solutions[i] = eig_solution.Eigensolution(
                alpha=self.alpha,
                reynolds=self.reynolds,
                wavelength=self.wavelength,
                wavespeed=evalue,
                wave_u=evec[0::6],
                wave_v=evec[2::6],
                wave_w=evec[3::6],
                wave_p=evec[5::6])

        return self.solutions
