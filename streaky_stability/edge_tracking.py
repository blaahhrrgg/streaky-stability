import logging
from scipy import optimize

from . import eigensystem


class EdgeTracker:
    """Tracks the "edge" between linearly stable and unstable flows through
    numerical continuation of the neutral curve for a given streaky flow,
    u(y, z), and parameter values.

    That is, we wish to find solutions to

        F(u, alpha, Reynolds, wavelength) = 0

    where the function F is the growth rate of the most unstable mode.
    """

    def __init__(self, streak, tol=1e-3):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.streak = streak
        self.tol = tol
        self.neutral_curve = []

    def _bisect_alpha(self, alpha_lower, alpha_upper, reynolds, wavelength):
        """Finds the (neutral) value of alpha such that the most unstable mode
        has zero growth rate using the bisection method.

        The bisection method searches within the interval defined by
        alpha_lower and alpha_upper. The most unstable mode at alpha_lower and
        alpha_upper cannot have the same sign.

        Parameters
        ----------
        alpha_lower : float
            The lower bound of alpha to initialise the search
        alpha_upper : float
            The upper bound of alpha to initialise the search
        reynolds : float
            The (fixed) Reynolds number
        wavelength : float
            The (fixed) wavelength
        """

        def _problem(alpha, reynolds, wavelength):
            """Wrapper to return the most unstable growth rate"""
            out = eigensystem.EigenvalueSystem(
                alpha=alpha,
                reynolds=reynolds,
                wavelength=wavelength,
                streak=self.streak
            ).solve(n_eigs=25)

            # TODO: Ensure zero'th solution is the most unstable
            return out[0].growth_rate

        alpha_neutral = optimize.bisect(
            _problem, alpha_lower, alpha_upper, (reynolds, wavelength),
            xtol=self.tol
        )

        self.neutral_curve.append([alpha_neutral, reynolds, wavelength])

    def continue_in_reynolds(
            self, alpha, reynolds, wavelength, dalpha, dreynolds, n_iter=25):
        """Numerically continue the neutral curve by incrementing the Reynolds
        number and finding the neutral value of alpha.

        Parameters
        ----------
        alpha : float
            The initial value of alpha
        reynolds : float
            The initial value of the Reynolds number
        wavelength : float
            The initial value of the wavelength
        dalpha : float
            The increment in alpha
        dreynolds : float
            The increment in the Reynolds number
        n_iter : int, optional
            The number of iterations to perform, by default 25
        """

        for i in range(n_iter):

            alpha_lower = alpha - dalpha
            alpha_upper = alpha + dalpha

            try:
                self._bisect_alpha(
                    alpha_lower, alpha_upper, reynolds, wavelength)

                alpha, reynolds, wavelength = self.neutral_curve[-1]

                print(alpha, reynolds, wavelength)

            except Exception as e:
                self.logger.warning(e)
                # Halve step size in reynolds
                reynolds -= dreynolds
                dreynolds /= 2

            reynolds += dreynolds
