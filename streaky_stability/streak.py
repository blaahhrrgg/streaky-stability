import abc
import numpy

from scipy import interpolate


class Streak(abc.ABC):
    """Class to define the streaky base flow
    """

    @abc.abstractmethod
    def u(self, y, z):
        """Method to return the value of the velocity field u at point (y,z)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def u_grid(self, y, z):
        """Returns a grid of velocity field values defined by the domain
        """
        raise NotImplementedError

    @property
    def uy_grid(self):
        """Returns second-order finite difference estimate of the first
        derivative of the streak with respect to the y-coordinate over the
        domain grid

        Returns
        -------
        numpy.array
            The second-order finite difference approximation of the first
            derivative with respect to the wall-normal y coordinate
        """
        u = self.u_grid
        uy = numpy.zeros(shape=u.shape)
        dy = self.domain.y_step_size

        for i in range(self.domain.nz):
            for j in range(self.domain.ny):

                if j == 0:
                    uy[i, j] = (-u[i, j+2] + 4*u[i, j+1] - 3*u[i, j])

                elif j == self.domain.ny - 1:
                    uy[i, j] = (3*u[i, j] - 4*u[i, j-1] + u[i, j-2])

                else:
                    uy[i, j] = (u[i, j+1] - u[i, j-1])

        return (1./(2*dy))*uy

    @property
    def uyy_grid(self):
        """Returns second-order finite difference estimate of the second
        derivative of the streak with respect to the y-coordinate over the
        domain grid

        Returns
        -------
        numpy.array
            The second-order finite difference approximation of the second
            derivative with respect to the wall-normal y coordinate
        """
        u = self.u_grid
        uyy = numpy.zeros(shape=u.shape)
        dy = self.domain.y_step_size

        for i in range(self.domain.nz):
            for j in range(self.domain.ny):

                if j == 0:
                    uyy[i, j] = \
                        (-u[i, j+3] + 4*u[i, j+2] - 5*u[i, j+1] + 2*u[i, j])

                elif j == self.domain.ny - 1:
                    uyy[i, j] = \
                        (2*u[i, j] - 5*u[i, j-1] + 4*u[i, j-2] - u[i, j-3])

                else:
                    uyy[i, j] = (u[i, j+1] - 2*u[i, j] + u[i, j-1])

        return (1./(dy**2))*uyy


class GridValuesStreak(Streak):
    """Specify the streak by providing a grid of values"""

    def __init__(self, values, domain):
        self.values = values
        self.domain = domain
        self.interp = interpolate.interp2d(
            self.domain.Z,
            self.domain.Y,
            self.u_grid,
            kind='linear')
        self.u_fourier = numpy.fft.fft(self.u_grid, axis=0)

    def u(self, y, z):
        return self.interp(z, y)

    @property
    def u_grid(self):
        return self.values


class FunctionalFormStreak(Streak):
    """Specify the streak by providing a function"""

    def __init__(self, f, domain):
        self.f = f
        self.domain = domain

    def u(self, y, z):
        return self.f(y, z)

    @property
    def u_grid(self):
        return self.f(self.domain.Y, self.domain.Z)
