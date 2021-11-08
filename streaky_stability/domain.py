import numpy


class Domain(object):
    """Defines a grid of uniformly spaced points"""

    def __init__(self, y_lower, y_upper, ny, z_lower, z_upper, nz):
        self.y_lower = y_lower
        self.y_upper = y_upper
        self.z_lower = z_lower
        self.z_upper = z_upper
        self.ny = ny
        self.nz = nz
        self.y = numpy.linspace(y_lower, y_upper, ny)
        self.z = numpy.linspace(z_lower, z_upper, nz)
        self.Y, self.Z = numpy.meshgrid(self.y, self.z)

    @property
    def y_step_size(self):
        """The step size in y"""
        return (self.y_upper - self.y_lower) / self.ny

    @property
    def z_step_size(self):
        """The step size in z"""
        return (self.z_upper - self.z_lower) / self.nz
