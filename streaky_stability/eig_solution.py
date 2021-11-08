import dataclasses
import numpy


@dataclasses.dataclass
class Eigensolution:
    """Class for storing solutions to the eigenvalue problem"""
    alpha: float
    reynolds: float
    wavelength: float
    wavespeed: complex
    wave_u: numpy.array
    wave_v: numpy.array
    wave_w: numpy.array
    wave_p: numpy.array

    @property
    def growth_rate(self):
        """The growth rate of the eigenmode."""
        return self.alpha*self.wavespeed.imag
