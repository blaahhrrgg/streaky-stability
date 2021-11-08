import numpy
import pytest

import streaky_stability


def test_unstable_mode():
    """
    For plane Poiseuille flow with (alpha, Reynolds) = (1.0, 10000), the most
    unstable eigenvalue equals,

            0.23752649 + 0.00373967i

    See:
        * Orszag, S. (1971). Accurate solution of the Orr–Sommerfeld stability
        equation. Journal of Fluid Mechanics, 50(4), 689-703.
        doi:10.1017/S0022112071002842
    """

    ny = 2**12

    domain = streaky_stability.domain.Domain(-1, 1, ny, -numpy.pi, numpy.pi, 3)

    streak = streaky_stability.streak.FunctionalFormStreak(
        lambda y, z: 1 - y**2, domain)

    sys = streaky_stability.eigensystem.EigenvalueSystem(
        alpha=1.0,
        reynolds=10000,
        wavelength=1.0,
        streak=streak
    )

    out = sys.solve(n_eigs=25)
    c_real = out[0].wavespeed.real
    c_imag = out[0].wavespeed.imag

    numpy.testing.assert_almost_equal(c_real, 0.23752649, decimal=4)
    numpy.testing.assert_almost_equal(c_imag, 0.00373967, decimal=4)


def test_convergence():
    """
    Test as the number of steps in y is increased the most unstable mode
    converges to the expected value
    """
    expected = 0.23752649 + 0.00373967*1j
    rel_errors = []

    for i in range(6, 14):

        # Arrange
        ny = 2**i
        domain = streaky_stability.domain.Domain(
            -1, 1, ny, -numpy.pi, numpy.pi, 3)

        streak = streaky_stability.streak.FunctionalFormStreak(
            lambda y, z: 1 - y**2, domain)

        sys = streaky_stability.eigensystem.EigenvalueSystem(
            alpha=1.0,
            reynolds=10000,
            wavelength=1.0,
            streak=streak
        )

        out = sys.solve(n_eigs=25)

        actual = out[0].wavespeed

        rel_errors.append(numpy.abs(actual - expected) / numpy.abs(expected))

    # Check relative error decreases as number of y_steps increases
    assert numpy.all(numpy.diff(rel_errors) < 0)
