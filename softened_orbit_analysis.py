import numpy as np
from starforge_mult_search.code.starforge_constants import GN


def PotentialKernel(r, h):
    """
    Returns the equivalent of -1/r for a cubic-spline mass distribution of compact support radius h. Used to calculate the softened gravitational potential.

    Arguments:
    r - radius
    h - softening
    """
    if h == 0.0:
        return -1.0 / r
    hinv = 1.0 / h
    q = r * hinv
    if q <= 0.5:
        return (-2.8 + q * q * (5.33333333333333333 + q * q * (6.4 * q - 9.6))) * hinv
    elif q <= 1:
        return (
            -3.2
            + 0.066666666666666666666 / q
            + q
            * q
            * (
                10.666666666666666666666
                + q * (-16.0 + q * (9.6 - 2.1333333333333333333333 * q))
            )
        ) * hinv
    else:
        return -1.0 / r


def phi_softened(r, mtot, eps):
    return GN * mtot * PotentialKernel(r, eps)


def eff_pot(r, L2, mtot, eps):
    return 0.5 * L2 / (r * r) + phi_softened(r, mtot, eps)


def root_function(r, E, L2, mtot, eps):
    return E - eff_pot(r, L2, mtot, eps)


def bisect_root(E, L2, mtot, eps, a, b):
    rtol = 1e-8
    maxiter = 300

    fa = root_function(a, E, L2, mtot, eps)
    fb = root_function(b, E, L2, mtot, eps)

    if fa * fb > 0:
        return np.nan  # No bracketed root

    log_a = np.log10(a)
    log_b = np.log10(b)

    for _ in range(maxiter):
        log_mid = 0.5 * (log_a + log_b)
        mid = 10.0**log_mid
        fc = root_function(mid, E, L2, mtot, eps)

        if abs(log_b - log_a) < rtol:
            return mid

        if fa * fc < 0.0:
            log_b = log_mid
            fb = fc
        else:
            log_a = log_mid
            fa = fc

    return 10.0**log_mid  # return last midpoint if no convergence


def get_peri_softened(x, y, z, vx, vy, vz, mtot, eps):
    r0 = np.sqrt(x * x + y * y + z * z)
    v2 = vx * vx + vy * vy + vz * vz
    phi = phi_softened(r0, mtot, eps)
    E = 0.5 * v2 + phi

    # Angular momentum squared
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    L2 = Lx * Lx + Ly * Ly + Lz * Lz

    rmin = 1e-20  # Avoid divide-by-zero
    rmax = r0  # Assume current sep is outside pericenter

    return bisect_root(E, L2, mtot, eps, rmin, rmax)


def get_apo_softened(x, y, z, vx, vy, vz, mtot, eps):
    r0 = np.sqrt(x * x + y * y + z * z)
    v2 = vx * vx + vy * vy + vz * vz
    phi = phi_softened(r0, mtot, eps)
    E = 0.5 * v2 + phi

    # Angular momentum squared
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    L2 = Lx * Lx + Ly * Ly + Lz * Lz

    rmin = r0
    rmax = 10 * r0

    root = bisect_root(E, L2, mtot, eps, rmin, rmax)
    while np.isnan(root):
        rmax *= 10
        root = bisect_root(E, L2, mtot, eps, rmin, rmax)
    return root
