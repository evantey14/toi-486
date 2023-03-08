import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

# Suppress scientific notation
np.set_printoptions(suppress=True)


def trapezoid_model(t, per, epo, dep, qtran, qin, zpt):
    """
    Trapezoid model used to represent a transit. Assumes magnitudes (transit in positive direction).

    Parameters
    ----------
    t: array_like
        BJD observation times.
    per: float
        Orbital period in days.
    epo: float
        Transit epoch in BJD.
    dep: float
        Transit depth.
    qtran: float
        Transit duration divided by orbital period.
    qin: float
        Transit ingress duration divided by transit duration.
    zpt: float
        Out-of-transit (zero-point) magnitude.

    Returns
    -------
    model: array_like
        Trapezoid model values defined at each time.
    """
    phase = np.abs(phasefold(t, per, epo))
    transit = np.zeros(len(phase))
    qflat = qtran * (1 - qin * 2.0)
    transit[phase <= qflat / 2.0] = dep
    in_eg = (phase > qflat / 2.0) & (phase <= qtran / 2.0)
    transit[in_eg] = dep - ((dep / ((qtran - qflat) / 2.0)) * (phase[in_eg] - qflat / 2.0))
    model = zpt + transit
    return model

def fit_trapezoid_model_fixed(t, y, dy, per, epo, dep, qtran, zpt):
    """
    Fit trapezoid model to light curve with orbital period fixed.

    Parameters
    ----------
    t: array_like
        BJD observation times.
    y: array_like
        Light curve magnitude values.
    dy: array_like
        Light curve magnitude uncertainties.
    per: float
        Orbital period in days.
    epo: float
        Initial guess of transit epoch in BJD.
    dep: float
        Initial guess of transit depth.
    qtran: float
        Initial guess of transit duration divided by orbital period.
    zpt: float
        Initial guess of out-of-transit (zero-point) magnitude.

    Returns
    -------
    popt: array_like
        Trapezoid model fitted values (epo, dep, q, qin, zpt).
    """
    dur = per * qtran
    popt, pcov = curve_fit(
        lambda t, epo, dep, qtran, qin, zpt: trapezoid_model(t, per, epo, dep, qtran, qin, zpt),
        t,
        y,
        p0=[epo, dep, qtran, 0.25, zpt],
        sigma=dy,
        bounds=([epo - 0.5 * dur, -1, 0, 0, 0], [epo + 0.5 * dur, 0, 1, 0.5, 2]),
    )
    return popt, pcov


def phasefold(t, per, epo):
    phase = np.mod(t - epo, per) / per
    phase[phase > 0.5] -= 1
    return phase

"""
Print utils
"""

def get_stats(samples, sigma=1):
    sigma_map = {
        1: 68.27,
        2: 95.45,
        3: 99.73,
    }
    median = np.nanmedian(samples)
    median_minus_one_sigma, median_plus_one_sigma = np.nanpercentile(
        samples, [100 - sigma_map[sigma], sigma_map[sigma]]
    )
    return median, median_minus_one_sigma, median_plus_one_sigma


def print_stats(name, samples, n=None, sigma=1):
    median, median_minus_one_sigma, median_plus_one_sigma = get_stats(samples, sigma)

    if n is None:
        n = max(
            get_rounding_digits(median - median_minus_one_sigma),
            get_rounding_digits(median_plus_one_sigma - median),
        )

    print(
        "{name:10}{value:10.{n}f} [{lower:10.{n}f}, {upper:10.{n}f}]".format(
            name=name,
            value=median,
            lower=median_minus_one_sigma - median,
            upper=median_plus_one_sigma - median,
            n=n,
        )
    )


def print_latex(prefix, name, value, error=None, n=None):
    if isinstance(value, str):
        print(string_template.format(name=prefix+name, value=value))
    elif error is None:
        print_n = n if n > 0 else 0
        print(value_template.format(name=prefix+name, value=round(value, n), n=print_n))
    else:
        if isinstance(error, list) and len(error) == 2:
            lower, upper = error
            if n is None:
                lower_n, upper_n = get_rounding_digits(lower), get_rounding_digits(upper)
                n = max(lower_n, upper_n)
            print_n = n if n > 0 else 0
            print(value_errors_template.format(
                name=prefix+name,
                value=round(value, n),
                lower=round(lower, n),
                upper=round(upper, n),
                n=print_n,
            ))
        else:
            if n is None:
                n = get_rounding_digits(error)
            print_n = n if n > 0 else 0
            print(value_error_template.format(
                name=prefix+name,
                value=round(value, n),
                error=round(error, n),
                n=print_n,
            ))


def rnd(value, n=None):
    if value == 0:
        return 0
    if n is None:
        n = get_rounding_digits(value)
    return np.round(value, n)


def get_rounding_digits(value):
    """Get the number of digits value should be rounded to assuming 2 sig figs.

    Returns:
        int, to be used in np.round(). i.e. rounding to the...
            tens place -> -1
            ones place -> 0
            tenths place -> 1
    """
    if value == 0 or np.isnan(value):
        raise ValueError("Cannot round 0 or nan")
    if abs(value) < 1:
        return -(int(np.log10(abs(value))) - 1) + 1
    else:
        return -(int(np.log10(abs(value)))) + 1



"""
Plot utils
"""

def plot_binned(time, flux, period, epoch, bins=None, **kwargs):
    if bins is None:
        plt.scatter(phasefold(time, period, epoch), flux, marker=".", **kwargs)
    else:
        binned_flux, bin_edges, _ = binned_statistic(
            phasefold(time, period, epoch),
            flux,
            statistic=lambda x: np.nanmedian(x),
            bins=bins,
            range=(-.5, .5)
        )
        binned_phase = bin_edges[:-1] + np.diff(bin_edges[:2])
        plt.scatter(
            binned_phase, binned_flux, marker=".", **kwargs
        )


"""
Calculation utils
"""
def get_a(period, m_star):
    """Return semi-major axis [AU] given period [days] and stellar mass [M_sun]."""
    G = 2.959e-4  # au^3 / m_sun / day^2
    return (period ** 2 * G * m_star / 4 / np.pi ** 2) ** (1 / 3)  # in AU


def get_radius(ror, r_star):
    return ror * r_star * 109.1


def get_mass(ror, r_star):
    # Empirical relation for sub-neptunes
    # from Wolfgang et al. 2016 https://arxiv.org/pdf/1504.07557.pdf
    return np.random.normal(2.7 * get_radius(ror, r_star) ** 1.3, 1.9)


def get_teq(period, r_star, t_star, m_star):
    """Return equilibrium temperature [K] given period [days] and stellar params [solar units]."""
    return t_star * np.sqrt(r_star / 2 / get_a(period, m_star) / 215)


def get_tsm(period, ror, m_J, r_star, t_star, m_star):
    """Return TSM for a small Neptune given period [days], transit depth, J
    magnitude, and stellar params [solar units].
    """
    # Transmission spectroscopy metric (https://arxiv.org/pdf/1805.03671.pdf)
    # TSM \propto r_p^3 t_eq / m_p / r_*^2 * 10^(-m_J / 5)
    # for small subneptunes, want TSM > 50 (see natalia's TOI paper)
    r_p = get_radius(ror, r_star)
    m_p = get_mass(ror, r_star)
    t_eq = get_teq(period, r_star, t_star, m_star)
    scale_factor = 1.26  # Table 1 from paper, for r_earth between 1.5 and 2.75
    return scale_factor * r_p ** 3 * t_eq / m_p / r_star ** 2 * 10 ** (-m_J / 5)


def get_insolation(period, r_star, t_star, m_star):
    """Return insolation [S_earth] given period and stellar params [solar units]."""
    SB = 5.67e-5  # erg / cm^2 / s / K^4
    return SB * (r_star / get_a(period, m_star) / 215) ** 2 * t_star ** 4 / 1.361e6


def get_aor(period, r_star, m_star):
    return get_a(period, m_star) / r_star * 215  # 215 solar radii == 1 AU


def get_inclination(period, b, r_star, m_star):
    aor = get_aor(period, r_star, m_star)
    return np.arccos(b / aor) * 57.2958  # convert to degrees


def get_k(period, ror, b, r_star, m_star):
    # http://exoplanets.astro.yale.edu/workshop/EPRV/Bibliography_files/Radial_Velocity.pdf
    i = get_inclination(period, b, r_star, m_star)
    m_planet = get_mass(ror, r_star)
    return (
        28.4329
        * m_planet
        * 0.00314558
        * np.sin(i / 57.2958)
        * (m_star + 3.00273e-6 * m_planet) ** (-2 / 3)
        * (period / 365.25) ** (-1 / 3)
    )


def get_duration(period, ror, b, r_star, m_star):
    # https://www.astro.ex.ac.uk/people/alapini/Publications/PhD_chap1.pdf
    return (
        24
        * period
        / np.pi
        * np.arcsin(
            get_aor(period, r_star, m_star) ** (-1)
            * (1 / np.sin(get_inclination(period, b, r_star, m_star) / 57.2958))
            * np.sqrt((1 + ror) ** 2 - b ** 2)
        )
    )


def get_transit_shape(ror, b):
    return np.sqrt(((1 - ror) ** 2 - b ** 2) / ((1 + ror) ** 2 - b ** 2))


def get_rho(period, duration, depth, transit_shape):
    # following 1.14 from aude thesis
    G = 2942 # Rsun^3 / Msun / day^2, so return is in solar densities
    # this has some assumptions so i should check them
    return 32 * period * depth**0.75 / G / 3.14159 / duration**3 / (1 - transit_shape**2)**1.5


def get_transit_shape(ror, b):
    # Eq 1.4 from https://www.astro.ex.ac.uk/people/alapini/Publications/PhD_chap1.pdf
    return np.sqrt(((1 - ror) ** 2 - b ** 2) / ((1 + ror) ** 2 - b ** 2))
