######################################################
# ephemeris
#
# Manuel F. Diaz Ramos
#
# Functions to compute several kinds of ephemeris.
######################################################

import numpy as np

# Meeus' coefficients for computing orbital elements of the earth (output will be in deg and AU)
a_L_earth = np.array([100.466449, 35999.3728519,-0.00000568,0.0])                    # Mean Longitude coefficients
a_a_earth = np.array([1.000001018,0.0,0.0,0.0])                                      # Semimajor axis coefficients
a_e_earth = np.array([0.01670862,-0.000042037, -0.0000001236, 0.00000000004])        # Eccentricity coefficients
a_i_earth = np.array([0.0, 0.0130546,-0.00000931, -0.000000034])                     # Inclination coefficients
a_raan_earth = np.array([174.873174, -0.2410908, 0.00004067, -0.000001327])          # RAAN coefficients
a_longPeriapse_earth = np.array([102.937348, 0.3225557, 0.00015026, 0.000000478])    # Longitude of periapse coefficients

def ephemerisMeeus(a, T):
    """
    Meeus algorithm to compute orbital elements.
    :param a: [1-dimensional numpy array] 4 coefficients
    :param T: Julian centuries from J2000.
    :return: The estimated orbital element.
    """
    element = a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3
    return element

def computeOrbitalElementsMeeus(a_L, a_a, a_e, a_i, a_raan, a_longPeriapse, JD, R_1AU):
    """
    Computes all orbital elements of a planet about the sun using Meeus algorithm.
    :param a_L: [1-dimensional numpy array] Mean longitude coefficients
    :param a_a: [1-dimensional numpy array] Semimajor axis coefficients
    :param a_e: [1-dimensional numpy array] Eccentricity coefficients
    :param a_raan:  [1-dimensional numpy array] RAAN coefficients
    :param a_longPeriapse: [1-dimensional numpy array] Longitude of periapse coefficients
    :param JD: [double] Julian date.
    :param R_1AU: [double] 1 Astronomical Unit distance (the unit will defined the units of the semi-major axis)
    :return: [tuple] Semimajor axis, eccentricity, inclination, RAAN, argument of perigee and true anomaly given in
    Earth Mean Orbital Plane of the J2000 (EMO2000).
    """
    T = JDCenturiesFromJ2000(JD)
    mean_longitude = np.deg2rad(ephemerisMeeus(a_L, T))             # [rad] Mean longitude
    a = R_1AU * ephemerisMeeus(a_a, T)                              # [km] Semimajor axis
    e = ephemerisMeeus(a_e, T)                                      # Eccentricity
    i = np.deg2rad(ephemerisMeeus(a_i, T))                          # [rad] Inclination
    raan = np.deg2rad(ephemerisMeeus(a_raan, T))                    # [rad] Right Ascension of Ascending Node
    longPeriapse = np.deg2rad(ephemerisMeeus(a_longPeriapse, T))    # [rad] Longitude of Periapse (RAAN + arg periapse)

    argPeriapse = longPeriapse - raan                               # [rad] Argument of Periapse
    M = mean_longitude - longPeriapse                               # [rad] Mean anomaly

    # Ccen is in radians
    Ccen = (2.0*e - e**3/4 + 5.0/96.0 * e**5) * np.sin(M) + \
           (5.0/4.0 * e**2 -11.0/24.0 * e**4) * np.sin(2*M) + \
           (13.0/12.0 * e**3 - 43.0/64.0 * e**5) * np.sin(3*M) + \
           103.0/96.0 * e**4 * np.sin(4*M) + \
           1097.0/960.0 * e**5 * np.sin(5*M)

    nu = M + Ccen                                                   # [rad] True anomaly

    i = i%(np.pi)
    raan = raan%(2*np.pi)
    argPeriapse = argPeriapse%(2*np.pi)
    nu = nu%(2*np.pi)

    return (a, e, i, raan, argPeriapse, nu)

def JDCenturiesFromJ2000(JD):
    """
    Computes Julian Centuries from J2000.
    :param JD: Julian date.
    :return:
    """
    # Julian years (365.25 days) are considered
    JD_centuries = (JD - 2451545.0)/(365.25*100)

    return JD_centuries

def JDplusSeconds(JD, t):
    """
    For a given Julian Date, computes a new julian date when t seconds after JD have elapsed.
    :param JD: Julian Date.
    :param t: Seconds elapsed since JD.
    :return:
    """
    return JD + t/(3600*24)