######################################################
# coordinateTransformations
#
# Manuel F. Diaz Ramos
#
# Functions to perform coordinate transformations.
######################################################

import numpy as np

def eci2ecef(r_eci, GMST):
    """
    Transforms an ECI position into an ECEF position.
    :param r_eci: Position in ECI
    :param GMST: Greenwhich Mean Sidereal Time (in radians).
    :return: Position in ECEF.
    """
    DCM = ROT3(GMST) # Rotation matrix

    r_ecef = DCM.dot(r_eci)

    return r_ecef

def ecef2eci(r_ecef, GMST):
    """
    Transforms an ECEF position into an ECI position.
    :param r_ecef: Position in ECEF.
    :param GMST: Greenwhich Mean Sidereal Time (in radians).
    :return: Position in ECI.
    """
    DCM = ROT3(-GMST) # Rotation matrix

    r_eci = DCM.dot(r_ecef)

    return r_eci

def ecef2AzElRange(r_ecef, r_local, R_eq, e_planet):
    """
    Computes Azimuth (angle from north measured to east), Elevation, and range
    of the position r_ecef with respect to a reference r_local.
    :param r_ecef: Position in ECEF.
    :param r_local: Position of the reference in ECEF.
    :param R_eq: Mean equatorial radius.
    :param e_planet: Planet eccentricity.
    :return: A vector with Azimuth, Elevation, and Range in radians.
    """
    tol = 0.001 * np.pi/180.0 # Tolerance (0.001 deg)

    lla = ecef2lla(r_local, R_eq, e_planet, tol) # Compute Latitude, Longitude, Altitude

    r_sez = ecef2sez(r_ecef, lla[0], lla[1], lla[2], R_eq, e_planet)

    azElRange = sez2AzElRange(r_sez)

    return azElRange


def ecef2lla(r_ecef, R_eq, e_planet, tol):
    """
    Transforms a position in ECEF to Latitude, Longitude, and Altitude using a ellipsoid model for the planet.
    :param r_ecef: Position in ECEF.
    :param R_eq: Mean equatorial radius of the planet.
    :param e_planet: Eccentricity of the planet.
    :param tol: Tolerance in radians to loop over the solution.
    :return: A vector LLA.
    """
    x = r_ecef[0]
    y = r_ecef[1]
    z = r_ecef[2]
    aux = x**2 + y**2
    r = np.sqrt(aux + z**2)
    rho = np.sqrt(aux)

    if r == 0:
        lat = 0     # Not defined
        long = 0    # Not defined
        height = 0
        return np.array([lat, long, height])

    if rho == 0:
        lat = np.pi/2
        long = 0 # Actually, not defined
        height = z
        return np.array([lat, long, height])

    long = np.arctan2(y, x)

    # Iteration
    loop = True
    lat = np.arcsin(z/r) # lat is between -90 and 90. No quadrant check necessary
    while loop:
        C = R_eq/np.sqrt(1-e_planet**2 * np.sin(lat)**2)  # radius of curvature of the meridian

        lat_new = np.arctan((z + C * e_planet**2 * np.sin(lat))/rho)

        # Exit condition
        if np.abs(lat_new - lat) <= tol:
            loop = False

        lat = lat_new

    C = R_eq/np.sqrt(1-e_planet**2 * np.sin(lat)**2)  # radius of curvature of the meridian

    if (lat > 89*np.pi/180): # near the poles, cos(lat) ~ 0
        height = z/np.sin(lat) - C * (1-e_planet**2)
    else:
        height = rho/np.cos(lat) - C

    return np.array([lat, long, height])

def lla2ecef(lat, long, height, R_eq, e_planet):
    """
    Transforms Latitude, Longitude, and Altitude to ECEF using an ellisoid model for the planet.
    :param lat: Latidude [rad].
    :param long: Longitude [rad].
    :param height: Height.
    :param R_eq: Mean Equatorial radius.
    :param e_planet: Planet eccentricity.
    :return: Position vector in ECEF.
    """
    C = R_eq/np.sqrt(1 - e_planet**2 * np.sin(lat)**2)  # radius of curvature of the meridian
    S = C * (1 - e_planet**2)

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_long = np.cos(long)
    sin_long = np.sin(long)

    C_h = (C + height)
    S_h = (S + height)

    r_ecef = np.array([ C_h*cos_lat*cos_long,
                       C_h*cos_lat*sin_long,
                       S_h*sin_lat])

    return r_ecef

def ecef2sez(r_ecef, latitude, longitude, altitude, R_eq, e_planet):
    """
    Transforms ECEF position into SEZ (South, East, Zenith) using LLA of a reference position and an ellipsoid model for the planet.
    :param r_ecef: Position in ECEF.
    :param latitude: Latitude [rad].
    :param longitude: Longitude [rad].
    :param altitude: Altitude.
    :param R_eq: Mean equatorial radius of the planet.
    :param e_planet: Planet eccentricity.
    :return: A vector with the SEZ position.
    """
    r_site = lla2ecef(latitude, longitude, altitude, R_eq, e_planet)

    r_sez = ROT2(np.pi/2-latitude).dot(ROT3(longitude)).dot(r_ecef-r_site)

    return r_sez

def sez2AzElRange(r_sez):
    """
    Transforms the SEZ position (South-East-Zenith) into Azimuth, Elevation, Range.
    :param r_sez: SEZ position.
    :return:  A vector with Azimuth [rad], Elevation [rad], and Range.
    """
    range = np.linalg.norm(r_sez)

    rx = r_sez[0]
    ry = r_sez[1]
    rz = r_sez[2]

    elevation = np.arcsin(rz/range)

    azimuth = np.arctan2(ry, -rx)

    if azimuth < 0:
        azimuth = azimuth + 2*np.pi

    return np.array([azimuth, elevation, range])


def eci2RightAscensionDeclinationRange(r_eci):
    """
    Tranforms from ECI to right ascension, declination, and range.
    :param r_eci: Position in ECI.
    :return:
    """
    x = r_eci[0]
    y = r_eci[1]
    z = r_eci[2]
    r_xy = np.sqrt(x**2+y**2)
    r = np.sqrt(x**2+y**2+z**2)

    rightAs = np.arctan2(y, x)
    dec = np.arctan2(z,r_xy) # declination is between -90 and 90

    return np.array([rightAs, dec, r])

def ROT1(alpha):
    """
    Basic Rotation through 1st axis by an Euler Angle alpha
    :param alpha: Angle in radians.
    :return: The Direction Cosine Matrix.
    """
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    DCM = np.array([[1,      0,            0],
                    [0,      cos_al,  sin_al],
                    [0,     -sin_al,  cos_al]])

    return DCM

def ROT2(alpha):
    """
    Basic Rotation through 2nd axis by an Euler Angle alpha
    :param alpha: Angle in radians.
    :return: The Direction Cosine Matrix.
    """
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    DCM = np.array([[cos_al, 0, -sin_al],
                    [0,      1,       0],
                    [sin_al, 0,  cos_al]])

    return DCM

def ROT3(alpha):
    """
    Basic Rotation through 3rd axis by an Euler Angle alpha
    :param alpha: Angle in radians.
    :return: The Direction Cosine Matrix.
    """
    cos_al = np.cos(alpha)
    sin_al = np.sin(alpha)

    DCM = np.array([[cos_al,    sin_al, 0],
                    [-sin_al,   cos_al, 0],
                    [0,         0,      1]])

    return DCM
