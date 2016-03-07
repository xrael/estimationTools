######################################################
# orbitalElements
#
# Manuel F. Diaz Ramos
#
# Functions to deal with orbital elements.
######################################################

import numpy as np
from coordinateTransformations import ROT1, ROT2, ROT3

# Constants
ENERGY_THRES = 1e-8         # Threshold between parabolic and ellyptical orb
ECCENTRICITY_THRES = 1e-8   # Threshold between ellyptical and circular orb
INCLINATION_THRES  = 1e-8   # Threshold between equatorial and non-eq orb

#######################################################################################################################
# orbitalElements
#######################################################################################################################
class orbitalElements:
    """
    Describes a satellite state in orbit.
    # Description:
    # r: Position.
    # n: Line of Ascending Node.
    # e: Eccentricity Vector. Direction of periapsis.
    # a: Semi-major axis.
    # e: Eccentricity.
    # i: Inclination (0 to 180 deg). Angle between h and z.
    # Lambda: True Longitude. Angle between x and r.
    # RAAN: Right Ascension of Ascending Node. Angle between x and n.
    # w: Argument of Periapsis. Angle between n and e.
    # w_true: True Argument of Periapsis. Angle between x and e.
    # u: Argument of Latitude. Angle between n and r.
    # nu: True anomaly. Angle between e and r.
    # E: Eccentric Anomaly.
    # M: Mean Anomaly.
    """

    def __init__(self) :
        self.reset()

    def reset(self):
        self.a = 0
        self.e = 0
        self.i = 0
        self.lambd = 0
        self.raan = 0
        self.w = 0
        self.w_true = 0
        self.u = 0
        self.nu = 0
        self.E = 0
        self.M = 0

        return

    @classmethod
    def getOrbitalElementsObj(cls, a, e, i, raan, w, nu, lambd, w_true, u):
        """
        Factory method: use it to get an instance
        :param a: Semi-major axis.
        :param e: Eccentricity.
        :param i: Inclination.
        :param raan: Right Ascension of the ascending node.
        :param w: Argument of periapse.
        :param nu: True anomaly.
        :param lambd: True longitude.
        :param w_true: True argument of periapse.
        :param u: Argument of latitude.
        :return: orbitalElements object.
        """
        orbEl = orbitalElements()
        orbEl.a = a
        orbEl.e = e
        orbEl.i = i
        orbEl.lambd = lambd
        orbEl.raan = raan
        orbEl.w = w
        orbEl.w_true = w_true
        orbEl.u = u
        orbEl.nu = nu
        orbEl._nu2E()
        orbEl._E2M()

        return orbEl

    ##-----------------------Public Interface---------------------------
    def setTrueAnomaly(self, nu):
        self.nu = nu
        self._nu2E()
        self._E2M()
        return

    def setMeanAnomaly(self, M):
        self.M = M
        self._M2E()
        self._E2nu()
        return

    def setEccentricAnomaly(self, E):
        self.E = E
        self._E2nu()
        self._E2M()
        return

    ##-----------------------Private methods---------------------------
    def _nu2E(self):
        """
        Computes Eccentric Anomaly from True Anomaly.
        :return:
        """
        nu = self.nu
        e = self.e
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        cos_E = (e + cos_nu)/(1 + e*cos_nu);
        sin_E = np.sqrt(1-e**2)*sin_nu/(1 + e*cos_nu);

        E = np.arctan2(sin_E, cos_E);
        if E < 0: # arctan2 returns numbers between -pi and pi
            E = 2*np.pi + E

        self.E = E

        return self.E

    def _E2nu(self):
        """
        Computes True Anomaly from Eccentric Anomaly.
        :return:
        """
        cos_E = np.cos(self.E)
        sin_E = np.sin(self.E)
        e = self.e

        cos_nu = (cos_E - e)/(1 - e*cos_E);
        sin_nu = np.sqrt(1-e**2)*sin_E/(1-e*cos_E);

        nu = np.arctan2(sin_nu, cos_nu);
        if nu < 0: # atan2 returns numbers between -pi and pi
            nu = 2*np.pi + nu;

        self.nu = nu
        self.u = self.w + self.nu

        return self.nu

    def _E2M(self):
        """
        Computes Mean anomaly from Eccentric Anomaly.
        :return:
        """
        E = self.E
        e = self.e
        self.M = E - e * np.sin(E)

        return self.M

    def _M2E(self):
        """
        Computes Eccentric anomaly from Mean anomaly.
        :return:
        """
        tolerance = 0.001
        max_attempts = 10

        M = self.M
        E = M # Initial guess
        e = self.e

        err = np.abs(E - e * np.sin(E) - M)

        i = 0;
        while ((err > tolerance) and (i < max_attempts)) :

            jacobian = 1.0 - e * np.cos(E)

            func = E - e * np.sin(E) - M

            E = E - jacobian/func

            err = np.abs(E - e * np.sin(E) - M)

            i = i + 1

            if i >= max_attempts :
                raise 'No solution found'

        self.E = E

        return self.E

#######################################################################################################################

def orbitalElements2PositionVelocity(mu, a, e, i, raan, w, nu):
    """
    Computes position and velocity from orbital elements.
    :param a: Semi-major axis.
    :param e: Eccentricity
    :param i: Inclination
    :param raan: Right Ascension of Ascending Node
    :param w: Argument of Periaps (or true longitude of periapse for equatorial non-circular orbits)
    :param nu: True Anomaly (or True Longitude for equatorial circular orbits, or Argument of Latitude, for non-equatorial circular orbits)
    :return: [tuple] position, velocity and information on the orbit.
    """
    p = a*(1-np.square(e))          # Semi-latus Rectum

    if i < INCLINATION_THRES: # Equatorial Orbit
        i = 0
        p = a
        if e < ECCENTRICITY_THRES: # Equatorial Circular Orbit
            e = 0
            raan = 0
            w = 0
            # nu represents lambda_true (true longitude, the angle between x and the position)
            ret = 1
        else : # Equatorial non-circular Orbit
            raan = 0
            # w represents w_true (true longitude of periapse, the angle between x and the periapse)
            ret = 0 # Ellyptical orbit default
    else : # Non-equatorial orbit
        if e < ECCENTRICITY_THRES: # Non-equatorial Circular Orbit
            e = 0
            w = 0
            # nu represents u (argument of latitude, the angle between the line of nodes and the position)
            ret = 1
        else:
             ret = 0 # Ellyptical orbit default
    #---------------------------------------
    r_vec_perif = p/(1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
    v_vec_perif = np.sqrt(mu/p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    DCM_w = ROT3(-w)
    DCM_i = ROT1(-i)
    DCM_raan = ROT3(-raan)
    DCM = DCM_raan.dot(DCM_i).dot(DCM_w)

    r_vec = DCM.dot(r_vec_perif)
    v_vec = DCM.dot(v_vec_perif)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    E = np.square(v)/2 - mu/r
    if E >= 0:
        ret = 3 # The orbit is either parabolic or hyperbolic

    if np.abs(E) < ENERGY_THRES:
        ret = 2 # The orbit is parabolic

    return (r_vec, v_vec, ret)

def positionVelocity2OrbitalElements(mu, r_vec, v_vec):
    ret = 0
    x_vec = np.array([1.0, 0.0, 0.0]) # Unit inertial x-vector
    z_vec = np.array([0.0, 0.0, 1.0]) # Unit inertial z-vector

    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    if h < 1e-6:
        #Degenerate orbit
        return 4 # Degenerate orbit

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Line of ascending node
    n_vec = np.cross(z_vec, h_vec)
    n = np.linalg.norm(n_vec)

    # Energy
    E = np.square(v)/2 - mu/r
    if E >= 0:
        ret = 3 # The orbit is either parabolic or hyperbolic
    else:
        ret = 0 # Ellyptical orbit default

    e_vec = np.cross(v_vec, h_vec)/mu - r_vec/r
    e = np.linalg.norm(e_vec)

    if np.abs(E) < ENERGY_THRES:
        a = 0 # not defined
        p = h**2/mu
        e = 1.0
        ret = 2 # The orbit is parabolic
    else:
        # Orbit shape
        a = -mu/(2*E)              # Semi-major axis
        p = a*(1-e**2)             # Semi-latus rectum

    # Inclination
    i = np.arccos(h_vec[2]/h) # returns angles between 0 and pi

    # True Longitude: angle between x_inertial and r_vec
    lambd = np.arccos(np.dot(x_vec, r_vec)/r)
    if r_vec[1] < 0: # lambda > 180 deg
        lambd = 2*np.pi - lambd

    if i < INCLINATION_THRES: # (i ~ 0) less than 0.1'
        # Equatorial orbit
        # not defined for any equatorial orbit:
        raan = 0.0        # Right Ascension of Ascending Node
        w = 0.0           # Argument of Periapsis
        u = 0.0           # Argument of Latitude
        if e < ECCENTRICITY_THRES:
            #Equatorial circular orbit: line of nodes and periapsis not defined.
            w_true = 0.0  # True Argument of Periapsis
            nu = 0.0      # True Anomaly
            ret = 1
        else:
            #Equatorial non-circular orbit: line of nodes not defined.

            # True Argument of Periapsis
            w_true = np.arccos(np.dot(x_vec, e_vec)/e)
            if e_vec[1] < 0:
                w_true = 2*np.pi - w_true

            # True anomaly
            nu = np.arccos(np.dot(e_vec, r_vec)/(e*r))
            if np.dot(r_vec, v_vec) < 0:
                nu = 2*np.pi - nu
    else:
        #non-Equatorial orbit

        # Right-ascension of ascending node
        raan = np.arctan2(h_vec[0], -h_vec[1])

        # Argument of Latitude (w + nu)
        u = np.arccos(np.dot(n_vec, r_vec)/(n*r))
        if r_vec[2] < 0 : #u > 180 deg
            u = 2*np.pi - u

        if e < ECCENTRICITY_THRES:
            #Non-equatorial circular orbit: periapsis not defined
            #not defined:
            w_true = 0  # True Argument of Periapsis
            w = 0       # Argument of Periapsis
            nu = 0      # True Anomaly
            ret = 1
        else:
            #Non-equatorial non-circular orbit

            # Argument of Periapsis
            w = np.arccos(np.dot(n_vec, e_vec)/(n*e))
            if e_vec[2] < 0:
                w = 2*np.pi - w

            # True Argument of Periapsis
            w_true = np.arccos(np.dot(x_vec, e_vec)/e)
            if e_vec[1] < 0:
                w_true = 2*np.pi - w_true

            # True Anomaly
            nu = np.arccos(np.dot(e_vec, self._r_vec)/(e*r))
            if np.dot(self._r_vec, self._v_vec) < 0:
                nu = 2*np.pi - nu
    #-----------------------

    self._orbitalElements = orbitalElements.getOrbitalElementsObj(a, e, i, raan, w, nu, lambd, w_true, u)

    return ret
