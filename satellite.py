######################################################
# Manuel F. Diaz Ramos
######################################################

import numpy as np

######################################################
# orbitalElements
######################################################
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

######################################################

######################################################
# satelliteState
######################################################
class satelliteState:
    """
    Stores information about state and provides methods to transform between different descriptions
    """

    # Constants
    ENERGY_THRES = 1e-6         # Threshold between parabolic and ellyptical orb
    ECCENTRICITY_THRES = 1e-6   # Threshold between ellyptical and circular orb
    INCLINATION_THRES  = 1e-6   # Threshold between equatorial and non-eq orb

    ##-------------Constructors and Factory Methods---------------------
    # Constructor: do not use it. Use the factory methods instead
    def __init__(self):
        self._mu = 0
        self._r_vec = np.array([0,0,0])
        self._v_vec = np.array([0,0,0])
        self._U = 0
        self._orbitalElements = None
        self._orbitInformation = -1
        self._time = 0
        self._period = 0
        self._timePeriapse = 0

    # Factory method: from RV
    @classmethod
    def getSatelliteStateObjFromRV(cls, mu, r_vec, v_vec, time = 0):
        state = satelliteState()
        state._mu = mu
        state._r_vec = r_vec
        state._v_vec = v_vec
        state._time = time
        #state._orbitalElements = orbitalElements()
        state._orbitInformation = state._getOrbitalElements()

        a = state._orbitalElements.a
        n = np.sqrt(mu/a**3)
        if a > 0:
            state._period = 2*np.pi/n
            angle_periapse = np.mod(time, state._period)*n - state._orbitalElements.M

            if angle_periapse < -np.pi:
                angle_periapse = angle_periapse + 2*np.pi
            elif angle_periapse > np.pi:
                angle_periapse = angle_periapse - 2*np.pi

            state._timePeriapse = angle_periapse/n
        else:
            state._period = 0
            state._timePeriapse = 0

        return state

    # Factory method: from Orbital Elements
    @classmethod
    def getSatelliteStateObjFromOrbElem(cls, mu, orb_elem, time = 0):
        state = satelliteState()
        state._mu = mu
        state._orbitalElements = orb_elem
        state._time = time
        state._orbitInformation = state._getRV()

        return state
    #--------------------------------------------------------

    def advanceTime(self, dt):
        """
        Advances time by dt modifying the state.
        :param dt:
        :return:
        """
        self._time += dt

        orbEl = self._orbitalElements

        n = np.sqrt(self._mu / orbEl.a**3)

        orbEl.setMeanAnomaly(np.mod(self._orbitalElements.M + n*dt, 2*np.pi))

        self._orbitInformation = self._getRV() # update RV

        return self._orbitInformation

    ##----------------------Getters---------------------------
    def getTime(self):
        return self._time

    def getPosition(self):
        return self._r_vec

    def getRadius(self):
        return np.sqrt(np.sum(np.square(self._r_vec)))

    def getVelocity(self):
        return self._v_vec

    def getSpeed(self):
        return np.sqrt(np.sum(np.square(self._v_vec)))

    def getOrbitalElements(self):
        return self._orbitalElements

    def getRealEnergy(self):
        return self.getSpeed()**2/2 - self._U

    def getAngularMomentum(self):
        return np.cross(self._r_vec, self._v_vec)

    def getOrbitPeriod(self):
        return self._period

    def getTimeOfPeriapse(self):
        return self._timePeriapse

    ##-------------------Print Methods------------------------
    def printOrbitalElements(self, deg = 1):
        if deg != 0:
            print 'Semi-major axis (a): ' + str(self._orbitalElements.a) + ' [km]'
            print 'Eccentricity (e): ' + str(self._orbitalElements.e) + ''
            print 'Inclination (i): ' + str(self._orbitalElements.i*180/np.pi) + ' [deg]'
            print 'Right Ascension of Ascending Node (RAAN): ' + str(self._orbitalElements.raan*180/np.pi) + ' [deg]'
            print 'Argument of Periapsis (w): ' + str(self._orbitalElements.w*180/np.pi) + ' [deg]'
            print 'True Anomaly (nu): ' + str(self._orbitalElements.nu*180/np.pi) + ' [deg]'
            print 'True Longitude (Lambda): ' + str(self._orbitalElements.lambd*180/np.pi) + ' [deg]'
            print 'True Argument of Periapsis (w_true): ' + str(self._orbitalElements.w_true*180/np.pi) + ' [deg]'
            print 'Argument of Latitude (u): ' + str(self._orbitalElements.u*180/np.pi) + ' [deg]'
        else:
            print 'Semi-major axis (a): ' + str(self._orbitalElements.a) + ' [km]'
            print 'Eccentricity (e): ' + str(self._orbitalElements.e) + ''
            print 'Inclination (i): ' + str(self._orbitalElements.i) + ' [rad]'
            print 'Right Ascension of Ascending Node (RAAN): ' + str(self._orbitalElements.raan) + ' [rad]'
            print 'Argument of Periapsis (w): ' + str(self._orbitalElements.w) + ' [rad]'
            print 'True Anomaly (nu): ' + str(self._orbitalElements.nu) + ' [rad]'
            print 'True Longitude (Lambda): ' + str(self._orbitalElements.lambd) + ' [rad]'
            print 'True Argument of Periapsis (w_true): ' + str(self._orbitalElements.w_true) + ' [rad]'
            print 'Argument of Latitude (u): ' + str(self._orbitalElements.u) + ' [rad]'


    def printRV(self):
        print 'Position (r): ' + str(self._r_vec)
        print 'Velocity (v): ' + str(self._v_vec)
    #--------------------------------------------------------

    def getOrbitInformation(self):
        if self._orbitInformation == 0:
            return 'Elliptycal'
        elif self._orbitInformation == 1:
            return 'Circular'
        elif self._orbitInformation == 2:
            return 'Parabolic Orbit'
        elif self._orbitInformation == 3:
            return 'Hyperbolic Orbit'
        elif self._orbitInformation == 4:
            return 'Degenerate Orbit'
        else:
            return 'Unknown State'

    # Transfromations state-orbital elements-----------------
    def _getOrbitalElements(self):
        ret = 0

        #orbEl = self._orbitalElements

        x_vec = np.array([1, 0, 0]) # Unit inertial x-vector
        z_vec = np.array([0, 0, 1]) # Unit inertial z-vector

        # Angular momentum
        h_vec = np.cross(self._r_vec, self._v_vec)
        h = np.sqrt(np.sum(np.square(h_vec)))
        if h < 0.0001:
            #Degenerate orbit
            self._orbitalElements = orbitalElements()
            #orbEl.reset()
            return 4

        r = self.getRadius()
        v = self.getSpeed()

        # Line of ascending node
        n_vec = np.cross(z_vec, h_vec)
        n = np.sqrt(np.sum(np.square(n_vec)))

        # Energy
        E = np.square(v)/2 - self._mu/r
        if E >= 0:
            ret = 3 # The orbit is either parabolic or hyperbolic
        else:
            ret = 0 # Ellyptical orbit default

        e_vec = np.cross(self._v_vec, h_vec)/self._mu - self._r_vec/r

        if np.abs(E) < satelliteState.ENERGY_THRES:
            a = 0 # not defined
            e = 1
            ret = 2 # The orbit is parabolic
        else:
            # Orbit shape
            a = - self._mu/(2*E)              # Semi-major axis
            # From p = h^2/mu = a*(1-e^2):
            #e = np.sqrt(np.abs(1 - np.square(h)/(self._mu*a)))    # Eccentricity
            e = np.sqrt(np.sum(np.square(e_vec)))

        # Inclination
        i = np.arccos(h_vec[2]/h) # returns angles between 0 and pi

        # True Longitude: angle between x_inertial and r_vec
        lambd = np.arccos(np.dot(x_vec, self._r_vec)/r)
        if self._r_vec[1] < 0: # lambda > 180 deg
            lambd = 2*np.pi - lambd

        if i < satelliteState.INCLINATION_THRES: # (i ~ 0) less than 0.1'
            # Equatorial orbit
            # not defined for any equatorial orbit:
            raan = 0        # Right Ascension of Ascending Node
            w = 0           # Argument of Periapsis
            u = 0           # Argument of Latitude
            if e < satelliteState.ECCENTRICITY_THRES:
                #Equatorial circular orbit: line of nodes and periapsis not defined.
                w_true = 0  # True Argument of Periapsis
                nu = 0      # True Anomaly
                ret = 1
            else:
                #Equatorial non-circular orbit: line of nodes not defined.

                # True Argument of Periapsis
                w_true = np.arccos(np.dot(x_vec, e_vec)/e)
                if e_vec[1] < 0:
                    w_true = 2*np.pi - w_true

                # True anomaly
                nu = np.arccos(np.dot(e_vec, self._r_vec)/(e*r))
                if np.dot(self._r_vec, self._v_vec) < 0:
                    nu = 2*np.pi - nu
        else:
            #non-Equatorial orbit

            # Right-ascension of ascending node
            raan = np.arctan2(h_vec[0], -h_vec[1])

            # Argument of Latitude (w + nu)
            u = np.arccos(np.dot(n_vec, self._r_vec)/(n*r))
            if self._r_vec[2] < 0 : #u > 180 deg
                u = 2*np.pi - u

            if e < satelliteState.ECCENTRICITY_THRES:
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

    def _getRV(self):
        a = self._orbitalElements.a
        e = self._orbitalElements.e
        i = self._orbitalElements.i
        raan = self._orbitalElements.raan
        w = self._orbitalElements.w
        nu = self._orbitalElements.nu
        lambd = self._orbitalElements.lambd
        w_true = self._orbitalElements.w_true
        u = self._orbitalElements.u

        p = a*(1-np.square(e))          # Semi-latus Rectum

        if i < satelliteState.INCLINATION_THRES: # Equatorial Orbit
            if e < satelliteState.ECCENTRICITY_THRES: # Equatorial Circular Orbit
                r_vec_perif = a * np.array([np.cos(lambd), np.sin(lambd), 0])
                v_vec_perif = np.sqrt(self._mu/a) * np.array([-np.sin(lambd), np.cos(lambd), 0])

                DCM = np.matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
                ret = 1
            else : # Equatorial non-circular Orbit
                r_vec_perif = p/(1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
                v_vec_perif = np.sqrt(self._mu/p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

                DCM = np.matrix(np.array([[np.cos(w_true), -np.sin(w_true), 0], [np.sin(w_true), np.cos(w_true), 0], [0, 0, 1]]))
                ret = 0 # Ellyptical orbit default
        else : # Non-equatorial orbit
            if e < satelliteState.ECCENTRICITY_THRES: # Non-equatorial Circular Orbit
                r_vec_perif = a * np.array([np.cos(u), np.sin(u), 0])
                v_vec_perif = np.sqrt(self._mu/a) * np.array([-np.sin(u), np.cos(u), 0])

                DCM_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
                DCM_raan = np.array([[np.cos(raan), -np.sin(raan), 0], [np.sin(raan), np.cos(raan), 0], [0, 0, 1]])
                DCM = np.matrix(DCM_raan) * np.matrix(DCM_i)
                ret = 1
            else : # Non-equatorial non-circular orbit
                r_vec_perif = p/(1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
                v_vec_perif = np.sqrt(self._mu/p) * np.array([-np.sin(nu), e + np.cos(nu), 0])

                DCM_w = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
                DCM_i = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
                DCM_raan = np.array([[np.cos(raan), -np.sin(raan), 0], [np.sin(raan), np.cos(raan), 0], [0, 0, 1]])
                DCM = np.matrix(DCM_raan) * np.matrix(DCM_i) * np.matrix(DCM_w)
                ret = 0 # Ellyptical orbit default
        #---------------------------------------

        r_vec = DCM * np.matrix.transpose(np.matrix(r_vec_perif))
        v_vec = DCM * np.matrix.transpose(np.matrix(v_vec_perif))

        r = np.sqrt(np.sum(np.square(r_vec)))
        v = np.sqrt(np.sum(np.square(v_vec)))

        E = np.square(v)/2 - self._mu/r
        if E >= 0:
            ret = 3 # The orbit is either parabolic or hyperbolic

        if np.abs(E) < satelliteState.ENERGY_THRES:
            ret = 2 # The orbit is parabolic

        self._r_vec = np.squeeze(np.asarray(r_vec))
        self._v_vec = np.squeeze(np.asarray(v_vec))

        return ret

######################################################