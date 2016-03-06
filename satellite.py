######################################################
# Manuel F. Diaz Ramos
######################################################

import numpy as np
from orbitalElements import *

######################################################
# satelliteState
######################################################
class satelliteState:
    """
    Stores information about state and provides methods to transform between different descriptions
    """

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

    @classmethod
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

        if np.abs(E) < ENERGY_THRES:
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

        if i < INCLINATION_THRES: # (i ~ 0) less than 0.1'
            # Equatorial orbit
            # not defined for any equatorial orbit:
            raan = 0        # Right Ascension of Ascending Node
            w = 0           # Argument of Periapsis
            u = 0           # Argument of Latitude
            if e < ECCENTRICITY_THRES:
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

        if i < INCLINATION_THRES: # Equatorial Orbit
            if e < ECCENTRICITY_THRES: # Equatorial Circular Orbit
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
            if e < ECCENTRICITY_THRES: # Non-equatorial Circular Orbit
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

        if np.abs(E) < ENERGY_THRES:
            ret = 2 # The orbit is parabolic

        self._r_vec = np.squeeze(np.asarray(r_vec))
        self._v_vec = np.squeeze(np.asarray(v_vec))

        return ret

######################################################

######################################################
#Statelist:
# Implements a list of states.
######################################################
class stateList:

    # Constructors and factory methods--------------------
    def __init__(self):
        self._stateVector = []

    @classmethod
    def getFromVectors(cls, mu, state_vectors, time_vector):
        nmbr_states = time_vector.size

        state_list = stateList()

        for i in range(0, nmbr_states):
            st = satelliteState.getSatelliteStateObjFromRV(mu, state_vectors[i][0:3],state_vectors[i][3:6], time_vector[i])
            state_list.add(st)

        return  state_list


    # reset the list
    def emptyList(self):
        del self._stateVector[:]
        self._stateVector = []

    def add(self, state):
        self._stateVector.append(state)

    def getTimeVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        time = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            time[index] = state.getTime()
            index = index + 1
        return time

    def getPositionVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        r_vec = np.zeros((len(self._stateVector), 3))
        index = 0
        for state in self._stateVector:
            r_vec[index] = state.getPosition()
            index = index + 1
        return r_vec

    def getRadiusVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        r = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            r[index] = state.getRadius()
            index = index + 1
        return r

    def getVelocityVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        v_vec = np.zeros((len(self._stateVector), 3))
        index = 0
        for state in self._stateVector:
            v_vec[index] = state.getVelocity()
            index = index + 1
        return v_vec

    def getStateVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        state_vec = np.zeros((len(self._stateVector), 6))
        index = 0
        for state in self._stateVector:
            state_vec[index] = np.concatenate([state.getPosition(), state.getVelocity()])
            index = index + 1
        return state_vec

    def getSpeedVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        v = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            v[index] = state.getSpeed()
            index = index + 1
        return v

    def getSemimajorAxisVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        a = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            a[index] = state.getOrbitalElements().a
            index = index + 1
        return a

    def getEccentricityVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        e = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            e[index] = state.getOrbitalElements().e
            index = index + 1
        return e

    def getInclinationVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        i = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            i[index] = state.getOrbitalElements().i
            index = index + 1
        return i

    def getRAANVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        raan = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            raan[index] = state.getOrbitalElements().raan
            index = index + 1
        return raan

    def getArgumentOfPeriapsisVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        w = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            w[index] = state.getOrbitalElements().w
            index = index + 1
        return w

    def getTrueAnomalyVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        nu = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            nu[index] = state.getOrbitalElements().nu
            index = index + 1
        return nu

    def getEccentricAnomalyVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        Ecc = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            Ecc[index] = state.getOrbitalElements().E
            index = index + 1
        return Ecc

    def getEnergyVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        E = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            E[index] = state.getRealEnergy()
            index = index + 1
        return E

    def getAngularMomentumVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        h = np.zeros([len(self._stateVector), 3])
        index = 0
        for state in self._stateVector:
            h[index] = state.getAngularMomentum()
            index = index + 1
        return h

    def getTimeOfPeriapseVector(self):
        if not self._stateVector: #empty list
            return np.array([])

        tp = np.zeros(len(self._stateVector))
        index = 0
        for state in self._stateVector:
            tp[index] = state.getTimeOfPeriapse()
            index = index + 1
        return tp
######################################################