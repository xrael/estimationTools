######################################################
# Attitude Observation Models
#
# Manuel F. Diaz Ramos
#
# These classes compute symbollically and numerically
# attitude bservation models and their derivatives.
######################################################

from modelBase import observerModelBase
import attitudeKinematics
import sympy as sp
import numpy as np

# Star catalog:
# Catalog of 57 stars:
# [Number, name, Sidereal Hour Angle (360 - right ascension) [deg], Declination [deg]]
star_catalog = [
[1, "Alpheratz", 		358, 29],
[2,  "Ankaa", 			354,-42],
[3,  "Schedar", 		350, 56],
[4,  "Diphda", 			349, -18],
[5,  "Achernar", 		336, -57],
[6,  "Hamal", 			328, 23],
[7,  "Acamar", 			316, -40],
[8,  "Menkar", 			315, 4],
[9,  "Mirfak",			309, 50],
[10, "Aldebaran",		291, 16],
[11, "Rigel",			282, -8],
[12, "Capella",			281, 46],
[13, "Bellatrix",		279, 06],
[14, "Elnath",			279, 29],
[15, "Alnilam",			276, -01],
[16, "Betelgeuse",		271, 7],
[17, "Canopus",			264, -53],
[18, "Sirius",			259, -17],
[19, "Adhara",			256, -29],
[20, "Procyon",			245, 5],
[21, "Pollux",			244, 28],
[22, "Avior",			234, -59],
[23, "Suhail",			223, -43],
[24, "Miaplacidus",		222, -70],
[25, "Alphard",			218, -9],
[26, "Regulus",			208, 12],
[27, "Dubhe",			194, 62],
[28, "Denebola",		183, 15],
[29, "Gienah",			176, -17],
[30, "Acrux",			174, -63],
[31, "Gacrux",			172, -57],
[32, "Alioth",			167, 56],
[33, "Spica",			159, -11],
[34, "Alkaid",			153, 49],
[35, "Hadar",			149, -60],
[36, "Menkent",			149, -36],
[38, "Rigil Kentaurus",	140, -61],
[37, "Arcturus",		146, 19],
[39, "Zubenelgenubi",	138, -16],
[40, "Kochab",			137, 74],
[41, "Alphecca",		127, 27],
[42, "Antares",			113, -26],
[43, "Atria",			108, -69],
[44, "Sabik",			103, -16],
[45, "Shaula",			97, -37],
[46, "Rasalhague",		96, 13],
[47, "Eltanin",			91, 51],
[48, "Kaus",			84, -34],
[49, "Vega",			81, 39],
[50, "Nunki",			76, -26],
[51, "Altair",			63, -9],
[52, "Peacock",			54, -57],
[53, "Deneb",			50, 45],
[54, "Enif",			34, 10],
[55, "Al Na'ir",		28, -47],
[56, "Fomalhaut",		16, -30],
[57, "Markab",			14, 15]]


class mrpObs(observerModelBase):

    _obs_rate = None
    _last_obs = -1

    def __init__(self, stateSymb, params, observerCoordinates, obsRate, inputSymb = None):
        super(mrpObs, self).__init__("MRPobservations", stateSymb, params, observerCoordinates, inputSymb)

        self._obs_rate = obsRate
        self._last_obs = -1

        return

    # Factory methods
    @classmethod
    def getObserverModel(cls, obsRate):

        params = ()
        observer_coordinates = None # THIS IS NOT USED. This simple model assumes that the mrp can be observed from any attitude
        symbState = mrpObs.buildSymbolicState()
        inputSymb = mrpObs.buildSymbolicInput()
        obs_mod = mrpObs(symbState, params, observer_coordinates, obsRate, inputSymb)

        return obs_mod

    ## THIS IS THE METHOD TO BE MODIFIED IF A CHANGE IN THE STATE IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicState(cls):
        """
        Modify this method to build a new symbolic state vector.
        :return: A list with the symbolic symbols of the state.
        """
        # MINIMUM STATE: [x, y, z, x_dot, y_dot, z_dot]
        mrp1, mrp2, mrp3 = sp.symbols('mrp1 mrp2 mrp3')
        w1, w2, w3 = sp.symbols('w1 w2 w3')

        X_symb = [mrp1, mrp2, mrp3, w1, w2, w3]
        return X_symb

    ## -------------------------Public Interface--------------------------
    # This is the important interface por the estimation processors!

    def normalizeOutput(self, out):
        return attitudeKinematics.switchMRPrepresentation(out)

    def normalizePrefitResiduals(self, y_prefit, Y_obs, Y_computed):

        Y_obs_inner = np.inner(Y_obs, Y_obs)
        Y_obs_norm = np.sqrt(Y_obs_inner)
        if Y_obs_norm > 0.3:
            y_i_prime = -Y_obs/Y_obs_inner - Y_computed
            if np.linalg.norm(y_i_prime) < np.linalg.norm(y_prefit):
                y_prefit = y_i_prime

        return y_prefit

    def computeModel(self, X, t, params, u = None):
        """
        Computes the observation model function G: Y = G(X,t)
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The vector G(X,t) evaluating G.
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]

        # GS_nmbr = params
        # GS_coord = self.getObserverCoordinates()

        nmbrOfOutputs = self.getNmbrOutputs()

        G = np.zeros(nmbrOfOutputs)
        for i in range(0, nmbrOfOutputs):
            G[i] = self._modelLambda[i](mrp1, mrp2, mrp3)

        return G

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of G (Htilde).
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The jacobian evaluated at (X,t).
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]

        # GS_nmbr = params
        # GS_coord = self.getObserverCoordinates()

        nmbrOfOutputs = self.getNmbrOutputs()
        nmbrOfStates = self.getNmbrOfStates()

        Htilde = np.zeros([nmbrOfOutputs,nmbrOfStates])

        for i in range(0,nmbrOfOutputs):
            for j in range(0, nmbrOfStates) :
                Htilde[i][j] = self._jacobianLambda[i][j](mrp1, mrp2, mrp3)

        return Htilde

    def normalizeState(self, X):
        attitudeKinematics.switchMRPrepresentation(X)
        return


    def getNmbrOutputs(self):
        return 3

    def isObservable(self, X, t, params):
        """
        Method that decides if a state is observable for a given time and GS coordinates.
        :param X: State.
        :param t: Time.
        :param params: Dynamic parameters (GS coordinates).
        :return: Boolean indicating if it is visible.
        """

        # GS_number = params[0]
        # GS_coord = self.getObserverCoordinates()

        period_obs = 1.0/self._obs_rate

        # It is assumed that the observations cannot be taken faster than obs_rate.
        if self._last_obs == -1:
            self._last_obs = t
            return True
        elif self._last_obs + period_obs <= t:
            self._last_obs = t
            return True
        else:
            return False


    ## -------------------------Private Methods--------------------------

    def computeSymbolicModel(self):
        """
        Symbollically computes G(X,t) and stores the models and lambda functions.
        :return:
        """
        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]

        g1 = sp.lambdify((mrp1, mrp2, mrp3), mrp1, "numpy")
        g2 = sp.lambdify((mrp1, mrp2, mrp3), mrp2, "numpy")
        g3 = sp.lambdify((mrp1, mrp2, mrp3), mrp3, "numpy")

        self._modelSymb = [mrp1, mrp2, mrp3]
        self._modelLambda = [g1, g2, g3]

        return self._modelSymb

    def computeSymbolicJacobian(self):

        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]

        G = self._modelSymb

        nmbrOfStates = self.getNmbrOfStates()
        nmbrOfOutputs = self.getNmbrOutputs()

        dG = [[0 for i in range(0,nmbrOfStates)] for i in range(0,nmbrOfOutputs)]
        Htilde_lambda = [[0 for i in range(0,nmbrOfStates)] for i in range(0,nmbrOfOutputs)]

        for i in range(0, nmbrOfOutputs) :
            for j in range(0, nmbrOfStates) :
                dG[i][j] = sp.diff(G[i], self._stateSymb[j])
                Htilde_lambda[i][j] = sp.lambdify((mrp1, mrp2, mrp3), dG[i][j], "numpy")

        self._jacobianSymb = dG
        self._jacobianLambda = Htilde_lambda

        return self._jacobianSymb

#######################################################################################################################
class starTrackerObs(observerModelBase):

    _obs_rate = None
    _last_obs = -1
    _starTrackerDirecBody = None
    _fov = None
    _last_obs = -1

    def __init__(self, stateSymb, params, observerCoordinates, obsRate, starTrackerDirecBody, fov, inputSymb = None):
        super(starTrackerObs, self).__init__("StarTracker", stateSymb, params, observerCoordinates, inputSymb)

        self._obs_rate = obsRate
        self._last_obs = -1
        self._starTrackerDirecBody = starTrackerDirecBody
        self._fov = fov
        self._last_obs = -1

        return

    # Factory methods
    @classmethod
    def getObserverModel(cls, obsRate, observerCoordinates, starTrackerDirecBody, fov):
        """

        :param obsRate:
        :param observerCoordinates: [List] Each entre on the list should include: [number of star in catalog, name, right ascension (rad), declination (rad)]
        :param starTrackerDirecBody: [List] Each entry contains a body direction in which the star tracker camera points.
        :param fov: [List] Each entry contains an angle [rad] with the Field of View. Each star tracker is only able to observe a cone angle of fov/2 around starTrackerDirecBody.
        :return:
        """
        params = ()
        symbState = starTrackerObs.buildSymbolicState()
        inputSymb = starTrackerObs.buildSymbolicInput()
        obs_mod = starTrackerObs(symbState, params, observerCoordinates, obsRate, starTrackerDirecBody, fov, inputSymb)

        return obs_mod

    ## THIS IS THE METHOD TO BE MODIFIED IF A CHANGE IN THE STATE IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicState(cls):
        """
        Modify this method to build a new symbolic state vector.
        :return: A list with the symbolic symbols of the state.
        """
        # MINIMUM STATE: [x, y, z, x_dot, y_dot, z_dot]
        mrp1, mrp2, mrp3 = sp.symbols('mrp1 mrp2 mrp3')
        #w1, w2, w3 = sp.symbols('w1 w2 w3')

        X_symb = [mrp1, mrp2, mrp3]#, w1, w2, w3]
        return X_symb

    ## -------------------------Public Interface--------------------------
    # This is the important interface por the estimation processors!

    # def normalizeOutput(self, out):
    #     out = out/np.linalg.norm(out) # The observations are normalized (unit vectors)

    def computeModel(self, X, t, params, u = None):
        """
        Computes the observation model function G: Y = G(X,t)
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The vector G(X,t) evaluating G.
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]

        star_nmbr = params[0]

        star_coord = self.getObserverCoordinates()

        star_right_as = star_coord[star_nmbr][2]
        star_dec = star_coord[star_nmbr][3]

        nmbrOfOutputs = self.getNmbrOutputs()

        G = np.zeros(nmbrOfOutputs)
        for i in range(0, nmbrOfOutputs):
            G[i] = self._modelLambda[i](mrp1, mrp2, mrp3, star_right_as, star_dec)

        return G

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of G (Htilde).
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The jacobian evaluated at (X,t).
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]

        star_nmbr = params[0]

        star_coord = self.getObserverCoordinates()

        star_right_as = star_coord[star_nmbr][2]
        star_dec = star_coord[star_nmbr][3]

        nmbrOfOutputs = self.getNmbrOutputs()
        nmbrOfStates = self.getNmbrOfStates()

        Htilde = np.zeros([nmbrOfOutputs,nmbrOfStates])

        for i in range(0,nmbrOfOutputs):
            for j in range(0, nmbrOfStates) :
                Htilde[i][j] = self._jacobianLambda[i][j](mrp1, mrp2, mrp3, star_right_as, star_dec)

        return Htilde

    def normalizeState(self, X):
        attitudeKinematics.switchMRPrepresentation(X)
        return


    def getNmbrOutputs(self):
        return 3


    def isObservable(self, X, t, params):
        """
        Method that decides if an attitude is observable for a given time and star coordinates.
        :param X: State.
        :param t: Time.
        :param params: Dynamic parameters (star coordinates).
        :return: Boolean indicating if it is visible.
        """
        period_obs = 1.0/self._obs_rate

        # It is assumed that the observations cannot be taken faster than obs_rate.
        if self._last_obs == -1:
            self._last_obs = t
        elif self._last_obs == t: # It is still the same sampling interval
            self._last_obs = t
        elif self._last_obs + period_obs <= t:
            self._last_obs = t
        else:
            return False

        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]

        mrp = np.array([mrp1, mrp2, mrp3])

        star_number = params[0]

        star_coord = self.getObserverCoordinates()

        star_right_as = star_coord[star_number][2]
        star_dec = star_coord[star_number][3]

        cos_alpha = np.cos(star_right_as)
        sin_alpha = np.sin(star_right_as)
        cos_delta = np.cos(star_dec)
        sin_delta = np.sin(star_dec)

        # Inertial direction of the star (Unit vector!!!)
        r_N = np.array([cos_delta*cos_alpha, cos_delta*sin_alpha, sin_delta])

        # Body attitude B relative to inertial frame N
        BN = attitudeKinematics.mrp2dcm(mrp)

        # Body direction of the star (Unit vector!!!)
        r_B = BN.dot(r_N)

        st_nmbr = len(self._starTrackerDirecBody) # Number of Star Trackers
        for i in range(0, st_nmbr):

            b_ST = self._starTrackerDirecBody[i] # Direction of the camera of the star
            fov_ST = self._fov[i]

            angle = np.arccos(r_B.dot(b_ST)) # angle between the star and the normal to the camera

            if angle <= fov_ST/2: # The star is observable
                return True

        return False

     ## -------------------------Private Methods--------------------------

    def computeSymbolicModel(self):
        """
        Symbollically computes G(X,t) and stores the models and lambda functions.
        :return:
        """
        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]

        right_as = sp.symbols('right_as')
        dec = sp.symbols('dec')

        # The inertial position of the star is indicated through its right ascension and declination
        r1 = sp.cos(right_as)*sp.cos(dec)
        r2 = sp.sin(right_as)*sp.cos(dec)
        r3 = sp.sin(dec)

        mrp_sq = mrp1**2 + mrp2**2 + mrp3**2

        # DCM parameterized using MRPs
        BN11 = (4*(mrp1**2 - mrp2**2 - mrp3**2) + (1 - mrp_sq)**2)/(1 + mrp_sq)**2
        BN12 = (8*mrp1*mrp2 + 4*mrp3*(1-mrp_sq))/(1 + mrp_sq)**2
        BN13 = (8*mrp1*mrp3 - 4*mrp2*(1-mrp_sq))/(1 + mrp_sq)**2

        BN21 = (8*mrp2*mrp1 - 4*mrp3*(1-mrp_sq))/(1 + mrp_sq)**2
        BN22 = (4*(-mrp1**2 + mrp2**2 - mrp3**2) + (1 - mrp_sq)**2)/(1 + mrp_sq)**2
        BN23 = (8*mrp2*mrp3 + 4*mrp1*(1-mrp_sq))/(1 + mrp_sq)**2

        BN31 = (8*mrp3*mrp1 + 4*mrp2*(1-mrp_sq))/(1 + mrp_sq)**2
        BN32 = (8*mrp3*mrp2 - 4*mrp1*(1-mrp_sq))/(1 + mrp_sq)**2
        BN33 = (4*(-mrp1**2 - mrp2**2 + mrp3**2) + (1 - mrp_sq)**2)/(1 + mrp_sq)**2

        # Output (Start vectors in B frame)
        b1 = BN11 * r1 + BN12 * r2 + BN13 * r3
        b2 = BN21 * r1 + BN22 * r2 + BN23 * r3
        b3 = BN31 * r1 + BN32 * r2 + BN33 * r3

        g1 = sp.lambdify((mrp1, mrp2, mrp3, right_as, dec), b1, "numpy")
        g2 = sp.lambdify((mrp1, mrp2, mrp3, right_as, dec), b2, "numpy")
        g3 = sp.lambdify((mrp1, mrp2, mrp3, right_as, dec), b3, "numpy")

        self._modelSymb = [b1, b2, b3]
        self._modelLambda = [g1, g2, g3]

        return self._modelSymb

    def computeSymbolicJacobian(self):

        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]

        right_as = sp.symbols('right_as')
        dec = sp.symbols('dec')

        G = self._modelSymb

        nmbrOfStates = self.getNmbrOfStates()
        nmbrOfOutputs = self.getNmbrOutputs()

        dG = [[0 for i in range(0,nmbrOfStates)] for i in range(0,nmbrOfOutputs)]
        Htilde_lambda = [[0 for i in range(0,nmbrOfStates)] for i in range(0,nmbrOfOutputs)]

        for i in range(0, nmbrOfOutputs) :
            for j in range(0, nmbrOfStates) :
                dG[i][j] = sp.diff(G[i], self._stateSymb[j])
                Htilde_lambda[i][j] = sp.lambdify((mrp1, mrp2, mrp3, right_as, dec), dG[i][j], "numpy")

        self._jacobianSymb = dG
        self._jacobianLambda = Htilde_lambda

        return self._jacobianSymb

#######################################################################################################################
class rateGyroObs(observerModelBase):

    _obs_rate = None
    _last_obs = -1
    _sigma_ARW = np.array([0,0,0]) # 3-component vector
    _sigma_RRW = np.array([0,0,0]) # 3-component vector
    _bias = np.array([0,0,0]) # 3-component vector

    def __init__(self, stateSymb, params, observerCoordinates, obsRate, inputSymb = None):
        super(rateGyroObs, self).__init__("RateGyro", stateSymb, params, observerCoordinates, inputSymb)

        self._obs_rate = obsRate
        self._last_obs = -1
        self._noiseFlag = False
        self._sigma_ARW = np.array([0,0,0]) # 3-component vector
        self._sigma_RRW = np.array([0,0,0]) # 3-component vector
        self._bias = np.array([0,0,0]) # 3-component vector

        return

    # Factory methods
    @classmethod
    def getObserverModel(cls, obsRate):

        params = ()
        observer_coordinates = None # THIS IS NOT USED
        symbState = rateGyroObs.buildSymbolicState()
        inputSymb = rateGyroObs.buildSymbolicInput()
        obs_mod = rateGyroObs(symbState, params, observer_coordinates, obsRate, inputSymb)

        return obs_mod

    ## THIS IS THE METHOD TO BE MODIFIED IF A CHANGE IN THE STATE IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicState(cls):
        """
        Modify this method to build a new symbolic state vector.
        :return: A list with the symbolic symbols of the state.
        """
        # MINIMUM STATE: [x, y, z, x_dot, y_dot, z_dot]
        mrp1, mrp2, mrp3 = sp.symbols('mrp1 mrp2 mrp3')
        w1, w2, w3 = sp.symbols('w1 w2 w3')

        X_symb = [mrp1, mrp2, mrp3, w1, w2, w3]
        return X_symb

    def setNoise(self, flag, whiteNoiseMean, whiteNoiseCovariance, otherParams = None):
        """
        Set Noise model. This method OVERRIDES the original in order to add ARW and RRW
        otherParams is used if more complex models are to be used. In that case, this method has to be OVERRIDEN.
        :param flag: [boolean] Sets the noise on and off (True or False)
        :param whiteNoiseMean: NOT USED
        :param whiteNoiseCovariance: NOT USED
        :param otherParams: list with 2 arrays:
        sigma_ARW: [1-dimensional numpy array] [rad/sqrt(sec)] sigma^2 is the ARW (Angular Random Walk, white noise in w) PSD (Power Spectral Density).
        sigma_RRW: [1-dimensional numpy array] [rad/sec^(3/2)] sigma^2 is the RRW (Rate Random Walk, white noise in w_dot) PSD.
        initial_bias: [1-dimensional numpy array] Initial gyro bias (usually (0,0,0)).
        :return:
        """
        self._noiseFlag = flag
        self._sigma_ARW  = otherParams[0]   # 3-component vector
        self._sigma_RRW = otherParams[1]    # 3-component vector
        self._bias = otherParams[2]         # 3-component vector
        return

    def noiseModel(self):
        """
        Returns an additive white gaussian noise model.
        This method has to be OVERRRIDEN if more complex models are to be used.
        :return:
        """
        outs = self.getNmbrOutputs()
        if self._noiseFlag:
            Nv = np.random.multivariate_normal(np.zeros(outs), np.eye(outs))
            Nu = np.random.multivariate_normal(np.zeros(outs), np.eye(outs))

            delta_t = 1.0/self._obs_rate

            bias_next = self._bias + self._sigma_RRW*np.sqrt(delta_t) * Nu
            noise = 0.5*(self._bias + bias_next) + np.sqrt(self._sigma_ARW**2/delta_t + 1.0/12.0 * self._sigma_RRW**2*delta_t) * Nv
            self._bias = bias_next
            return noise
        else:
            return np.zeros(outs)

    ## -------------------------Public Interface--------------------------
    # This is the important interface por the estimation processors!

    def computeModel(self, X, t, params, u = None):
        """
        Computes the observation model function G: Y = G(X,t)
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The vector G(X,t) evaluating G.
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        w1 = X[3]
        w2 = X[4]
        w3 = X[5]

        # GS_nmbr = params
        # GS_coord = self.getObserverCoordinates()

        nmbrOfOutputs = self.getNmbrOutputs()

        G = np.zeros(nmbrOfOutputs)
        for i in range(0, nmbrOfOutputs):
            G[i] = self._modelLambda[i](w1, w2, w3)

        return G

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of G (Htilde).
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The jacobian evaluated at (X,t).
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        w1 = X[3]
        w2 = X[4]
        w3 = X[5]

        # GS_nmbr = params
        # GS_coord = self.getObserverCoordinates()

        nmbrOfOutputs = self.getNmbrOutputs()
        nmbrOfStates = self.getNmbrOfStates()

        Htilde = np.zeros([nmbrOfOutputs,nmbrOfStates])

        for i in range(0,nmbrOfOutputs):
            for j in range(0, nmbrOfStates) :
                Htilde[i][j] = self._jacobianLambda[i][j](w1, w2, w3)

        return Htilde


    def getNmbrOutputs(self):
        return 3

    def isObservable(self, X, t, params):
        """
        Method that decides if a state is observable for a given time and GS coordinates.
        :param X: State.
        :param t: Time.
        :param params: Dynamic parameters (GS coordinates).
        :return: Boolean indicating if it is visible.
        """

        # GS_number = params[0]
        # GS_coord = self.getObserverCoordinates()

        period_obs = 1.0/self._obs_rate

        # It is assumed that the observations cannot be taken faster than obs_rate.
        if self._last_obs == -1:
            self._last_obs = t
            return True
        elif self._last_obs + period_obs <= t:
            self._last_obs = t
            return True
        else:
            return False


    ## -------------------------Private Methods--------------------------

    def computeSymbolicModel(self):
        """
        Symbollically computes G(X,t) and stores the models and lambda functions.
        :return:
        """
        w1 = self._stateSymb[3]
        w2 = self._stateSymb[4]
        w3 = self._stateSymb[5]

        g1 = sp.lambdify((w1, w2, w3), w1, "numpy")
        g2 = sp.lambdify((w1, w2, w3), w2, "numpy")
        g3 = sp.lambdify((w1, w2, w3), w3, "numpy")

        self._modelSymb = [w1, w2, w3]
        self._modelLambda = [g1, g2, g3]

        return self._modelSymb

    def computeSymbolicJacobian(self):

        w1 = self._stateSymb[3]
        w2 = self._stateSymb[4]
        w3 = self._stateSymb[5]

        G = self._modelSymb

        nmbrOfStates = self.getNmbrOfStates()
        nmbrOfOutputs = self.getNmbrOutputs()

        dG = [[0 for i in range(0,nmbrOfStates)] for i in range(0,nmbrOfOutputs)]
        Htilde_lambda = [[0 for i in range(0,nmbrOfStates)] for i in range(0,nmbrOfOutputs)]

        for i in range(0, nmbrOfOutputs) :
            for j in range(0, nmbrOfStates) :
                dG[i][j] = sp.diff(G[i], self._stateSymb[j])
                Htilde_lambda[i][j] = sp.lambdify((w1, w2, w3), dG[i][j], "numpy")

        self._jacobianSymb_posVel = dG
        self._jacobianLambda_posVel = Htilde_lambda

        return self._jacobianSymb