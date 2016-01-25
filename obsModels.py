######################################################
# Observation Models
#
# Manuel F. Diaz Ramos
#
# These classes compute symbollically and numerically
# observation models and their derivatives.
######################################################

from modelBase import observerModelBase
import coordinateTransformations
import sympy as sp
import numpy as np


### Range and Range-rate observations
class rangeRangeRateObsModel(observerModelBase):

    _jacobianSymb_posVel = None
    _jacobianLambda_posVel = None
    _jacobianSymb_GSpos = None
    _jacobianLambda_GSpos = None

    def __init__(self, stateSymb, params, observerCoordinates):
        super(rangeRangeRateObsModel, self).__init__(stateSymb, params, observerCoordinates)

        return

    # Factory methods
    @classmethod
    def getObserverModel(cls, theta_0, angular_rate, R_eq, e_planet, GS_coordinates):

        params = (theta_0, angular_rate, R_eq, e_planet)
        symbState = rangeRangeRateObsModel.buildSymbolicState()
        obs_mod = rangeRangeRateObsModel(symbState, params, GS_coordinates)

        return obs_mod

    ## THIS IS THE METHOD TO BE MODIFIED IF A CHANGE IN THE STATE IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicState(cls):
        """
        Modify this method to build a new symbolic state vector.
        :return: A list with the symbolic symbols of the state.
        """
        # MINIMUM STATE: [x, y, z, x_dot, y_dot, z_dot]
        x, y, z = sp.symbols('x, y, z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        #mu, CD_drag, J_2 = sp.symbols('mu CD_drag, J_2')
        #x_gs, y_gs, z_gs = sp.symbols('x_gs y_gs z_gs')
        # X_GS1, Y_GS1, Z_GS1 = sp.symbols('X_GS1 Y_GS1 Z_GS1')
        # X_GS2, Y_GS2, Z_GS2 = sp.symbols('X_GS2 Y_GS2 Z_GS2')
        # X_GS3, Y_GS3, Z_GS3 = sp.symbols('X_GS3 Y_GS3 Z_GS3')

        #X_symb = [x, y, z, x_dot, y_dot, z_dot, mu, CD_drag, J_2, x_gs, y_gs, z_gs, x_gs, y_gs, z_gs, x_gs, y_gs, z_gs]
        X_symb = [x, y, z, x_dot, y_dot, z_dot]
        return X_symb

    ## -------------------------Public Interface--------------------------
    # This is the important interface por the estimation processors!

    def computeModel(self, X, t, params):
        """
        Computes the observation model function G: Y = G(X,t)
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The vector G(X,t) evaluating G.
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # x_gs1 = X[9]
        # y_gs1 = X[10]
        # z_gs1 = X[11]
        # x_gs2 = X[12]
        # y_gs2 = X[13]
        # z_gs2 = X[14]
        # x_gs3 = X[15]
        # y_gs3 = X[16]
        # z_gs3 = X[17]
        theta_0 = self._params[0]
        theta_dot = self._params[1]
        GS_nmbr = params

        GS_coord = self.getObserverCoordinates()

        x_gs = GS_coord[GS_nmbr][0]
        y_gs = GS_coord[GS_nmbr][1]
        z_gs = GS_coord[GS_nmbr][2]

        # if GS_nmbr == 0:
        #     x_gs = x_gs1
        #     y_gs = y_gs1
        #     z_gs = z_gs1
        # elif GS_nmbr == 1:
        #     x_gs = x_gs2
        #     y_gs = y_gs2
        #     z_gs = z_gs2
        # else : # GS_nmbr == 2
        #     x_gs = x_gs3
        #     y_gs = y_gs3
        #     z_gs = z_gs3

        theta = theta_dot * t + theta_0

        nmbrOfOutputs = self.getNmbrOutputs()

        G = np.zeros(nmbrOfOutputs)
        for i in range(0, nmbrOfOutputs):
            G[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot)

        return G

    def computeJacobian(self, X, t, params):
        """
        Computes the Jacobian of G (Htilde).
        :param X: State.
        :param t: Time.
        :param params: Parameters of the model.
        :return: The jacobian evaluated at (X,t).
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        # x_gs1 = X[9]
        # y_gs1 = X[10]
        # z_gs1 = X[11]
        # x_gs2 = X[12]
        # y_gs2 = X[13]
        # z_gs2 = X[14]
        # x_gs3 = X[15]
        # y_gs3 = X[16]
        # z_gs3 = X[17]
        #-------------------------------------------
        theta_0 = self._params[0]
        theta_dot = self._params[1]
        GS_nmbr = params

        theta = theta_dot * t + theta_0

        GS_coord = self.getObserverCoordinates()

        x_gs = GS_coord[GS_nmbr][0]
        y_gs = GS_coord[GS_nmbr][1]
        z_gs = GS_coord[GS_nmbr][2]

        Htilde = np.zeros([2,6])

        # if GS_nmbr == 0:
        #     x_gs = x_gs1
        #     y_gs = y_gs1
        #     z_gs = z_gs1
        #     for i in range(0,2):
        #         for j in range(9, 12) :
        #             Htilde[i][j] = self._jacobianLambda_GSpos[i][j-9](x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot)
        #
        # elif GS_nmbr == 1:
        #     x_gs = x_gs2
        #     y_gs = y_gs2
        #     z_gs = z_gs2
        #     for i in range(0,2):
        #         for j in range(12, 15):
        #             Htilde[i][j] = self._jacobianLambda_GSpos[i][j-12](x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot)
        #
        # else : # GS_nmbr == 2
        #     x_gs = x_gs3
        #     y_gs = y_gs3
        #     z_gs = z_gs3
        #     for i in range(0,2):
        #         for j in range(15, 18) :
        #              Htilde[i][j] = self._jacobianLambda_GSpos[i][j-15](x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot)

        for i in range(0,2):
            for j in range(0, 6) :
                Htilde[i][j] = self._jacobianLambda_posVel[i][j](x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot)

        return Htilde

    def getNmbrOutputs(self):
        return 2

    def isObservable(self, X, t, params):
        """
        Method that decides if a state is observable for a given time and GS coordinates.
        :param X: State.
        :param t: Time.
        :param params: Dynamic parameters (GS coordinates).
        :return: Boolean indicating if it is visible.
        """
        x = X[0]
        y = X[1]
        z = X[2]

        theta_0 = self._params[0]
        theta_dot = self._params[1]
        R_eq = self._params[2]
        e_planet = self._params[3]

        GS_number = params[0]

        GS_coord = self.getObserverCoordinates()

        r_gs_ecef = GS_coord[GS_number]

        r_eci_vec = np.array([x,y,z])
        GMST = theta_dot * t + theta_0 # Greenwhich Mean Sidereal Time

        r_ecef_vec = coordinateTransformations.eci2ecef(r_eci_vec, GMST)

        azElRange = coordinateTransformations.ecef2AzElRange(r_ecef_vec, r_gs_ecef, R_eq, e_planet)

        if azElRange[1] > np.deg2rad(10): # Elevation Mask 10 deg
            return True
        else:
            return False


    ## -------------------------Private Methods--------------------------

    def _computeSymbolicModel(self):
        """
        Symbollically computes G(X,t) and stores the models and lambda functions.
        :return:
        """
        # satellite position in ECI
        x, y, z = sp.symbols('x y z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
        # Ground station position in ECEF
        x_gs, y_gs, z_gs = sp.symbols('x_gs y_gs z_gs')
        # Derivatives of Ground Station position in ECEF are not considered

        theta = sp.symbols('theta')             # GMST
        theta_dot = sp.symbols('theta_dot')

        # Rotation from ECEF to ECI is included

        # Range
        range = sp.sqrt(x**2 + y**2 + z**2 + x_gs**2 + y_gs**2 + z_gs**2
            - 2.0*(x*x_gs + y*y_gs)*sp.cos(theta) + 2.0*(x*y_gs - x_gs*y)* sp.sin(theta)
            - 2.0*z*z_gs)

        # Range rate
        range_rate = (x*x_dot + y*y_dot + z*z_dot - (x_dot*x_gs + y_dot*y_gs)*sp.cos(theta) +
            theta_dot*(x*x_gs + y*y_gs)*sp.sin(theta) + (x_dot*y_gs - y_dot*x_gs)*sp.sin(theta)
            + theta_dot*(x*y_gs - y*x_gs)*sp.cos(theta) - z_dot*z_gs)/range

        g1 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot), range, "numpy")
        g2 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot), range_rate, "numpy")

        self._modelSymb = [range, range_rate]
        self._modelLambda = [g1, g2]

        return self._modelSymb

    def _computeSymbolicJacobian(self):
        self._getObsModelDerivativesFromPosVel()
        self._getObsModelDerivativesFromGSpos()

        self._jacobianSymb = self._jacobianSymb_posVel
        self._jacobianLambda = self._jacobianLambda_posVel

        return self._jacobianSymb

    def _getObsModelDerivativesFromPosVel(self):
        G = self._modelSymb

        # satellite position in ECI
        x, y, z = sp.symbols('x y z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
        # Ground station position in ECEF
        x_gs, y_gs, z_gs = sp.symbols('x_gs y_gs z_gs')

        X = [x, y, z, x_dot, y_dot, z_dot] # State

        theta = sp.symbols('theta')
        theta_dot = sp.symbols('theta_dot')

        dG = [[0 for i in range(6)] for i in range(2)]
        Htilde_lambda = [[0 for i in range(6)] for i in range(2)]

        for i in range(0, 2) :
            for j in range(0, 6) :
                dG[i][j] = sp.diff(G[i], X[j])
                Htilde_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot), dG[i][j], "numpy")

        self._jacobianSymb_posVel = dG
        self._jacobianLambda_posVel = Htilde_lambda

        return self._jacobianSymb_posVel

    def _getObsModelDerivativesFromGSpos(self):
        G = self._modelSymb

        # satellite position in ECI
        x, y, z = sp.symbols('x y z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
        # Ground Station Position in ECEF
        x_gs, y_gs, z_gs = sp.symbols('x_gs y_gs z_gs')

        X = [x_gs, y_gs, z_gs] # State

        theta = sp.symbols('theta')
        theta_dot = sp.symbols('theta_dot')

        dG = [[0 for i in range(3)] for i in range(2)]
        Htilde_lambda = [[0 for i in range(3)] for i in range(2)]

        for i in range(0, 2) :
            for j in range(0, 3) :
                dG[i][j] = sp.diff(G[i],X[j])
                Htilde_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_gs, y_gs, z_gs, theta, theta_dot), dG[i][j], "numpy")

        self._jacobianSymb_GSpos = dG
        self._jacobianLambda_GSpos = Htilde_lambda

        return self._jacobianSymb_GSpos

