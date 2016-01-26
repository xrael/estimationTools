######################################################
# Dynamic models.
#
# Manuel F. Diaz Ramos
#
# The classes here must inherit dynamicModelBase(modelBase)
######################################################

import sympy as sp
import numpy as np
from modelBase import dynamicModelBase

######################################################
# zonalHarmonicsModel.
#
# Manuel F. Diaz Ramos
#
# Zonal Harmonics Model.
######################################################
class zonalHarmonicsModel(dynamicModelBase):
    """
    Dynamic Model including Zonal Harmonics.
    """

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params):
        super(zonalHarmonicsModel, self).__init__(stateSymb, params)

        return

    ## -------------------------Class Methods--------------------------
    ## Factory method
    @classmethod
    def getDynamicModel(cls, mu, R_E, J, include_two_body_dynamics = True):
        """
        Factory method. Use it to get an instance of the class.
        :param degree: Maximum degree to be used
        :param mu: Gravitational parameter.
        :param R_E: Reference radius for the model
        :param J: Array with the J coefficients (J_0 is not used but should be included!)
        :return: An orbitZonalHarmonics object
        """

        params = (mu, R_E, J, include_two_body_dynamics)
        symbState = zonalHarmonicsModel.buildSymbolicState()
        zoneHarmModel = zonalHarmonicsModel(symbState, params)

        return zoneHarmModel

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
        X_symb = [x, y, z, x_dot, y_dot, z_dot]
        return X_symb

    ## -------------------------Public Interface--------------------------

    def computeModel(self, X, t, params):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :return: The result of the function in a vector with the same size than X.
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        mu = self._params[0]
        R_E = self._params[1]
        J = self._params[2]
        #-------------------------------------------

        nmbrOfStates = self.getNmbrOfStates()

        F = np.zeros(nmbrOfStates)
        for i in range(0, nmbrOfStates):
            F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J])

        return F

    # Returns A matrix
    # This is application-specific. It depends on how state vector is defined.
    def computeJacobian(self, X, t, params):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :return:
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        mu = self._params[0]
        R_E = self._params[1]
        J = self._params[2]
        #-------------------------------------------

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        for i in range(0,nmbrOfStates):
            for j in range(0,nmbrOfStates):
                A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J])

        return A

    ## -------------------------Private Methods--------------------------
    def _computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions in the attributes
        _potential, _acceleration, _potentialLambda, _accelerationLambda.
        :return:
        """
        J_params = self._params[2]
        degree = J_params.size - 1

        includeTwoBodyDynamics = self._params[3]

        x, y, z = sp.symbols('x y z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
        r = sp.sqrt(x**2 + y**2 + z**2)

        u = z/r

        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')

        if includeTwoBodyDynamics:
            U = mu/r
        else:
            U = 0

        J = sp.symarray('J', degree + 1)
        P = sp.symarray('P', degree + 1)
        P[0] = 1

        if degree > 0:
            P[1] = u
            for l in range(1, degree + 1):
                if l >= 2:
                    P[l] = ((u*(2*l-1) * P[l-1] - (l-1)*P[l-2])/l)
                    P[l].simplify()

                if J_params[l] != 0:
                    U = U - mu/r * (R_E/r)**l * J[l] * P[l]

        dUx = sp.diff(U, x)
        dUy = sp.diff(U, y)
        dUz = sp.diff(U, z)

        nmbrOfStates = self.getNmbrOfStates()

        self._modelSymb = []
        self._modelSymb.append(x_dot)
        self._modelSymb.append(y_dot)
        self._modelSymb.append(z_dot)
        self._modelSymb.append(dUx)
        self._modelSymb.append(dUy)
        self._modelSymb.append(dUz)

        for i in range(6, nmbrOfStates): # for every other state
            self._modelSymb.append(0)

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]
        for i in range(0, nmbrOfStates):
            self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J]), self._modelSymb[i], "numpy")

        return self._modelSymb


    def _computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions in the attributes _jacobian, _jacobianLambda.
        :return:
        """
        degree = self._params[2].size - 1

        x, y, z = sp.symbols('x, y, z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', degree + 1)

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        for i in range(0, nmbrOfStates) :
            F[i] = self._modelSymb[i]
            for j in range(0, nmbrOfStates) :
                dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J]), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb


######################################################
# dragModel.
#
# Manuel F. Diaz Ramos
#
# Drag Model.
######################################################
class dragModel(dynamicModelBase):

     ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params):
        super(dragModel, self).__init__(stateSymb, params)

        return

    ## Factory method
    @classmethod
    def getDynamicModel(cls, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot):

        params = (CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)
        symbState = zonalHarmonicsModel.buildSymbolicState()
        drModel = dragModel(symbState, params)

        return drModel

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
        X_symb = [x, y, z, x_dot, y_dot, z_dot]
        return X_symb

    ## -------------------------Public Interface--------------------------

    def computeModel(self, X, t, params):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :return: The result of the function in a vector with the same size than X.
        """

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        CD_drag = self._params[0]
        A_drag = self._params[1]
        mass_sat = self._params[2]
        rho_0_drag = self._params[3]
        r0_drag = self._params[4]
        H_drag = self._params[5]
        theta_dot = self._params[6]
        #-------------------------------------------

        nmbrOfStates = self.getNmbrOfStates()

        F = np.zeros(nmbrOfStates)
        for i in range(0, nmbrOfStates):
            F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)

        return F

    def computeJacobian(self, X, t, params):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :return:
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        CD_drag = self._params[0]
        A_drag = self._params[1]
        mass_sat = self._params[2]
        rho_0_drag = self._params[3]
        r0_drag = self._params[4]
        H_drag = self._params[5]
        theta_dot = self._params[6]
        #-------------------------------------------

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        for i in range(0,nmbrOfStates):
            for j in range(0,nmbrOfStates):
                A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)

        return A

    ## -------------------------Private Methods--------------------------
    def _computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions.
        :return:
        """
        x, y, z = sp.symbols('x y z')
        r = sp.sqrt(x**2 + y**2 + z**2)

        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, \
        H_drag, theta_dot = sp.symbols('CD_drag A_drag mass_sat rho_0_drag r0_drag H_drag theta_dot')

        Va = sp.sqrt((x_dot + theta_dot * y)**2 + (y_dot - theta_dot * x)**2 + z_dot**2)

        rho_A_drag = rho_0_drag*sp.exp(-(r-r0_drag)/H_drag)
        aux = -sp.Rational(1,2) * CD_drag * A_drag/mass_sat  * rho_A_drag * Va

        drag_acc1 = aux * (x_dot + theta_dot * y)
        drag_acc2 = aux * (y_dot - theta_dot * x)
        drag_acc3 = aux * (z_dot)

        nmbrOfStates = self.getNmbrOfStates()

        self._modelSymb = []
        self._modelSymb.append(x_dot)
        self._modelSymb.append(y_dot)
        self._modelSymb.append(z_dot)
        self._modelSymb.append(drag_acc1)
        self._modelSymb.append(drag_acc2)
        self._modelSymb.append(drag_acc3)

        for i in range(6, nmbrOfStates): # for every other state
            self._modelSymb.append(0)

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]
        for i in range(0, nmbrOfStates):
            self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), self._modelSymb[i], "numpy")

        return self._modelSymb

    def _computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        x, y, z = sp.symbols('x y z')

        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, \
        H_drag, theta_dot = sp.symbols('CD_drag A_drag mass_sat rho_0_drag r0_drag H_drag theta_dot')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        for i in range(0, nmbrOfStates) :
            F[i] = self._modelSymb[i]
            for j in range(0, nmbrOfStates) :
                dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb

######################################################
# dragZonalHarmonicModel.
#
# Manuel F. Diaz Ramos
#
# Zonal Harmonics + Drag Model.
######################################################
class dragZonalHarmonicModel(dynamicModelBase):

    _zonalHarmonicModel = None
    _dragModel = None

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params):
        super(dragZonalHarmonicModel, self).__init__(stateSymb, params)
        return

     ## Factory method
    @classmethod
    def getDynamicModel(cls, mu, R_E, J, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, include_two_body_dynamics = True):
        """
        Factory method. Use it to get an instance of the class.
        :param degree: Maximum degree to be used
        :param mu: Gravitational parameter.
        :param R_E: Reference radius for the model
        :param J: Array with the J coefficients (J_0 is not used but should be included!)
        :return: An orbitZonalHarmonics object
        """
        params = (mu, R_E, J, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, include_two_body_dynamics)
        symbState = dragZonalHarmonicModel.buildSymbolicState()
        zonHarmDragMod = dragZonalHarmonicModel(symbState, params)

        return zonHarmDragMod

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

        mu, CD_drag, J_2 = sp.symbols('mu CD_drag, J_2')
        X_GS1, Y_GS1, Z_GS1 = sp.symbols('X_GS1 Y_GS1 Z_GS1')
        X_GS2, Y_GS2, Z_GS2 = sp.symbols('X_GS2 Y_GS2 Z_GS2')
        X_GS3, Y_GS3, Z_GS3 = sp.symbols('X_GS3 Y_GS3 Z_GS3')

        X_symb = [x, y, z, x_dot, y_dot, z_dot, mu, CD_drag, J_2, X_GS1, Y_GS1, Z_GS1, X_GS2, Y_GS2, Z_GS2, X_GS3, Y_GS3, Z_GS3]
        return X_symb

    ## -------------------------Public Interface--------------------------

    def computeModel(self, X, t, params):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :return: The result of the function in a vector with the same size than X.
        """

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]
        mu = X[6] #params[0]
        R_E = self._params[1]
        J = np.array([0,0,X[7]]) #params[2]
        CD_drag = X[8] #params[3]
        A_drag = self._params[4]
        mass_sat = self._params[5]
        rho_0_drag = self._params[6]
        r0_drag = self._params[7]
        H_drag = self._params[8]
        theta_dot = self._params[9]

        nmbrOfStates = self.getNmbrOfStates()

        F = np.zeros(nmbrOfStates)
        for i in range(0, nmbrOfStates):
            F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)

        return F

    def computeJacobian(self, X, t, params):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :return:
        """

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]
        mu = X[6] #params[0]
        R_E = self._params[1]
        J = np.array([0,0,X[7]]) #params[2]
        CD_drag = X[8] #params[3]
        A_drag = self._params[4]
        mass_sat = self._params[5]
        rho_0_drag = self._params[6]
        r0_drag = self._params[7]
        H_drag = self._params[8]
        theta_dot = self._params[9]

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        for i in range(0,nmbrOfStates):
            for j in range(0,nmbrOfStates):
                A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot,
                                                     mu, R_E, [J], CD_drag, A_drag, mass_sat,
                                                     rho_0_drag, r0_drag, H_drag, theta_dot)

        return A

    ## -------------------------Private Methods--------------------------
    def _computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions.
        :return:
        """

        mu_param = self._params[0]
        R_E_param = self._params[1]
        J_param = self._params[2]
        CD_drag_param = self._params[3]
        A_drag_param = self._params[4]
        mass_sat_param = self._params[5]
        rho_0_drag_param = self._params[6]
        r0_drag_param = self._params[7]
        H_drag_param = self._params[8]
        theta_dot_param = self._params[9]
        include_two_body_dynamics_param = self._params[10]

        zonHarmMod = zonalHarmonicsModel.getDynamicModel(mu_param, R_E_param, J_param, include_two_body_dynamics_param)

        dragMod = dragModel.getDynamicModel(CD_drag_param, A_drag_param, mass_sat_param, rho_0_drag_param, r0_drag_param, H_drag_param, theta_dot_param)

        zonHarmSymbMod = zonHarmMod.getSymbolicModel()
        dragSymbMod = dragMod.getSymbolicModel()

        x, y, z = sp.symbols('x y z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', J_param.size)

        CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, \
        H_drag, theta_dot = sp.symbols('CD_drag A_drag mass_sat rho_0_drag r0_drag H_drag theta_dot')

        nmbrOfStates = self.getNmbrOfStates()

        self._modelSymb = []
        self._modelSymb.append(x_dot)
        self._modelSymb.append(y_dot)
        self._modelSymb.append(z_dot)
        self._modelSymb.append(zonHarmSymbMod[3] + dragSymbMod[3])
        self._modelSymb.append(zonHarmSymbMod[4] + dragSymbMod[4])
        self._modelSymb.append(zonHarmSymbMod[5] + dragSymbMod[5])

        for i in range(6, nmbrOfStates): # for every other state
            self._modelSymb.append(0)

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]
        for i in range(0, len(self._modelSymb)):
            self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), self._modelSymb[i], "numpy")

        return self._modelSymb


    def _computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        degree = self._params[2].size - 1

        x, y, z = sp.symbols('x y z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', degree + 1)

        CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, \
        H_drag, theta_dot = sp.symbols('CD_drag A_drag mass_sat rho_0_drag r0_drag H_drag theta_dot')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        for i in range(0, nmbrOfStates) :
            F[i] = self._modelSymb[i]
            for j in range(0, nmbrOfStates) :
                dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb
