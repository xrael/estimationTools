#######################################################################################################################
# Dynamic models.
#
# Manuel F. Diaz Ramos
#
# The classes here must inherit dynamicModelBase(modelBase).
#
# Includes: gravity zonal harmonics, drag, solar radiation pressure and three body gravity.
#######################################################################################################################

import sympy as sp
import numpy as np
import ephemeris as eph
import orbitalElements as orbEl
from coordinateTransformations import ROT1, ROT2, ROT3
from modelBase import orbitalDynamicModelBase

#######################################################################################################################
# zonalHarmonicsModel.
#
# Manuel F. Diaz Ramos
#
# Zonal Harmonics Model.
#######################################################################################################################
class zonalHarmonicsModel(orbitalDynamicModelBase):
    """
    Dynamic Model including Zonal Harmonics.
    """

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(zonalHarmonicsModel, self).__init__("ZonalHarmonics", stateSymb, params, propagationFunction, inputSymb)

        return

    ## -------------------------Class Methods--------------------------
    ## Factory method
    @classmethod
    def getDynamicModel(cls, mu, R_E, J, include_two_body_dynamics = True, propagationFunction = 'F'):
        """
        Factory method. Use it to get an instance of the class.
        :param propagationFunction: Method to use for propagation: 'F' (Use F(X,t)), 'F_plus_STM' (Use F and STM), 'F_vector (propagate many states in parallel).
        :param mu: Gravitational parameter.
        :param R_E: Reference radius for the model
        :param J: Array with the J coefficients (J_0 is not used but should be included!)
        :param include_two_body_dynamics: True for computing the two-body dynamics term.
        :param propagationFunction: Method to use for propagation: 'F' (Use F(X,t)), 'F_plus_STM' (Use F and STM), 'F_vector (propagate many states in parallel).
        :return: An orbitZonalHarmonics object
        """
        params = (mu, R_E, J, include_two_body_dynamics)
        symbState = zonalHarmonicsModel.buildSymbolicState()
        inputSymb = zonalHarmonicsModel.buildSymbolicInput()
        zoneHarmModel = zonalHarmonicsModel(symbState, params, propagationFunction, inputSymb)

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

    # ## THIS IS THE METHOD TO OVERRIDE IF A CONSIDER PARAMETER ANALYSIS IS TO BE ACCOMPLISHED!!!
    # @classmethod
    # def buildSymbolicInput(cls):
    #     """
    #     Modify this method to build a new symbolic input vector (for control force or consider covariance analysis).
    #     :return: A list with the symbolic symbols of the input.
    #     """
    #     # DEFAULT
    #     J_3 = sp.symbols('J_3')
    #     U_symb = [J_3]
    #     return U_symb

    # def getInput(self, t):
    #     J = self._params[2]
    #     return J[3]

    ## -------------------------Public Interface--------------------------
    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State vector.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
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

        if self._usingDMC:
            w_x = X[-3] # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], [B])
        else:
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J])

        return F

    # Returns A matrix
    # This is application-specific. It depends on how state vector is defined.
    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :param u: Input vector
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

        if self._usingDMC:
            w_x = X[-3] # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J])

        return A

    def computeInputJacobian(self, X, t, params, u):
        """
        Computes the Jacobian wrt the input of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :param u: Input vector
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
        nmbrOfInputs = self.getNmbrInputs()
        B_input = np.zeros([nmbrOfStates,nmbrOfInputs])

        if self._usingDMC:
            w_x = X[-3] # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfInputs):
                    B_input[i][j] = self._jacobianInputLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfInputs):
                    B_input[i][j] = self._jacobianInputLambda[i][j](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J])

        return B_input


    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions in the attributes
        _potential, _acceleration, _potentialLambda, _accelerationLambda.
        :return:
        """
        J_params = self._params[2]
        degree = J_params.size - 1

        includeTwoBodyDynamics = self._params[3]

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

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

                #if J_params[l] != 0:
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

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            for i in range(6, nmbrOfStates-3): # for every other state
                self._modelSymb.append(0)
            w_x = self._stateSymb[-3] # DMC at the end of the state
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            self._modelSymb[3] += w_x
            self._modelSymb[4] += w_y
            self._modelSymb[5] += w_z
            self._modelSymb.append(-B[0]*w_x)
            self._modelSymb.append(-B[1]*w_y)
            self._modelSymb.append(-B[2]*w_z)

            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], [B]), self._modelSymb[i], "numpy")
        else:
            for i in range(6, nmbrOfStates): # for every other state
                self._modelSymb.append(0)
            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J]), self._modelSymb[i], "numpy")

        return self._modelSymb


    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions in the attributes _jacobian, _jacobianLambda.
        :return:
        """
        degree = self._params[2].size - 1

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', degree + 1)

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J]), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb

    def computeSymbolicInputJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions in the attributes _jacobian, _jacobianLambda.
        :return:
        """
        degree = self._params[2].size - 1

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', degree + 1)

        nmbrOfStates = self.getNmbrOfStates()
        nmbrOfInputs = self.getNmbrInputs()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfInputs)] for i in range(0, nmbrOfStates)]
        B_lambda = [[0 for i in range(0, nmbrOfInputs)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfInputs) :
                    dF[i][j] = sp.diff(F[i], self._inputSymb[j])
                    B_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfInputs) :
                    dF[i][j] = sp.diff(F[i], self._inputSymb[j])
                    B_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J]), dF[i][j], "numpy")

        self._jacobianInputSymb = dF
        self._jacobianInputLambda = B_lambda

        return self._jacobianInputSymb

#######################################################################################################################

#######################################################################################################################
# dragModel.
#
# Manuel F. Diaz Ramos
#
# Drag Model.
#######################################################################################################################
class dragModel(orbitalDynamicModelBase):

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(dragModel, self).__init__("drag", stateSymb, params, propagationFunction, inputSymb)

        return

    ## Factory method
    @classmethod
    def getDynamicModel(cls, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, propagationFunction = 'F'):

        params = (CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)
        symbState = dragModel.buildSymbolicState()
        inputSymb = dragModel.buildSymbolicInput()
        drModel = dragModel(symbState, params, propagationFunction, inputSymb)

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
    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
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

        if self._usingDMC:
            w_x = X[-3] # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B])
        else:
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)

        return F

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :param u: Input vector.
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

        if self._usingDMC:
            w_x = X[-3]  # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)

        return A

    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions.
        :return:
        """
        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        r = sp.sqrt(x**2 + y**2 + z**2)

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

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            for i in range(6, nmbrOfStates-3): # for every other state
                self._modelSymb.append(0)
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            self._modelSymb[3] += w_x
            self._modelSymb[4] += w_y
            self._modelSymb[5] += w_z
            self._modelSymb.append(-B[0]*w_x)
            self._modelSymb.append(-B[1]*w_y)
            self._modelSymb.append(-B[2]*w_z)

            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B]), self._modelSymb[i], "numpy")
        else:
            for i in range(6, nmbrOfStates): # for every other state
                self._modelSymb.append(0)
            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), self._modelSymb[i], "numpy")

        return self._modelSymb

    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, \
        H_drag, theta_dot = sp.symbols('CD_drag A_drag mass_sat rho_0_drag r0_drag H_drag theta_dot')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb
#######################################################################################################################


#######################################################################################################################
# SolarRadiationPressure model.
#
# Manuel F. Diaz Ramos
#
# Cannonball solar radiation pressure model.
#######################################################################################################################
class solarRadiationPressureModel(orbitalDynamicModelBase):

     ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(solarRadiationPressureModel, self).__init__("SRP", stateSymb, params, propagationFunction, inputSymb)

        return

    ## Factory method
    @classmethod
    def getDynamicModel(cls, C_R, A_m_ratio, R_1AU, srp_flux, speed_light, JD_0, a_meeus, inc_ecliptic, mu_sun, propagationFunction = 'F'):
        """

        :param C_R: SRP Reflectiity coefficient.
        :param A_m_ratio: Area-to-mass ratio [m^2/kg].
        :param R_1AU: 1 AU distance
        :param srp_flux: SRP flux [W/m^2].
        :param speed_light: Speed of light [m/s].
        :param JD_0: Initial Julian Date.
        :param a_meeus: [2-dimensional numpy array] Meeus coefficients. Used to compute the position of the reference planet (if the reference is the sun, all coefficients have to be zero).
        The position of the planet is given in the EMO-J2000 frame. The position needs to be rotated to use the inertial frame of the reference planet,
        usually in the equatorial plane.
        :param inc_ecliptic: [rad] Angle of inclination of the equatorial plane of the reference planet relative to the equatorial plane.
        :param propagationFunction: Method to use for propagation: 'F' (Use F(X,t)), 'F_plus_STM' (Use F and STM), 'F_vector (propagate many states in parallel).
        :return:
        """
        params = (C_R, A_m_ratio, R_1AU, srp_flux, speed_light, JD_0, a_meeus, inc_ecliptic, mu_sun)
        symbState = solarRadiationPressureModel.buildSymbolicState()
        inputSymb = solarRadiationPressureModel.buildSymbolicInput()
        srpModel = solarRadiationPressureModel(symbState, params, propagationFunction, inputSymb)

        return srpModel

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
    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
        :return: The result of the function in a vector with the same size than X.
        """

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        C_R = self._params[0]
        A_m = self._params[1]
        R_1AU = self._params[2]
        srp_flux = self._params[3]
        c = self._params[4]
        #-------------------------------------------

        x_sun_ref = params[0]
        y_sun_ref = params[1]
        z_sun_ref = params[2]

        nmbrOfStates = self.getNmbrOfStates()
        F = np.zeros(nmbrOfStates)

        if self._usingDMC:
            w_x = X[-3] # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c, [B])
        else:
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c)

        return F

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :param u: Input vector.
        :return:
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        C_R = self._params[0]
        A_m = self._params[1]
        R_1AU = self._params[2]
        srp_flux = self._params[3]
        c = self._params[4]
        #-------------------------------------------

        x_sun_ref = params[0]
        y_sun_ref = params[1]
        z_sun_ref = params[2]

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        if self._usingDMC:
            w_x = X[-3]
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c, [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c)

        return A

    def computeTimeDependentParameters(self, t):
        """
        OVERRIDEN METHOD
        Computes the sun position relative to the reference planet.
        :param t:
        :return:
        """
        JD_0 = self._params[5]
        a_meeus = self._params[6]
        inc_ecliptic = self._params[7]
        mu_sun = self._params[8]

        JD = eph.JDplusSeconds(JD_0, t) # Computes the new Julian Date

        # Computes the position of the reference planet wrt the sun.
        (a, e, i, raan, w, nu) = \
            eph.computeOrbitalElementsMeeus(a_meeus[0], a_meeus[1], a_meeus[2], a_meeus[3], a_meeus[4], a_meeus[5], JD)
        (r_ref_sun, v_ref_sun, ret) = orbEl.orbitalElements2PositionVelocity(mu_sun, a, e, i, raan, w, nu)
        # The position is given in the Earth Mean Orbital plane at J2000 reference frame
        # The position has to be rotated to the equatorial plane of the planet around the x axis (vernal equinox)
        r_ref_sun = ROT1(-inc_ecliptic).dot(r_ref_sun) # CHECK THE SIGN!!!

        x_sun_ref = -r_ref_sun[0]
        y_sun_ref = -r_ref_sun[1]
        z_sun_ref = -r_ref_sun[2]

        return (x_sun_ref, y_sun_ref, z_sun_ref)

    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions.
        :return:
        """
        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        C_R, A_m, R_1AU, srp_flux, c = sp.symbols('C_R A_m R_1AU srp_flux c')

        # Position of the sun relative to the reference from which (x, y, z) is computed
        x_sun_ref = sp.symbols('x_sun_ref')
        y_sun_ref = sp.symbols('y_sun_ref')
        z_sun_ref = sp.symbols('z_sun_ref')

        # Position of the sun relative to the spacecraft
        x_sun_sc = x_sun_ref - x
        y_sun_sc = y_sun_ref - y
        z_sun_sc = z_sun_ref - z

        r_sun_sc = sp.sqrt(x_sun_sc**2 + y_sun_sc**2 + z_sun_sc**2)

        coeff = -C_R * srp_flux/c * R_1AU**2/r_sun_sc**3 * A_m

        srp_1 = coeff * x_sun_sc
        srp_2 = coeff * y_sun_sc
        srp_3 = coeff * z_sun_sc

        nmbrOfStates = self.getNmbrOfStates()

        self._modelSymb = []
        self._modelSymb.append(x_dot)
        self._modelSymb.append(y_dot)
        self._modelSymb.append(z_dot)
        self._modelSymb.append(srp_1)
        self._modelSymb.append(srp_2)
        self._modelSymb.append(srp_3)

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            for i in range(6, nmbrOfStates-3): # for every other state
                self._modelSymb.append(0)
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            self._modelSymb[3] += w_x
            self._modelSymb[4] += w_y
            self._modelSymb[5] += w_z
            self._modelSymb.append(-B[0]*w_x)
            self._modelSymb.append(-B[1]*w_y)
            self._modelSymb.append(-B[2]*w_z)

            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c, [B]), self._modelSymb[i], "numpy")
        else:
            for i in range(6, nmbrOfStates): # for every other state
                self._modelSymb.append(0)
            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c), self._modelSymb[i], "numpy")

        return self._modelSymb

    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        C_R, A_m, R_1AU, srp_flux, c = sp.symbols('C_R A_m R_1AU srp_flux c')

        # Position of the sun relative to the reference from which (x, y, z) is computed
        x_sun_ref = sp.symbols('x_sun_ref')
        y_sun_ref = sp.symbols('y_sun_ref')
        z_sun_ref = sp.symbols('z_sun_ref')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c, [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, C_R, A_m, R_1AU, srp_flux, c), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb
#######################################################################################################################

#######################################################################################################################
# thirdBodyGravityModel
#
# Manuel F. Diaz Ramos
#
# Third body gravity model.
# TODO: For now, it only accounts for the sun third body effect.
#######################################################################################################################
class thirdBodyGravityModel(orbitalDynamicModelBase):

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(thirdBodyGravityModel, self).__init__("third_body", stateSymb, params, propagationFunction, inputSymb)

        return

    ## Factory method
    @classmethod
    def getDynamicModel(cls, mu_third, JD_0, a_meeus, inc_ecliptic, propagationFunction = 'F'):

        params = (mu_third, JD_0, a_meeus, inc_ecliptic)
        symbState = thirdBodyGravityModel.buildSymbolicState()
        inputSymb = thirdBodyGravityModel.buildSymbolicInput()
        thirdGravModel = thirdBodyGravityModel(symbState, params, propagationFunction, inputSymb)

        return thirdGravModel

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
    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
        :return: The result of the function in a vector with the same size than X.
        """

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        mu_third = self._params[0]
        #-------------------------------------------

        x_third_ref = params[0]
        y_third_ref = params[1]
        z_third_ref = params[2]

        nmbrOfStates = self.getNmbrOfStates()
        F = np.zeros(nmbrOfStates)

        if self._usingDMC:
            w_x = X[-3] # DMC is at the end of the state
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_third_ref, y_third_ref, z_third_ref, mu_third, [B])
        else:
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, x_third_ref, y_third_ref, z_third_ref, mu_third)

        return F

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model.
        :param u: Input vector.
        :return:
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        mu_third = self._params[0]
        #-------------------------------------------

        x_third_ref = params[0]
        y_third_ref = params[1]
        z_third_ref = params[2]

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        if self._usingDMC:
            w_x = X[6]
            w_y = X[7]
            w_z = X[8]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_third_ref, y_third_ref, z_third_ref, mu_third, [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, x_third_ref, y_third_ref, z_third_ref, mu_third)

        return A

    def computeTimeDependentParameters(self, t):
        """
        OVERRIDEN METHOD.
        Computes the third body position relative to the reference planet.
        :param t:
        :return:
        """
        # TODO: THIS METHOD ONLY WORKS FOR THE SUN AS A THIRD BODY. Change it for another celestial body.

        mu_third = self._params[0] # TODO: Assumming mu_third is mu_sun
        JD_0 = self._params[1]
        a_meeus = self._params[2]
        inc_ecliptic = self._params[3]

        JD = eph.JDplusSeconds(JD_0, t) # Computes the new Julian Date

        # Computes the position of the reference planet wrt the sun.
        (a, e, i, raan, w, nu) = \
            eph.computeOrbitalElementsMeeus(a_meeus[0], a_meeus[1], a_meeus[2], a_meeus[3], a_meeus[4], a_meeus[5], JD)
        (r_ref_sun, v_ref_sun, ret) = orbEl.orbitalElements2PositionVelocity(mu_third, a, e, i, raan, w, nu)
        # The position is given in the Earth Mean Orbital plane at J2000 reference frame
        # The position has to be rotated to the equatorial plane of the planet around the x axis (vernal equinox)
        r_ref_sun = ROT1(-inc_ecliptic).dot(r_ref_sun) # TODO: CHECK THE SIGN!!!

        x_sun_ref = -r_ref_sun[0]
        y_sun_ref = -r_ref_sun[1]
        z_sun_ref = -r_ref_sun[2]

        return (x_sun_ref, y_sun_ref, z_sun_ref)

    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions.
        :return:
        """
        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        mu_third = sp.symbols('mu_third')

        # Position of the third body relative to the reference from which (x, y, z) is computed
        x_third_ref = sp.symbols('x_third_ref')
        y_third_ref = sp.symbols('y_third_ref')
        z_third_ref = sp.symbols('z_third_ref')

        r_third_ref = sp.sqrt(x_third_ref**2 + y_third_ref**2 + z_third_ref**2)

        # Position of the third body relative to the spacecraft
        x_third_sc = x_third_ref - x
        y_third_sc = y_third_ref - y
        z_third_sc = z_third_ref - z

        r_third_sc = sp.sqrt(x_third_sc**2 + y_third_sc**2 + z_third_sc**2)

        third_body_1 = mu_third*(x_third_sc/r_third_sc**3 - x_third_ref/r_third_ref**3)
        third_body_2 = mu_third*(y_third_sc/r_third_sc**3 - y_third_ref/r_third_ref**3)
        third_body_3 = mu_third*(z_third_sc/r_third_sc**3 - z_third_ref/r_third_ref**3)

        nmbrOfStates = self.getNmbrOfStates()

        self._modelSymb = []
        self._modelSymb.append(x_dot)
        self._modelSymb.append(y_dot)
        self._modelSymb.append(z_dot)
        self._modelSymb.append(third_body_1)
        self._modelSymb.append(third_body_2)
        self._modelSymb.append(third_body_3)

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            for i in range(6, nmbrOfStates-3): # for every other state
                self._modelSymb.append(0)
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            self._modelSymb[3] += w_x
            self._modelSymb[4] += w_y
            self._modelSymb[5] += w_z
            self._modelSymb.append(-B[0]*w_x)
            self._modelSymb.append(-B[1]*w_y)
            self._modelSymb.append(-B[2]*w_z)

            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_third_ref, y_third_ref, z_third_ref, mu_third, [B]), self._modelSymb[i], "numpy")
        else:
            for i in range(6, nmbrOfStates): # for every other state
                self._modelSymb.append(0)
            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_third_ref, y_third_ref, z_third_ref, mu_third), self._modelSymb[i], "numpy")

        return self._modelSymb

    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]
        mu_third = sp.symbols('mu_third')

        # Position of the third body relative to the reference from which (x, y, z) is computed
        x_third_ref = sp.symbols('x_third_ref')
        y_third_ref = sp.symbols('y_third_ref')
        z_third_ref = sp.symbols('z_third_ref')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_third_ref, y_third_ref, z_third_ref, mu_third, [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_third_ref, y_third_ref, z_third_ref, mu_third), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb
#######################################################################################################################


#######################################################################################################################
# dragZonalHarmonicModel.
#
# Manuel F. Diaz Ramos
#
# Zonal Harmonics + Drag Model.
#######################################################################################################################
class dragZonalHarmonicModel(orbitalDynamicModelBase):

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(dragZonalHarmonicModel, self).__init__("ZonalHarmonicsPlusDrag", stateSymb, params, propagationFunction, inputSymb)
        return

     ## Factory method
    @classmethod
    def getDynamicModel(cls, mu, R_E, J, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, include_two_body_dynamics = True, propagationFunction = 'F'):
        """
        Factory method. Use it to get an instance of the class.
        :param mu: Gravitational parameter.
        :param R_E: Reference radius for the model
        :param J: Array with the J coefficients (J_0 is not used but should be included!)
        :return: An orbitZonalHarmonics object
        """
        params = (mu, R_E, J, CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, include_two_body_dynamics)
        symbState = dragZonalHarmonicModel.buildSymbolicState()
        inputSymb = dragZonalHarmonicModel.buildSymbolicInput()
        zonHarmDragMod = dragZonalHarmonicModel(symbState, params, propagationFunction, inputSymb)

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

        X_symb = [x, y, z, x_dot, y_dot, z_dot, mu, J_2, CD_drag, X_GS1, Y_GS1, Z_GS1, X_GS2, Y_GS2, Z_GS2, X_GS3, Y_GS3, Z_GS3]
        #X_symb = [x, y, z, x_dot, y_dot, z_dot]
        return X_symb

    ## -------------------------Public Interface--------------------------

    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
        :return: The result of the function in a vector with the same size than X.
        """

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # Change this part for adding more states
        # mu = self._params[0]
        mu = X[6]
        R_E = self._params[1]
        #J = self._params[2]
        J = np.array([0, 0, X[7]])
        # CD_drag = self._params[3]
        CD_drag = X[8]
        A_drag = self._params[4]
        mass_sat = self._params[5]
        rho_0_drag = self._params[6]
        r0_drag = self._params[7]
        H_drag = self._params[8]
        theta_dot = self._params[9]
        #---------------------------------

        nmbrOfStates = self.getNmbrOfStates()
        F = np.zeros(nmbrOfStates)

        if self._usingDMC:
            w_x = X[-3]
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B])
        else:
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot)

        return F

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model.
        :param u: Input vector.
        :return:
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]

        # Change this part for adding more states
        mu = X[6]
        R_E = self._params[1]
        #J = self._params[2]
        J = np.array([0, 0, X[7]])
        # CD_drag = self._params[3]
        CD_drag = X[8]
        A_drag = self._params[4]
        mass_sat = self._params[5]
        rho_0_drag = self._params[6]
        r0_drag = self._params[7]
        H_drag = self._params[8]
        theta_dot = self._params[9]
        #---------------------------------

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        if self._usingDMC:
            w_x = X[-3]
            w_y = X[-2]
            w_z = X[-1]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z,
                                                     mu, R_E, [J], CD_drag, A_drag, mass_sat,
                                                     rho_0_drag, r0_drag, H_drag, theta_dot, [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot,
                                                     mu, R_E, [J], CD_drag, A_drag, mass_sat,
                                                     rho_0_drag, r0_drag, H_drag, theta_dot)

        return A

    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
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

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

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

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            for i in range(6, nmbrOfStates-3): # for every other state
                self._modelSymb.append(0)
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            self._modelSymb[3] += w_x
            self._modelSymb[4] += w_y
            self._modelSymb[5] += w_z
            self._modelSymb.append(-B[0]*w_x)
            self._modelSymb.append(-B[1]*w_y)
            self._modelSymb.append(-B[2]*w_z)

            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B]), self._modelSymb[i], "numpy")
        else:
            for i in range(6, nmbrOfStates): # for every other state
                self._modelSymb.append(0)
            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), self._modelSymb[i], "numpy")

        return self._modelSymb


    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        degree = self._params[2].size - 1

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', degree + 1)

        CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, \
        H_drag, theta_dot = sp.symbols('CD_drag A_drag mass_sat rho_0_drag r0_drag H_drag theta_dot')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot, [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, [J], CD_drag, A_drag, mass_sat, rho_0_drag, r0_drag, H_drag, theta_dot), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb


#######################################################################################################################
# zonalHarmonicThirdBodySRPModel.
#
# Manuel F. Diaz Ramos
#
# Zonal Harmonics + Third body + Solar Radiation Pressure Model.
#######################################################################################################################
class zonalHarmonicThirdBodySRPModel(orbitalDynamicModelBase):

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(zonalHarmonicThirdBodySRPModel, self).__init__("ZonalHarmonics_ThirdBody_SRP", stateSymb, params, propagationFunction, inputSymb)
        return

     ## Factory method
    @classmethod
    def getDynamicModel(cls, mu, R_E, J, mu_third, mu_sun, C_R, A_m_ratio, R_1AU, srp_flux, speed_light, JD_0, a_meeus, inc_ecliptic, include_two_body_dynamics = True, propagationFunction = 'F'):
        """
        Factory method. Use it to get an instance of the class.
        :param mu: Gravitational parameter of the reference planet.
        :param R_E: Reference radius for the model.
        :param J: Array with the J coefficients (J_0 is not used but should be included!)
        :param mu_third: Third body gravitational parameter.
        :param mu_sun: Sun's gravitational parameter.
        :param C_R: SRP Reflectivity coefficient.
        :param A_m_ratio: Area-to-mass ratio [m^2/kg].
        :param R_1AU: 1 AU distance
        :param srp_flux: SRP flux [W/m^2].
        :param speed_light: Speed of light [m/s].
        :param JD_0: Julian date at the initial time.
        :param a_meeus: [2-dimensional numpy array] Meeus coefficients to compute the orbital elements of any planet wrt the sun.
        :param inc_ecliptic: [rad] Inclination of the plane of the ecliptic wrt the Earth Mean Orbital Plane at J2000.
        :param include_two_body_dynamics: [boolean] True to include two-body dynamics.
        :param propagationFunction: Method to use for propagation: 'F' (Use F(X,t)), 'F_plus_STM' (Use F and STM), 'F_vector (propagate many states in parallel).
        :return: An orbitZonalHarmonics object
        """
        params = (mu, R_E, J, mu_third, mu_sun, C_R, A_m_ratio, R_1AU, srp_flux, speed_light, JD_0, a_meeus, inc_ecliptic, include_two_body_dynamics)
        symbState = zonalHarmonicThirdBodySRPModel.buildSymbolicState()
        inputSymb = zonalHarmonicThirdBodySRPModel.buildSymbolicInput()
        zonHarmThirdSRPMod = zonalHarmonicThirdBodySRPModel(symbState, params, propagationFunction, inputSymb)

        return zonHarmThirdSRPMod

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

        # ADDITIONAL STATES
        C_R = sp.symbols('C_R')
        #a1, a2, a3 = sp.symbols('a1 a2 a3')

        X_symb = [x, y, z, x_dot, y_dot, z_dot, C_R]#, a1, a2, a3]
        return X_symb

    ## -------------------------Public Interface--------------------------

    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
        :return: The result of the function in a vector with the same size than X.
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]
        states = 6

        # Change this part for adding more states
        mu = self._params[0]
        R_E = self._params[1]
        J = self._params[2]
        mu_third = self._params[3]
        C_R = X[states] # ------> Estimated
        states += 1
        A_m = self._params[6]
        R_1AU = self._params[7]
        srp_flux = self._params[8]
        c = self._params[9]
        #---------------------------------

        # a1 = X[states+1]
        # a2 = X[states+1]
        # a3 = X[states+2]
        # states += 3

        x_sun_ref = params[0]
        y_sun_ref = params[1]
        z_sun_ref = params[2]
        x_third_ref = params[3]
        y_third_ref = params[4]
        z_third_ref = params[5]

        # r_vec = np.array([x,y,z])
        # r = np.linalg.norm(r_vec)
        # r_sun_vec = np.array([x_sun_ref, y_sun_ref, z_sun_ref])
        # r_sun = np.linalg.norm(r_sun_vec)
        # delta_vec = r_sun_vec-r_vec
        # delta = np.linalg.norm(delta_vec)
        #
        # two_body = -mu*r_vec/r**3
        # third_body = mu_third*(delta_vec/delta**3 - r_sun_vec/r_sun**3)
        # srp = -srp_flux*R_1AU**2/c*A_m*C_R*(delta_vec/delta**3)
        # print two_body
        # print third_body
        # print srp
        nmbrOfStates = self.getNmbrOfStates()
        F = np.zeros(nmbrOfStates)

        if self._usingDMC:
            w_x = X[states]
            w_y = X[states+1]
            w_z = X[states+2]
            B = self._DMCbeta
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref, mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c, [B])
        else:
            for i in range(0, nmbrOfStates):
                F[i] = self._modelLambda[i](x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref, mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c)

        return F

    def computeJacobian(self, X, t, params, u = None):
        """
        Computes the Jacobian of the dynamic function
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :param u: Input vector.
        :return:
        """
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]
        states = 6

        # a1 = X[states+1]
        # a2 = X[states+1]
        # a3 = X[states+2]
        # states += 3

        # Change this part for adding more states
        mu = self._params[0]
        R_E = self._params[1]
        J = self._params[2]
        mu_third = self._params[3]
        C_R = X[states] # ------> Estimated
        states += 1
        A_m = self._params[6]
        R_1AU = self._params[7]
        srp_flux = self._params[8]
        c = self._params[9]
        #---------------------------------

        x_sun_ref = params[0]
        y_sun_ref = params[1]
        z_sun_ref = params[2]
        x_third_ref = params[3]
        y_third_ref = params[4]
        z_third_ref = params[5]

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        if self._usingDMC:
            w_x = X[states]
            w_y = X[states+1]
            w_z = X[states+2]
            B = self._DMCbeta
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z,
                                                         x_sun_ref, y_sun_ref, z_sun_ref,
                                                         x_third_ref, y_third_ref, z_third_ref,
                                                         mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c, [B])
        else:
            for i in range(0,nmbrOfStates):
                for j in range(0,nmbrOfStates):
                    A[i][j] = self._jacobianLambda[i][j](x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref,
                                                         x_third_ref, y_third_ref, z_third_ref,
                                                         mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c)

        return A

    def computeTimeDependentParameters(self, t):
        """
        OVERRIDEN METHOD.
        Computes the third body position relative to the reference planet.
        :param t:
        :return:
        """
        # TODO: THIS METHOD ONLY WORKS FOR THE SUN AS A THIRD BODY. Change it for another celestial body.

        mu_third = self._params[3] # TODO: Assumming mu_third is mu_sun
        R_1AU = self._params[7]
        JD_0 = self._params[10]
        a_meeus = self._params[11]
        inc_ecliptic = self._params[12]

        JD = eph.JDplusSeconds(JD_0, t) # Computes the new Julian Date

        # Computes the position of the reference planet wrt the sun.
        (a, e, i, raan, w, nu) = \
            eph.computeOrbitalElementsMeeus(a_meeus[0], a_meeus[1], a_meeus[2], a_meeus[3], a_meeus[4], a_meeus[5], JD, R_1AU)
        (r_ref_sun, v_ref_sun, ret) = orbEl.orbitalElements2PositionVelocity(mu_third, a, e, i, raan, w, nu)
        # The position is given in the Earth Mean Orbital plane at J2000 reference frame
        # The position has to be rotated to the equatorial plane of the planet around the x axis (vernal equinox)
        r_ref_sun = ROT1(-inc_ecliptic).dot(r_ref_sun) # TODO: CHECK THE SIGN!!!

        x_sun_ref = -r_ref_sun[0]
        y_sun_ref = -r_ref_sun[1]
        z_sun_ref = -r_ref_sun[2]

        # TODO: assumes the third body is the sun. Generalization needed!!!
        x_third_ref = x_sun_ref
        y_third_ref = y_sun_ref
        z_third_ref = z_sun_ref

        return (x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref)

    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t) and stores the models and lambda functions.
        :return:
        """
        mu_param = self._params[0]
        R_E_param = self._params[1]
        J_param = self._params[2]
        mu_third_param = self._params[3]
        mu_sun_param = self._params[4]
        C_R_param = self._params[5]
        A_m_ratio_param = self._params[6]
        R_1AU_param = self._params[7]
        srp_flux_param = self._params[8]
        speed_light_param = self._params[9]
        JD_0_param = self._params[10]
        a_meeus_param = self._params[11]
        inc_ecliptic_param = self._params[12]
        include_two_body_dynamics_param = self._params[13]

        zonHarmMod = zonalHarmonicsModel.getDynamicModel(mu_param, R_E_param, J_param, include_two_body_dynamics_param)
        thirdBodyMod = thirdBodyGravityModel.getDynamicModel(mu_third_param, JD_0_param, a_meeus_param,inc_ecliptic_param)
        srpMod = solarRadiationPressureModel.getDynamicModel(C_R_param, A_m_ratio_param, R_1AU_param, srp_flux_param, speed_light_param, JD_0_param, a_meeus_param, inc_ecliptic_param, mu_sun_param)
        zonHarmSymbMod = zonHarmMod.getSymbolicModel()
        thirdBodySymbMod = thirdBodyMod.getSymbolicModel()
        srpSymbMod = srpMod.getSymbolicModel()

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        # Zonal Harmonics parameters
        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', J_param.size)

        # Third body parameters
        mu_third = sp.symbols('mu_third')
        # Position of the third body relative to the reference from which (x, y, z) is computed
        x_third_ref = sp.symbols('x_third_ref')
        y_third_ref = sp.symbols('y_third_ref')
        z_third_ref = sp.symbols('z_third_ref')

        ## SRP parameters
        C_R, A_m, R_1AU, srp_flux, c = sp.symbols('C_R A_m R_1AU srp_flux c')
        # Position of the sun relative to the reference from which (x, y, z) is computed
        x_sun_ref = sp.symbols('x_sun_ref')
        y_sun_ref = sp.symbols('y_sun_ref')
        z_sun_ref = sp.symbols('z_sun_ref')

        # # bias parameters
        # a1 = sp.symbols('a1')
        # a2 = sp.symbols('a2')
        # a3 = sp.symbols('a3')

        nmbrOfStates = self.getNmbrOfStates()

        self._modelSymb = []
        self._modelSymb.append(x_dot)
        self._modelSymb.append(y_dot)
        self._modelSymb.append(z_dot)
        self._modelSymb.append(zonHarmSymbMod[3] + thirdBodySymbMod[3] + srpSymbMod[3])# + a1)
        self._modelSymb.append(zonHarmSymbMod[4] + thirdBodySymbMod[4] + srpSymbMod[4])# + a2)
        self._modelSymb.append(zonHarmSymbMod[5] + thirdBodySymbMod[5] + srpSymbMod[5])# + a3)

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            for i in range(6, nmbrOfStates-3): # for every other state
                self._modelSymb.append(0)
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            self._modelSymb[3] += w_x
            self._modelSymb[4] += w_y
            self._modelSymb[5] += w_z
            self._modelSymb.append(-B[0]*w_x)
            self._modelSymb.append(-B[1]*w_y)
            self._modelSymb.append(-B[2]*w_z)

            for i in range(0, nmbrOfStates):
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref, mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c, [B]), self._modelSymb[i], "numpy")
        else:
            for i in range(6, nmbrOfStates): # for every other state
                self._modelSymb.append(0)
            for i in range(0, nmbrOfStates):
                #print "Model component ", i, " : ", self._modelSymb[i]
                self._modelLambda[i] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref, mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c), self._modelSymb[i], "numpy")

        return self._modelSymb


    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model with respect to position and velocity
        and stores the models and lambda functions.
        :return:
        """
        degree = self._params[2].size - 1

        x = self._stateSymb[0]
        y = self._stateSymb[1]
        z = self._stateSymb[2]
        x_dot = self._stateSymb[3]
        y_dot = self._stateSymb[4]
        z_dot = self._stateSymb[5]

        # Zonal Harmonics parameters
        mu = sp.symbols('mu')
        R_E = sp.symbols('R_E')
        J = sp.symarray('J', degree + 1)

        # Third body parameters
        mu_third = sp.symbols('mu_third')
        # Position of the third body relative to the reference from which (x, y, z) is computed
        x_third_ref = sp.symbols('x_third_ref')
        y_third_ref = sp.symbols('y_third_ref')
        z_third_ref = sp.symbols('z_third_ref')

        ## SRP parameters
        C_R, A_m, R_1AU, srp_flux, c = sp.symbols('C_R A_m R_1AU srp_flux c')
        # Position of the sun relative to the reference from which (x, y, z) is computed
        x_sun_ref = sp.symbols('x_sun_ref')
        y_sun_ref = sp.symbols('y_sun_ref')
        z_sun_ref = sp.symbols('z_sun_ref')

        # # bias parameters
        # a1 = sp.symbols('a1')
        # a2 = sp.symbols('a2')
        # a3 = sp.symbols('a3')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        if self._usingDMC:
            w_x = self._stateSymb[-3]
            w_y = self._stateSymb[-2]
            w_z = self._stateSymb[-1]
            B = sp.symarray('B', 3)
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z, x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref, mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c, [B]), dF[i][j], "numpy")
        else:
            for i in range(0, nmbrOfStates) :
                F[i] = self._modelSymb[i]
                for j in range(0, nmbrOfStates) :
                    dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                    #print "Model Partial [", i, ",", j, "]: ", dF[i][j]
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, x_sun_ref, y_sun_ref, z_sun_ref, x_third_ref, y_third_ref, z_third_ref, mu, R_E, [J], mu_third, C_R, A_m, R_1AU, srp_flux, c), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb
