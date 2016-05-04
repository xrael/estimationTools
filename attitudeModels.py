#######################################################################################################################
# Attitude models.
#
# Manuel F. Diaz Ramos
#
# The classes here must inherit attitudeModelBase(modelBase).
#
#######################################################################################################################

import sympy as sp
import numpy as np
import attitudeKinematics
from modelBase import attitudeModelBase

#######################################################################################################################
# MRPs.
#
# Modified Rodrigues Parameter kinematic model.
#######################################################################################################################

class mrpKinematicsModel(attitudeModelBase):


    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(mrpKinematicsModel, self).__init__("MRP_attitude", stateSymb, params, propagationFunction, inputSymb)

        return

    ## -------------------------Class Methods--------------------------
    ## Factory method
    @classmethod
    def getDynamicModel(cls, getAngVelObs, propagationFunction = 'F'):
        """
        Factory method. Use it to get an instance of the class.
        :param getAngVelObs: Pointer to the function to get the observations.
        :param propagationFunction: Method to use for propagation: 'F' (Use F(X,t)), 'F_plus_STM' (Use F and STM), 'F_vector (propagate many states in parallel).
        :return: An mrpKinematicModel object.
        """
        params = (getAngVelObs,)
        symbState = mrpKinematicsModel.buildSymbolicState()
        inputSymb = mrpKinematicsModel.buildSymbolicInput()
        mrpKinematicsMod = mrpKinematicsModel(symbState, params, propagationFunction, inputSymb)

        return mrpKinematicsMod

    ## THIS IS THE METHOD TO BE MODIFIED IF A CHANGE IN THE STATE IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicState(cls):
        """
        Modify this method to build a new symbolic state vector.
        :return: A list with the symbolic symbols of the state.
        """

        mrp1, mrp2, mrp3 = sp.symbols('mrp1 mrp2 mrp3')
        beta1, beta2, beta3 = sp.symbols('beta1 beta2 beta3')
        X_symb = [mrp1, mrp2, mrp3, beta1, beta2, beta3]
        return X_symb

    ## THIS IS THE METHOD TO OVERRIDE IF A CONSIDER PARAMETER ANALYSIS IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicInput(cls):
        """
        Modify this method to build a new symbolic input vector (for control force or consider covariance analysis).
        :return: A list with the symbolic symbols of the input.
        """
        # DEFAULT
        w1_obs, w2_obs, w3_obs = sp.symbols('w1_obs w2_obs w3_obs')
        U_symb = [w1_obs, w2_obs, w3_obs]
        return U_symb

    ## -------------------------Public Interface--------------------------

    ### I THINK THIS CODE SHOULD BE SOMEWHERE ELSE-----------------
    def normalizeOutput(self, out):
        attitudeKinematics.switchMRPrepresentation(out)
        return

    def normalizeCovariance(self, X, P):
        attitudeKinematics.switchMRPcovariance(X, P)
        return
    ##------------------------------------------------------------

    """
    Think of this function as a callback that computes the input signal u(t).
    OVERRIDE IF THE MODEL HAS AN INPUT VECTOR.
    :param t: [double] Current time.
    :return: [1-dimensional numpy array] Input vector at time t.
    """
    def getInput(self, t):
        getAngVelObs = self._params[0]
        return getAngVelObs(t)

    def computeModel(self, X, t, params, u = None):
        """
        Computes the dynamic function F(X,t)
        :param X: State vector.
        :param t: Time.
        :param params: parameters used by the function.
        :param u: Input vector.
        :return: The result of the function in a vector with the same size than X.
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        beta1 = X[3]
        beta2 = X[4]
        beta3 = X[5]

        # # CHANGE THIS PART FOR ADDING MORE STATES!!!
        # getAngVelObs = self._params[0]
        # #-------------------------------------------

        #w_obs = getAngVelObs(t)
        w1_obs = u[0]
        w2_obs = u[1]
        w3_obs = u[2]

        nmbrOfStates = self.getNmbrOfStates()
        F = np.zeros(nmbrOfStates)

        for i in range(0, nmbrOfStates):
            F[i] = self._modelLambda[i](mrp1, mrp2, mrp3, beta1, beta2, beta3, w1_obs, w2_obs, w3_obs)

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
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        beta1 = X[3]
        beta2 = X[4]
        beta3 = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        getAngVelObs = self._params[0]
        #-------------------------------------------

        #w_obs = getAngVelObs(t)
        w1_obs = u[0]
        w2_obs = u[1]
        w3_obs = u[2]

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        for i in range(0,nmbrOfStates):
            for j in range(0,nmbrOfStates):
                A[i][j] = self._jacobianLambda[i][j](mrp1, mrp2, mrp3, beta1, beta2, beta3, w1_obs, w2_obs, w3_obs)

        return A

    """
    Numerically computes the noise jacobian dF/dw(X,u,w,t)
    :param X: [1-dimension numpy vector] State.
    :param t: [double] Time.
    :param params: [tuple] Variable parameters.
    :param u: [1-dimension numpy vector] Input variable (control force or consider parameters).
    """
    def computeNoiseJacobian(self, X, t, params, u):
        """
        Computes the Jacobian dFW/dw of the dynamic function F(X,u,w,t).
        :param X: State
        :param t: time
        :param params: parameters used by the model
        :param u: Input vector
        :return:
        """
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        beta1 = X[3]
        beta2 = X[4]
        beta3 = X[5]

        # # CHANGE THIS PART FOR ADDING MORE STATES!!!
        # getAngVelObs = self._params[0]
        # #-------------------------------------------

        #w_obs = getAngVelObs(t)
        w1_obs = u[0]
        w2_obs = u[1]
        w3_obs = u[2]

        nmbrOfStates = self.getNmbrOfStates()
        G = np.zeros([nmbrOfStates,nmbrOfStates])

        for i in range(0,nmbrOfStates):
            for j in range(0,nmbrOfStates):
                G[i][j] = self._jacobianNoiseLambda[i][j](mrp1, mrp2, mrp3, beta1, beta2, beta3, w1_obs, w2_obs, w3_obs)

        return G

    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t)
        :return:
        """

        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]
        beta1 = self._stateSymb[3]
        beta2 = self._stateSymb[4]
        beta3 = self._stateSymb[5]

        w1_obs = self._inputSymb[0]
        w2_obs = self._inputSymb[1]
        w3_obs = self._inputSymb[2]

        mrp_sq = mrp1**2 + mrp2**2 + mrp3**2

        B11 = (1 - mrp_sq + 2*mrp1**2)
        B12 = 2*(mrp1*mrp2 - mrp3)
        B13 = 2*(mrp1*mrp3 + mrp2)

        B21 = 2*(mrp2*mrp1 + mrp3)
        B22 = (1 - mrp_sq + 2*mrp2**2)
        B23 = 2*(mrp2*mrp3 - mrp1)

        B31 = 2*(mrp3*mrp1 - mrp2)
        B32 = 2*(mrp3*mrp2 + mrp1)
        B33 = (1 - mrp_sq + 2*mrp3**2)

        mrp1_dot = sp.Rational(1,4)*(B11*(w1_obs - beta1) + B12*(w2_obs - beta2) + B13*(w3_obs - beta3))
        mrp2_dot = sp.Rational(1,4)*(B21*(w1_obs - beta1) + B22*(w2_obs - beta2) + B23*(w3_obs - beta3))
        mrp3_dot = sp.Rational(1,4)*(B31*(w1_obs - beta1) + B32*(w2_obs - beta2) + B33*(w3_obs - beta3))

        self._modelSymb = []
        self._modelSymb.append(mrp1_dot)
        self._modelSymb.append(mrp2_dot)
        self._modelSymb.append(mrp3_dot)
        self._modelSymb.append(0)
        self._modelSymb.append(0)
        self._modelSymb.append(0)

        nmbrOfStates = self.getNmbrOfStates()

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        for i in range(6, nmbrOfStates): # for every other state
            self._modelSymb.append(0)
        for i in range(0, nmbrOfStates):
            self._modelLambda[i] = sp.lambdify((mrp1, mrp2, mrp3, beta1, beta2, beta3, w1_obs, w2_obs, w3_obs), self._modelSymb[i], "numpy")

        return self._modelSymb

    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model.
        :return:
        """
        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]
        beta1 = self._stateSymb[3]
        beta2 = self._stateSymb[4]
        beta3 = self._stateSymb[5]

        w1_obs = self._inputSymb[0]
        w2_obs = self._inputSymb[1]
        w3_obs = self._inputSymb[2]

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        for i in range(0, nmbrOfStates) :
            F[i] = self._modelSymb[i]
            for j in range(0, nmbrOfStates) :
                dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                A_lambda[i][j] = sp.lambdify((mrp1, mrp2, mrp3, beta1, beta2, beta3, w1_obs, w2_obs, w3_obs), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb

    def computeSymbolicNoiseJacobian(self):
        """

        :return:
        """
        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]
        beta1 = self._stateSymb[3]
        beta2 = self._stateSymb[4]
        beta3 = self._stateSymb[5]

        w1_obs = self._inputSymb[0]
        w2_obs = self._inputSymb[1]
        w3_obs = self._inputSymb[2]

        mrp_sq = mrp1**2 + mrp2**2 + mrp3**2

        B11 = (1 - mrp_sq + 2*mrp1**2)
        B12 = 2*(mrp1*mrp2 - mrp3)
        B13 = 2*(mrp1*mrp3 + mrp2)

        B21 = 2*(mrp2*mrp1 + mrp3)
        B22 = (1 - mrp_sq + 2*mrp2**2)
        B23 = 2*(mrp2*mrp3 - mrp1)

        B31 = 2*(mrp3*mrp1 - mrp2)
        B32 = 2*(mrp3*mrp2 + mrp1)
        B33 = (1 - mrp_sq + 2*mrp3**2)

        nmbrOfStates = self.getNmbrOfStates()

        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        G_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        dF[0][0] = -sp.Rational(1,4) * B11
        dF[0][1] = -sp.Rational(1,4) * B12
        dF[0][2] = -sp.Rational(1,4) * B13
        dF[1][0] = -sp.Rational(1,4) * B21
        dF[1][1] = -sp.Rational(1,4) * B22
        dF[1][2] = -sp.Rational(1,4) * B23
        dF[2][0] = -sp.Rational(1,4) * B31
        dF[2][1] = -sp.Rational(1,4) * B32
        dF[2][2] = -sp.Rational(1,4) * B33
        dF[3][3] = 1
        dF[4][4] = 1
        dF[5][5] = 1

        for i in range(0, nmbrOfStates) :
            for j in range(0, nmbrOfStates) :
                G_lambda[i][j] = sp.lambdify((mrp1, mrp2, mrp3, beta1, beta2, beta3, w1_obs, w2_obs, w3_obs), dF[i][j], "numpy")

        self._jacobianNoiseSymb = dF
        self._jacobianNoiseLambda = G_lambda

        return self._jacobianNoiseSymb



class rigidBodyAttittudeDynamicsModelMRPs(attitudeModelBase):

    ## Constructor: DO NOT USE IT!
    def __init__(self, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        super(rigidBodyAttittudeDynamicsModelMRPs, self).__init__("attitude_dynamics", stateSymb, params, propagationFunction, inputSymb)

        return

    ## -------------------------Class Methods--------------------------
    ## Factory method
    @classmethod
    def getDynamicModel(cls, I11, I22, I33, propagationFunction = 'F'):
        """
        Factory method. Use it to get an instance of the class.
        :param I11:
        :param I22:
        :param I33:
        :param propagationFunction: Method to use for propagation: 'F' (Use F(X,t)), 'F_plus_STM' (Use F and STM), 'F_vector (propagate many states in parallel).
        :return: A rigidBodyAttittudeDynamicsModelMRPs object.
        """
        params = (I11, I22, I33)
        symbState = rigidBodyAttittudeDynamicsModelMRPs.buildSymbolicState()
        inputSymb = rigidBodyAttittudeDynamicsModelMRPs.buildSymbolicInput()
        rigidBodyModel = rigidBodyAttittudeDynamicsModelMRPs(symbState, params, propagationFunction, inputSymb)

        return rigidBodyModel

    ## THIS IS THE METHOD TO BE MODIFIED IF A CHANGE IN THE STATE IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicState(cls):
        """
        Modify this method to build a new symbolic state vector.
        :return: A list with the symbolic symbols of the state.
        """
        mrp1, mrp2, mrp3 = sp.symbols('mrp1 mrp2 mrp3')
        w1, w2, w3 = sp.symbols('w1 w2 w3')
        X_symb = [mrp1, mrp2, mrp3, w1, w2, w3]
        return X_symb

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
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        w1 = X[3]
        w2 = X[4]
        w3 = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        I11 = self._params[0]
        I22 = self._params[1]
        I33 = self._params[2]

        L1 = 0.0
        L2 = 0.0
        L3 = 0.0
        #-------------------------------------------

        nmbrOfStates = self.getNmbrOfStates()
        F = np.zeros(nmbrOfStates)

        for i in range(0, nmbrOfStates):
            F[i] = self._modelLambda[i](mrp1, mrp2, mrp3, w1, w2, w3, I11, I22, I33, L1, L2, L3)

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
        mrp1 = X[0]
        mrp2 = X[1]
        mrp3 = X[2]
        w1 = X[3]
        w2 = X[4]
        w3 = X[5]

        # CHANGE THIS PART FOR ADDING MORE STATES!!!
        I11 = self._params[0]
        I22 = self._params[1]
        I33 = self._params[2]

        L1 = 0.0
        L2 = 0.0
        L3 = 0.0

        nmbrOfStates = self.getNmbrOfStates()
        A = np.zeros([nmbrOfStates,nmbrOfStates])

        for i in range(0,nmbrOfStates):
            for j in range(0,nmbrOfStates):
                A[i][j] = self._jacobianLambda[i][j](mrp1, mrp2, mrp3, w1, w2, w3, I11, I22, I33, L1, L2, L3)

        return A


    ## -------------------------Private Methods--------------------------
    def computeSymbolicModel(self):
        """
        Symbollically computes F(X,t)
        :return:
        """

        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]
        w1 = self._stateSymb[3]
        w2 = self._stateSymb[4]
        w3 = self._stateSymb[5]

        mrp_sq = mrp1**2 + mrp2**2 + mrp3**2

        B11 = (1 - mrp_sq + 2*mrp1**2)
        B12 = 2*(mrp1*mrp2 - mrp3)
        B13 = 2*(mrp1*mrp3 + mrp2)

        B21 = 2*(mrp2*mrp1 + mrp3)
        B22 = (1 - mrp_sq + 2*mrp2**2)
        B23 = 2*(mrp2*mrp3 - mrp1)

        B31 = 2*(mrp3*mrp1 - mrp2)
        B32 = 2*(mrp3*mrp2 + mrp1)
        B33 = (1 - mrp_sq + 2*mrp3**2)

        mrp1_dot = sp.Rational(1,4)*(B11*w1 + B12*w2 + B13*w3)
        mrp2_dot = sp.Rational(1,4)*(B21*w1 + B22*w2 + B23*w3)
        mrp3_dot = sp.Rational(1,4)*(B31*w1 + B32*w2 + B33*w3)

        I11, I22, I33 = sp.symbols('I11 I22 I33')
        L1, L2, L3 = sp.symbols('L1 L2 L3')

        w1_dot = -(I33 - I22)/I11 * w2*w3 + L1/I11
        w2_dot = -(I11 - I33)/I22 * w3*w1 + L2/I22
        w3_dot = -(I22 - I11)/I33 * w1*w2 + L3/I33

        self._modelSymb = []
        self._modelSymb.append(mrp1_dot)
        self._modelSymb.append(mrp2_dot)
        self._modelSymb.append(mrp3_dot)
        self._modelSymb.append(w1_dot)
        self._modelSymb.append(w2_dot)
        self._modelSymb.append(w3_dot)

        nmbrOfStates = self.getNmbrOfStates()

        self._modelLambda = [0 for i in range(0, nmbrOfStates)]

        for i in range(6, nmbrOfStates): # for every other state
            self._modelSymb.append(0)
        for i in range(0, nmbrOfStates):
            self._modelLambda[i] = sp.lambdify((mrp1, mrp2, mrp3, w1, w2, w3, I11, I22, I33, L1, L2, L3), self._modelSymb[i], "numpy")

        return self._modelSymb

    def computeSymbolicJacobian(self):
        """
        Symbollically computes the Jacobian matrix of the model.
        :return:
        """
        mrp1 = self._stateSymb[0]
        mrp2 = self._stateSymb[1]
        mrp3 = self._stateSymb[2]
        w1 = self._stateSymb[3]
        w2 = self._stateSymb[4]
        w3 = self._stateSymb[5]

        I11, I22, I33 = sp.symbols('I11 I22 I33')
        L1, L2, L3 = sp.symbols('L1 L2 L3')

        nmbrOfStates = self.getNmbrOfStates()

        F = [0 for i in range(0, nmbrOfStates)]
        dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
        A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]

        for i in range(0, nmbrOfStates) :
            F[i] = self._modelSymb[i]
            for j in range(0, nmbrOfStates) :
                dF[i][j] = sp.diff(F[i], self._stateSymb[j])
                A_lambda[i][j] = sp.lambdify((mrp1, mrp2, mrp3, w1, w2, w3, I11, I22, I33, L1, L2, L3), dF[i][j], "numpy")

        self._jacobianSymb = dF
        self._jacobianLambda = A_lambda

        return self._jacobianSymb