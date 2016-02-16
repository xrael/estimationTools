import numpy as np
from abc import ABCMeta, abstractmethod
import sympy as sp

######################################################
# modelBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# functions of a state and time F(X,t).
#
# The interface is:
# Abstract:
# computeModel(): Returns the result of the dynamic function F(X,t)
# computeJacobian(): Returns the Jacobian of F evaluated at X,t.
# getNmbrOfOutputs(): Return the number of outputs of the function.
#
# getModelFunction(): Returns a function that permits computing F(X,t).
# getSymbolicModel(): Returns a sympy object with the symbolic model.
# getSymbolicState(): Returns a list of symbols with the state.
# getName(): Returns the name of the model.
# defineSymbolicState(): Redefines the state.
# computeModelFromVector(): Computes the model using a vector of states and a time vector.
# getNmbrOfStates(): Return the number of states.
# getParameters(): Returns the constant parameters used by F.
######################################################
class modelBase:
    """
    Base class of every model F(X,t)
    """
    __metaclass__ = ABCMeta

    ##-----------------Attributes---------------
    _params = None
    _modelSymb = None
    _modelLambda = None
    _jacobianSymb = None
    _jacobianLambda = None
    _stateSymb = None
    _name = ""
    ##------------------------------------------

    def __init__(self, name, stateSymb, params):
        """
        Constructor.
        :param stateSymb: [List with sympy expressions] Symbolic description of the state (List with symbolic expressions).
        :param params: [tuple] Static parameters used by the model.
        :return:
        """
        self._name = name
        self._stateSymb = stateSymb             # Total state
        self._params = params
        self._computeSymbolicModel()
        self._computeSymbolicJacobian()
        return

    ## ------------------------Public Interface---------------------------

    ## Abstract methods (to be defined by the inherited classes)
    """
    Numerically computes the Model function F(X,t)
    :param X: [1-dimension numpy vector] State.
    :param t: [double] Time.
    :param params: [tuple] Variable parameters.
    """
    @abstractmethod
    def computeModel(self, X, t, params): pass

    """
    Numerically computes the Model jacobian dF/dX(X,t)
    :param X: [1-dimension numpy vector] State.
    :param t: [double] Time.
    :param params: [tuple] Variable parameters.
    """
    @abstractmethod
    def computeJacobian(self, X, t, params): pass

    """
    Returns the dimension of the output F(X,t).
    """
    @abstractmethod
    def getNmbrOutputs(self): pass

    def getModelFunction(self):
        """
        Returns a Lambda function to numerically compute the function model.
        :return: [Lambda func] A function to compute F(X,t).
        """
        F = lambda X, t, params : self.computeModel(X, t, params)
        return F

    def getSymbolicModel(self):
        """
        Returns the symbolic model.
        :return: [list of sympy expressions] A list with the symbolic models.
        """
        return self._modelSymb

    def getSymbolicState(self):
        """
        Returns the symbolic state.
        :return: [list of sympy expressions] A list with the symbolic state.
        """
        return self._stateSymb

    def getName(self):
        return self._name

    def defineSymbolicState(self, stateSymb):
        """
        Redefines the symbolic state assigned in the constructor.
        Gets rid of all sub-models.
        :param stateSymb: [List of Sympy expressions] State vector.
        :return:
        """
        self._stateSymb = stateSymb
        self._computeSymbolicModel()
        self._computeSymbolicJacobian()

        return

    def computeModelFromVector(self, X_vec, t_vec, params):
        """
        Computes the model using a vector of states and a time vector.
        :param X_vec: [2-dimension numpy array] Vector of state vectors (p x dim(X)).
        :param t_vec: [1-dimension numpy array] Time vector (p values).
        :param params: [tuple] Parameters used
        :return: [2-dimension numpy array] A matrix (p x dim(F(X,t))) with the outputs.
        """
        nmbrPoints = t_vec.size

        output = np.zeros((nmbrPoints, self.getNmbrOutputs()))

        for i in range(0, nmbrPoints):
            output[i] = self.computeModel(X_vec[i], t_vec[i], params)

        return output

    def getParameters(self):
        """
        Returns the constant parameters.
        :return: [tuple] Parameters
        """
        return self._params

    def getNmbrOfStates(self):
        """
        Returns the number of states used.
        :return: [int] Number of states.
        """
        return len(self._stateSymb)

    ## ------------------------Private Methods---------------------------
    @abstractmethod
    def _computeSymbolicModel(self): pass

    @abstractmethod
    def _computeSymbolicJacobian(self): pass


######################################################
# dynamicModelBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# function dynamic models X_dot = F(X,t).
######################################################
class dynamicModelBase(modelBase):
    """
    Base class of every dynamic model F(X,t).
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, stateSymb, params):
        super(dynamicModelBase, self).__init__(name, stateSymb, params)
        return

    def getNmbrOutputs(self):
        return self.getNmbrOfStates() # it's the same for a dynamical model!

    def computeModelPlusSTMFunction(self, X, t, params):
        """
        Computes the function [F(X,t), STM] which is useful for propagating X and the STM in
        [X_dot, STM_dot] = [F(X,t), STM]
        :param X: [1-dimension numpy array] State.
        :param t: [double] Time.
        :param params: [tuple] Variable parameters.
        :return: [1-dimension numpy array] The output of [F(X,t), STM] in a vector.
        """
        dX = self.computeModel(X, t, params)
        A = self.computeJacobian(X, t, params)

        nmbrStates = self.getNmbrOfStates()

        for i in range(0, nmbrStates):  # loop for each STM column vector (phi)
            phi = X[(nmbrStates + nmbrStates*i):(nmbrStates + nmbrStates + nmbrStates*i):1]
            dphi = A.dot(phi)
            dX = np.concatenate([dX, dphi])

        return dX

    def getModelPlusSTMFunction(self):
        """
        Returns a Lambda function with the function [F(X,t), STM].
        :return: [lambda func] A function to compute [F(X,t), STM].
        """
        F = lambda X, t, params: self.computeModelPlusSTMFunction(X, t, params)
        return F


######################################################
# orbitalDynamicModelBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# function dynamic models X_dot = F(X,t), where
# the state is, at least, position and velocity.
# The minimum number of states is 6, being these position
# and velocity.
######################################################
class orbitalDynamicModelBase(dynamicModelBase):
    """
    Base class of every orbit dynamic model F(X,t).
    """
    __metaclass__ = ABCMeta

    _process_noise_frame = ""
    _usingSNC = False
    _usingDMC = False
    _DMCbeta = None

    def __init__(self, name, stateSymb, params):
        super(orbitalDynamicModelBase, self).__init__(name, stateSymb, params)
        self._process_noise_frame = "INERTIAL"
        self._usingSNC = False
        self._usingDMC = False
        self._DMCbeta = None
        return

    def useStateNoiseCompensation(self, useSNC, process_noise_frame = "INERTIAL"):
        """
        Enable SNC.
        :param useSNC: [boolean]
        :param process_noise_frame: [string] Frame in which the SNC covariance matrix is expressed.
            Options: "INERTIAL" (default), "RIC"
        :return:
        """
        self._process_noise_frame = process_noise_frame
        self._usingSNC = useSNC
        if useSNC == True:
            if self._usingDMC == True:
                self._stateSymb.pop()
                self._stateSymb.pop()
                self._stateSymb.pop()
                self._DMCbeta = None
                self.defineSymbolicState(self._stateSymb)
                self._usingDMC = False
        return

    def useDynamicModelCompensation(self, useDMC, beta):
        """
        Enable DMC.
        :param useDMC: [boolean]
        :param beta: [2-dimension numpy array] Diagonal matrix with the inverse of the time constants for DMC.
        :return:
        """
        if useDMC == True:
            self._usingSNC = False
            if self._usingDMC == False:
                self._usingDMC = True
                self._DMCbeta = beta
                w_x, w_y, w_z = sp.symbols('w_x w_y w_z')
                self.defineSymbolicState(self._stateSymb + [w_x, w_y, w_z])
        else: # useDMC == False
            if self._usingDMC == True:
                self._usingDMC = False
                self._stateSymb.pop()
                self._stateSymb.pop()
                self._stateSymb.pop()
                self._DMCbeta = None
                self.defineSymbolicState(self._stateSymb)

        return

    def usingSNC(self):
        return self._usingSNC

    def usingDMC(self):
        return self._usingDMC

    def getSncCovarianceMatrix(self, t_i_1, t_i, state_i, Q_i_1):
        """
        Gets the Covariance matrix part associated to the SNC.
        Assumes that the noise is added to acceleration.
        Uses the constant-velocity approximation.
        The PNSTM is nx3, where n is the number of states.
        :param t_i_1: [double] Time i-1
        :param t_i: [double] Time i
        :param state_i: [1-dimension numpy array] Estimation of the state at time t_i.
        :param Q_i_1: [2-dimension numpy array] Noise covariance at time t_i_1.
        :return: [2-dimension numpy array] The value PNSTM^T*Q*PNSTM, where Q is rotated to Inertial frame.
        """
        delta_t = t_i - t_i_1
        # Process Noise Transition Matrix with constant velocity approximation
        pntm_i = np.zeros((self.getNmbrOfStates(), 3))
        pntm_i[0:3, :] = delta_t**2/2 * np.eye(3)
        pntm_i[3:6, :] = delta_t * np.eye(3)

        Q_rot = self.transformCovariance(state_i, Q_i_1)

        return pntm_i.dot(Q_rot).dot(pntm_i.T)

    def getSmcCovarianceMatrix(self, t_i_1, t_i, Q_i_1):
        nmbrStates = self.getNmbrOfStates()
        delta_t = t_i - t_i_1
        beta = self._DMCbeta
        Q_DMC = np.zeros((nmbrStates, nmbrStates))

        for i in range(0,3):
            sigma_i_sq = Q_i_1[i,i]
            beta_i = beta[i]

            #Qrr
            Q_DMC[i,i] = sigma_i_sq * (1.0/(3*beta_i**2) * delta_t**3 - 1.0/beta_i**3 * delta_t**2 + 1.0/beta_i**4 * delta_t -2.0/beta_i**4 * np.exp(-beta_i*delta_t) * delta_t + 1.0/(2*beta_i**5) * (1 - np.exp(-2*beta_i*delta_t)))
            #Qvv
            Q_DMC[3+i,3+i] = sigma_i_sq * (1.0/beta_i**2 * delta_t - 2.0/beta_i**3 * (1 - np.exp(-beta_i*delta_t)) + 1.0/(2*beta_i**3) * (1 - np.exp(-2*beta_i*delta_t)))
            #Qww
            Q_DMC[-3 + i,-3 + i] = sigma_i_sq * (1.0/(2*beta_i)) * (1 - np.exp(-2*beta_i*delta_t))
            #Qvr = Qrv
            Q_DMC[i,3+i] = sigma_i_sq * (1.0/(2*beta_i**2) * delta_t**2 - 1.0/beta_i**3 * delta_t + 1.0/beta_i**3 * np.exp(-beta_i*delta_t) * delta_t + 1.0/beta_i**4 * (1 - np.exp(-beta_i*delta_t)) - 1.0/(2*beta_i**4) * (1 - np.exp(-2*beta_i*delta_t)))
            Q_DMC[3+i,i] = Q_DMC[i,3+i]
            #Qwr = Qrw
            Q_DMC[i,-3+i] = sigma_i_sq * (1.0/(2*beta_i**3) * (1 - np.exp(-2*beta_i*delta_t)) - 1.0/beta_i**2 * np.exp(-beta_i*delta_t) * delta_t)
            Q_DMC[-3+i,i] = Q_DMC[i,-3+i]
            #Qwv = Qvw
            Q_DMC[3+i,-3+i] = sigma_i_sq * (1.0/(2*beta_i**2) * (1 + np.exp(-2*beta_i*delta_t)) - 1.0/(beta_i**2) * np.exp(-beta_i*delta_t))
            Q_DMC[-3+i,3+i] = Q_DMC[3+i,-3+i]

        return Q_DMC

    def transformCovariance(self, state, Q):
        """
        Transforms a covariance given in a given frame to inertial frame.
        :param state: [1-dimension numpy array] current state (at least position and velocity).
        :param Q: [2-dimension numpy array] Covariance matrix.
        :return: The covariance matrix transformed.
        """
        position = state[0:3]
        velocity = state[3:6]

        if self._process_noise_frame == "RIC":  # Radial - In-track (actually, along track) - Cross-track
            DCM = np.zeros((3,3)) # DCM from inertial to RIC
            r = position/np.linalg.norm(position)
            w = np.cross(position, velocity)
            w = w/np.linalg.norm(w)
            s = np.cross(w, r)

            DCM[0,:] = r    # radial
            DCM[1,:] = s    # along-track
            DCM[2,:] = w    # cross-track

            return (DCM.T).dot(Q).dot(DCM)
        else: # INERTIAL frame is default
            return Q


######################################################
# observerBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# observation model Y = G(X,t).
######################################################
class observerModelBase(modelBase):
    """
    Base class of every observation model G(X,t).
    Every observation model has, at least, one set of coordinates against which
    the observations is measured.
    """
    __metaclass__ = ABCMeta

    ##-----------------Attributes---------------
    _observerCoordinates = None
    ##------------------------------------------

    def __init__(self, name, stateSymb, params, observerCoordinates):
        super(observerModelBase, self).__init__(name, stateSymb, params)

        self._nmbrCoordinates = observerCoordinates.shape[0]
        self._observerCoordinates = observerCoordinates
        return

    def getObserverCoordinates(self):
        """
        Returns the coordinates of the observers associated.
        Each observation instance has a set of coordinates for the observers.
        Each observer makes the same observation from different coordinates.
        :return: [2-dimension numpy array] a matrix with each set of coordinates in rows.
        """
        return self._observerCoordinates

    def getNmbrObservers(self):
        """
        Returns the number of observers (number of sets of coordinates).
        """
        return self._nmbrCoordinates

    """
    Method that computes if the state can be observed for a given time and given the coordinates of the observer.
    """
    @abstractmethod
    def isObservable(self, X, t, params): pass


# class dmcModel(dynamicModelBase):
#
#     ## Constructor: DO NOT USE IT!
#     def __init__(self, name, stateSymb, params):
#         super(dmcModel, self).__init__(name, stateSymb, params)
#
#         return
#
#     @classmethod
#     def getDynamicModel(cls, B):
#         """
#         Factory method. Use it to get an instance of the class.
#         :param B:
#         :return: An dmcModel object
#         """
#         params = (B,)
#         name = "DMC" # DO NOT CHANGE IT!
#         symbState = dmcModel.buildSymbolicState()
#         dmc = dmcModel(name, symbState, params)
#
#         return dmc
#
#     @classmethod
#     def buildSymbolicState(cls):
#         """
#         State vector for DMC.
#         :return: A list with the symbolic symbols of the state.
#         """
#         x, y, z = sp.symbols('x, y, z')
#         x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
#         w_x, w_y, w_z = sp.symbols('w_x, w_y, w_z')
#         X_symb = [x, y, z, x_dot, y_dot, z_dot, w_x, w_y, w_z]
#         return X_symb
#
#     def computeModel(self, X, t, params):
#         """
#         Computes the dynamic function F(X,t)
#         :param X: State.
#         :param t: Time.
#         :param params: parameters used by the function.
#         :return: The result of the function in a vector with the same size than X.
#         """
#         v_x = X[3]
#         v_y = X[4]
#         v_z = X[5]
#         w_x = X[6]
#         w_y = X[7]
#         w_z = X[8]
#         B = self._params[0]
#
#         nmbrOfStates = self.getNmbrOfStates()
#
#         F = np.zeros(nmbrOfStates)
#
#         F[3] = v_x
#         F[4] = v_y
#         F[5] = v_z
#         F[6] = -B[0]*w_x
#         F[7] = -B[1]*w_y
#         F[8] = -B[2]*w_z
#
#         # for i in range(0, nmbrOfStates):
#         #     F[i] = self._modelLambda[i](w_x, w_y, w_z, [B])
#
#         return F
#
#     # Returns A matrix
#     # This is application-specific. It depends on how state vector is defined.
#     def computeJacobian(self, X, t, params):
#         """
#         Computes the Jacobian of the dynamic function
#         :param X: State
#         :param t: time
#         :param params: parameters used by the model
#         :return:
#         """
#         w_x = X[6]
#         w_y = X[7]
#         w_z = X[8]
#         B = self._params[0]
#
#         nmbrOfStates = self.getNmbrOfStates()
#         A = np.zeros([nmbrOfStates,nmbrOfStates])
#
#         A
#
#         for i in range(0,nmbrOfStates):
#             for j in range(0,nmbrOfStates):
#                 A[i][j] = self._jacobianLambda[i][j](w_x, w_y, w_z, [B])
#
#         return np.diag(B)
#
#     ## -------------------------Private Methods--------------------------
#     def _computeSymbolicModel(self):
#         """
#
#         :return:
#         """
#         x = self._stateSymb[0]
#         y = self._stateSymb[1]
#         z = self._stateSymb[2]
#         x_dot = self._stateSymb[3]
#         y_dot = self._stateSymb[4]
#         z_dot = self._stateSymb[5]
#         w_x = self._stateSymb[6]
#         w_y = self._stateSymb[7]
#         w_z = self._stateSymb[8]
#
#         B = sp.symarray('B', 3)
#
#         self._modelSymb = [0, 0, 0, w_x, w_y, w_z, -B[0]*w_x,  -B[1]*w_y,  -B[2]*w_z]
#
#         nmbrOfStates = self.getNmbrOfStates()
#
#         self._modelLambda = [0 for i in range(0, nmbrOfStates)]
#         for i in range(0, nmbrOfStates):
#             self._modelLambda[i] = sp.lambdify((w_x, w_y, w_z, [B]), self._modelSymb[i], "numpy")
#
#         return self._modelSymb
#
#     def _computeSymbolicJacobian(self):
#         """
#
#         :return:
#         """
#         x = self._stateSymb[0]
#         y = self._stateSymb[1]
#         z = self._stateSymb[2]
#         x_dot = self._stateSymb[3]
#         y_dot = self._stateSymb[4]
#         z_dot = self._stateSymb[5]
#         w_x = self._stateSymb[6]
#         w_y = self._stateSymb[7]
#         w_z = self._stateSymb[8]
#
#         B = sp.symarray('B', 3)
#
#         nmbrOfStates = self.getNmbrOfStates()
#
#         F = [0 for i in range(0, nmbrOfStates)]
#         dF = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
#         A_lambda = [[0 for i in range(0, nmbrOfStates)] for i in range(0, nmbrOfStates)]
#
#         for i in range(0, nmbrOfStates) :
#             F[i] = self._modelSymb[i]
#             for j in range(0, nmbrOfStates) :
#                 dF[i][j] = sp.diff(F[i], self._stateSymb[j])
#                 A_lambda[i][j] = sp.lambdify((w_x, w_y, w_z, [B]), dF[i][j], "numpy")
#
#         self._jacobianSymb = dF
#         self._jacobianLambda = A_lambda
#
#         return self._jacobianSymb