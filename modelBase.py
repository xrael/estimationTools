import numpy as np
from abc import ABCMeta, abstractmethod
import sympy as sp

######################################################
# modelBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# functions of a state and time F(X,t)
# or the state, time, and an external signal u F(x,u,t).
# This is useful for consider analysis and control force modeling
#
# The interface is:
# Abstract:
# computeModel(): Returns the result of the dynamic function F(X,t)
# computeJacobian(): Returns the Jacobian of F evaluated at X,t.
# getNmbrOfOutputs(): Return the number of outputs of the function.
# computeTimeDependentParameters(): Computes the time dependent parameters that can be an input to computeModel() and computeJacobian().
#
# getModelFunction(): Returns a function that permits computing F(X,t).
# getSymbolicModel(): Returns a sympy object with the symbolic model.
# getSymbolicState(): Returns a list of symbols with the state.
# getName(): Returns the name of the model.
# defineSymbolicState(): Redefines the state.
# computeModelFromVector(): Computes the model using a vector of states and a time vector.
# computeModelFromManyStates(): Computes the the model using many states stacked in a single vector for the same time.
# getModelFromManyStatesFunction(): Returns a lambda function pointing to computeModelFromManyStates().
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
    _jacobianInputSymb = None
    _jacobianInputLambda = None
    _stateSymb = None
    _inputSymb = None
    _name = ""
    _computeModelFunc = None
    _computeModelFromManyStatesFunc = None
    _computeModelPlusSTMFunc = None
    _propFunction = None
    ##------------------------------------------

    def __init__(self, name, stateSymb, params, propagationFunction = 'F', inputSymb = None):
        """
        Constructor.
        :param stateSymb: [List with sympy expressions] Symbolic description of the state (List with symbolic expressions).
        :param params: [tuple] Static parameters used by the model.
        :param propagationFunction: Chooses the function to use to propagate. Options:
        'F': the function F(X,t).
        'F_vector': propagates many vectors at the same time.
        'F_plus_STM': propagates F(X,t) and the STM.
        :param inputSymb: [List with sympy expressions] Symbolic description of the input. "None" if the system has no inputs.
        :return:
        """
        self._name = name
        self._stateSymb = stateSymb             # Total state
        if inputSymb == None:
            self._inputSymb = []
        else:
            self._inputSymb = inputSymb
        self._params = params
        self.computeSymbolicModel()
        self.computeSymbolicJacobian()
        self.computeSymbolicInputJacobian()

        self._computeModelFunc = lambda X, t, params : self.computeModel(X, t, params)
        self._computeModelFromManyStatesFunc = lambda X, t, params: self.computeModelFromManyStates(X, t, params)
        self._computeModelPlusSTMFunc =  lambda X, t, params: self.computeModelPlusSTMFunction(X, t, params)

        # Function used for propagation (used by dynamicSimulator)
        if propagationFunction == 'F':
            self._propFunction = lambda X, t, params : self._computeModelFunc(X, t, self.computeTimeDependentParameters(t) + params)
        elif propagationFunction == 'F_vector':
            self._propFunction = lambda X, t, params : self._computeModelFromManyStatesFunc(X, t, self.computeTimeDependentParameters(t) + params)
        elif propagationFunction == 'F_plus_STM':
            self._propFunction = lambda X, t, params : self._computeModelPlusSTMFunc(X, t, self.computeTimeDependentParameters(t) + params)
        else: # Default
            self._propFunction = lambda X, t, params : self._computeModelFunc(X, t, self.computeTimeDependentParameters(t) + params)

        return

    ## THIS IS THE METHOD TO OVERRIDE IF A CONSIDER PARAMETER ANALYSIS IS TO BE ACCOMPLISHED!!!
    @classmethod
    def buildSymbolicInput(cls):
        """
        Modify this method to build a new symbolic input vector (for control force or consider covariance analysis).
        :return: A list with the symbolic symbols of the input.
        """
        # DEFAULT
        U_symb = []
        return U_symb

    ## ------------------------Public Interface---------------------------

    ## Abstract methods (to be defined by the inherited classes)
    """
    Numerically computes the Model function F(X,t)
    :param X: [1-dimension numpy vector] State.
    :param t: [double] Time.
    :param params: [tuple] Variable parameters.
    :param u: [optional] Input variable (control force or consider parameters).
    """
    @abstractmethod
    def computeModel(self, X, t, params, u = None): pass

    """
    Numerically computes the Model jacobian dF/dX(X,t)
    :param X: [1-dimension numpy vector] State.
    :param t: [double] Time.
    :param params: [tuple] Variable parameters.
    :param u: [optional] Input variable (control force or consider parameters).
    """
    @abstractmethod
    def computeJacobian(self, X, t, params, u = None): pass

    """
    Numerically computes the input jacobian dF/du(X,u,t)
    :param X: [1-dimension numpy vector] State.
    :param t: [double] Time.
    :param params: [tuple] Variable parameters.
    :param u: [1-dimension numpy vector] Input variable (control force or consider parameters).
    """
    def computeInputJacobian(self, X, t, params, u):
        return None

    """
    Returns the dimension of the output F(X,t).
    """
    @abstractmethod
    def getNmbrOutputs(self): pass

    """
    Computes extra time-dependent parameters that the model needs.
    It's useful to compute parameters that depend on time only once and pass them to computeModel() and computeJacobian()
    instead of computing the parameters twice.
    OVERRIDE IF THE MODEL HAS TIME-DEPENDENT PARAMETERS.
    :param t: [double] Current time.
    :return: [tuple] A tuple with the parameters. This tuple can be inputed into computeModel() and computeJacobian().
    """
    def computeTimeDependentParameters(self, t):
        return () # OVERRIDE IF THE MODEL HAS TIME-DEPENDENT PARAMETERS.


    """
    Think of this function as a callback that computes the input signal u(t).
    OVERRIDE IF THE MODEL HAS AN INPUT VECTOR.
    :param t: [double] Current time.
    :return: [1-dimensional numpy array] Input vector at time t.
    """
    def getInput(self, t):
        return None

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

    def getSymbolicInput(self):
        """
        Returns the symbolic input.
        :return: [list of sympy expressions] A list with the symbolic input.
        """
        return self._inputSymb

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
        self.computeSymbolicModel()
        self.computeSymbolicJacobian()
        self.computeSymbolicInputJacobian()

        return

    def defineSymbolicInput(self, inputSymb):
        """
        Redefines the symbolic input assigned in the constructor.
        Gets rid of all sub-models.
        :param inputSymb: [List of Sympy expressions] Input vector.
        :return:
        """
        if inputSymb == None:
            self._inputSymb = []
        else:
            self._inputSymb = inputSymb
        self.computeSymbolicModel()
        self.computeSymbolicJacobian()
        self.computeSymbolicInputJacobian()

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
            t_i = t_vec[i]
            u_i = self.getInput(t_i)
            output[i] = self.computeModel(X_vec[i], t_i, params, u_i)

        return output

    def computeModelFromManyStates(self, X, t, params):
        """
        Computes the output of a model stacking several states
        :param X: [1-dimensional numpy array] Several states stacked into a single 1-dimensional array.
        :param t: [double] Time.
        :param params: [tuple] extra parameters.
        :return: [1-dimensional numpy array] The stacked output.
        """
        nmbrStates = self.getNmbrOfStates()
        nmbrOutputs = self.getNmbrOutputs()
        nmbrStateVecs = X.size/nmbrStates
        output = np.zeros(nmbrStateVecs*nmbrOutputs)

        u = self.getInput(t)
        for i in range(0, nmbrStateVecs):
            x = X[i*nmbrStates:(i+1)*nmbrStates]
            out = self.computeModel(x, t, params, u)
            output[i*nmbrOutputs:(i+1)*nmbrOutputs] = out

        return output

    def computeModelPlusSTMFunction(self, X, t, params):
        """
        Computes the function [F(X,t), STM] which is useful for propagating X and the STM in
        [X_dot, STM_dot] = [F(X,t), STM]
        :param X: [1-dimension numpy array] State.
        :param t: [double] Time.
        :param params: [tuple] Variable parameters.
        :return: [1-dimension numpy array] The output of [F(X,t), STM] in a vector.
        """
        nmbrStates = self.getNmbrOfStates()
        nmbrInputs = self.getNmbrInputs()

        if nmbrInputs > 0:
            u = self.getInput(t)
            dX = self.computeModel(X, t, params, u)
            A = self.computeJacobian(X, t, params, u)
            B = self.computeInputJacobian(X, t, params, u)
        else:
            dX = self.computeModel(X, t, params)
            A = self.computeJacobian(X, t, params)
            B = 0

        for i in range(0, nmbrStates):  # loop for each STM column vector (phi)
            phi = X[(nmbrStates + nmbrStates*i):(nmbrStates + nmbrStates + nmbrStates*i):1]
            dphi = A.dot(phi)
            dX = np.concatenate([dX, dphi])

        for i in range(0, nmbrInputs): # loop for each STM_input column vector (theta)
            # dim(theta) = nmbrStates x numberInputs
            theta = X[(nmbrStates*(nmbrStates+1) + nmbrStates*i):(nmbrStates*(nmbrStates+1) + nmbrStates + nmbrStates*i):1]
            dtheta = A.dot(theta) + B[:,i]

            dX = np.concatenate([dX, dtheta])

        return dX

    def getModelFunction(self):
        """
        Returns a Lambda function to numerically compute the function model.
        :return: [Lambda func] A function to compute F(X,t) or F(X,u,t).
        """
        return self._computeModelFunc

    def getModelPlusSTMFunction(self):
        """
        Returns a Lambda function with the function [F(X,t), STM] or [F(X,t), STM, STM_input] .
        :return: [lambda func] A function to compute [F(X,t), STM] or [F(X,t), STM, STM_input].
        """
        return self._computeModelPlusSTMFunc

    def getModelFromManyStatesFunction(self):
        """
        Returns a lambda function with the method computeModelFromManyStates().
        :return:
        """
        return self._computeModelFromManyStatesFunc

    def getPropagationFunction(self):
        """
        Returns the function to be used for propagation.
        :return:
        """
        return  self._propFunction

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

    def getNmbrInputs(self):
        """
        Returns the number of inputs used.
        :return: [int] Number of inputs.
        """
        return len(self._inputSymb)


    ## ------------------------Private Methods---------------------------
    @abstractmethod
    def computeSymbolicModel(self): pass

    @abstractmethod
    def computeSymbolicJacobian(self): pass

    # OVERRIDE IF THE MODEL HAS AN INPUT VECTOR u(t).
    def computeSymbolicInputJacobian(self):
        return None


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

    def __init__(self, name, stateSymb, params, propagationFunction, inputSymb):
        super(dynamicModelBase, self).__init__(name, stateSymb, params, propagationFunction, inputSymb)
        return

    def getNmbrOutputs(self):
        return self.getNmbrOfStates() # it's the same for a dynamical model!

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

    def __init__(self, name, stateSymb, params, propagationFunction, inputSymb):
        super(orbitalDynamicModelBase, self).__init__(name, stateSymb, params, propagationFunction, inputSymb)
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
        """
        Computes the SMC covariance using the constant-velocity approximation.
        :param t_i_1:
        :param t_i:
        :param Q_i_1:
        :return:
        """
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

    def __init__(self, name, stateSymb, params, observerCoordinates, inputSymb):
        super(observerModelBase, self).__init__(name, stateSymb, params, propagationFunction = 'F', inputSymb=inputSymb) # Observer models just use the model function 'G'

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