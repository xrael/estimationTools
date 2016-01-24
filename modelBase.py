import numpy as np
from abc import ABCMeta, abstractmethod

######################################################
# modelBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# functions of a state and time F(X,t).
#
# The interface is:
# computeModel(): Returns the result of the dynamic function F(X,t)
# computeJacobian(): Returns the Jacobian of F evaluated at X,t.
# getModelFunction(): Returns a function that permits computing F(X,t).
# getParameters(): Returns the constant parameters used by F.
# getNmbrOfOutputs(): Return the number of outputs of the function.
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
    ##------------------------------------------

    def __init__(self, stateSymb, params):
        """
        Constructor.
        :param stateSymb: [List with sympy expressions] Symbolic description of the state (List with symbolic expressions).
        :param params: [tuple] Static parameters used by the model.
        :return:
        """
        self._stateSymb = stateSymb
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
# functions dynamic models X_dot = F(X,t).
######################################################
class dynamicModelBase(modelBase):
    """
    Base class of every dynamic model model F(X,t).
    """
    __metaclass__ = ABCMeta

    def __init__(self, stateSymb, params):
        super(dynamicModelBase, self).__init__(stateSymb, params)
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
# observerBase
#
# Manuel F. Diaz Ramos
#
# This class is a base for all classes which compute
# observation model Y = G(X,t).
######################################################
class observerBase(modelBase):
    """
    Base class of every observation model G(X,t).
    Every observation model has, at least, one set of coordinates against which
    the observations is measured.
    """
    __metaclass__ = ABCMeta

    ##-----------------Attributes---------------
    _observerCoordinates = None
    ##------------------------------------------

    def __init__(self, stateSymb, params, observerCoordinates):
        super(observerBase, self).__init__(stateSymb, params)

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
