######################################################
# dynamicSimulator
#
# Manuel F. Diaz Ramos
#
# This class simulates a system described by
# X_dot = F(X, t).
#
# The class use the DynModel interface with the following methods:
# getF(): Returns the result of the dynamic function F(X,t)
# getA(): Returns the Jacobian of F evaluated at X,t.
# getModelFunction(): Returns a function that permits computing F(X,t).
# getNmbrOfStates()
# getParams(): Return the parameters used by the model
# getProcessSTM()
######################################################

import numpy as np
from scipy.integrate import odeint

class dynamicSimulator:
    """
    Class that simulates a system described by X_dot = F(X, t)
    """

    ## Constructor: DO NOT USE IT!
    def __init__(self):

        self._dynModel = None
        self._statesVec = None
        self._timeVec = None

        return

    @classmethod
    def getDynamicSimulator(cls, dynModel):
        """
        Factory method used to get an instance of the class.
        :param dynModel: Interface dynModel.
        :return: An instance of this class.
        """
        dynSim = dynamicSimulator()

        dynSim._dynModel = dynModel

        return dynSim

    def getStatesVector(self):
        """
        Getter of the states computed by simulate().
        :return:
        """
        return self._statesVec

    def getTimeVector(self):
        """
        Getter of the time vector used in simulate()
        :return:
        """
        return self._timeVec

    def propagate(self, initialState, params, t0, tf, dt, rtol, atol):
        """
        Simulate the model X_dot = F(X, t)
        :param initialState: Initial conditions.
        :param params: Variable parameters for the dynamic model.
        :param t0: Initial time.
        :param tf: Final time.
        :param dt: sample time.
        :param rtol: Relative tolerance of the integrator.
        :param atol: Absolute tolerance of the integrator.
        :return: An array of vectors with the sampled evolution X(t).
        """

        num = int((tf - t0)/dt) + 1
        tf = (num - 1) * dt + t0 # includes the last value
        time = np.linspace(t0, tf, num)

        modelFunc = self._dynModel.getPropagationFunction()

        self._statesVec = odeint(modelFunc, initialState, time, args = (params,), rtol = rtol, atol = atol)
        self._timeVec = time

        return (self._timeVec, self._statesVec)

    def propagateManyStateVectors(self, initialState, params, t0, dt, tf, rtol, atol):

        num = int((tf - t0)/dt) + 1
        tf = (num - 1) * dt + t0 # includes the last value
        time = np.linspace(t0, tf, num)

        modelFunc = self._dynModel.getPropagationFunction()

        statesVec = odeint(modelFunc, initialState, time, args = (params,), rtol = rtol, atol = atol)
        timeVec = time

        final_state = statesVec[-1]

        return (timeVec, statesVec, final_state)


    def propagateWithSTM(self, initial_state, initial_stm, params, t0, dt, tf, rtol, atol):
        """
        Simulate the model X_dot = F(X,t), propagating X(t) and the State Transition Matrix (STM).
        The STM equation is STM_dot = A(t)*STM, where A(t) is the Jacobian of F.
        :param initialState: Initial state conditions.
        :param initial_stm: Initial STM conditions.
        :param params: Variable parameters for the dynamic model.
        :param t0: Initial time.
        :param tf: Final time.
        :param dt: sample time.
        :param rtol: Relative tolerance of the integrator.
        :param atol: Absolute tolerance of the integrator.
        :return:
        """
        num = int((tf - t0))/dt + 1
        tf = (num - 1) * dt + t0 # includes the last value
        timeVec = np.linspace(t0, tf, num)

        state_length = initial_state.size
        stm_shape = initial_stm.shape
        stm_length = initial_stm.size
        total_length = state_length + stm_length

        modelPlusSTMFunc = self._dynModel.getPropagationFunction()
        #modelPlusSTMFunc = self._dynModel.getModelPlusSTMFunction()

        X0 = np.concatenate([initial_state, initial_stm.T.reshape(stm_length)]) # STM reshaped by columns
        X = odeint (modelPlusSTMFunc, X0, timeVec, args = (params,), rtol = rtol, atol = rtol)

        Xf = X[-1] # last state

        final_state = Xf[0:state_length:1]

        final_stm = Xf[state_length:total_length:1]
        final_stm = final_stm.reshape(stm_shape).T

        stms = np.zeros([X.shape[0], state_length, state_length])
        for i in range(0, X.shape[0]) :
            stms[i,:,:] = X[i,state_length:].reshape(stm_shape).T

        states = X[:,0:state_length]

        return (states, stms, timeVec, final_state, final_stm)


    def propagateWithSTMtimeVec(self, initial_state, initial_stm, params, time_vec, rtol, atol):
        """
        Simulate the model X_dot = F(X,t), propagating X(t) and the State Transition Matrix (STM).
        The STM equation is STM_dot = A(t)*STM, where A(t) is the Jacobian of F.
        This overload receives a time vector.
        :param initialState: Initial state conditions.
        :param initial_stm: Initial STM conditions.
        :param params: Variable parameters for the dynamic model.
        :param time_vec: Time vector.
        :param rtol: Relative tolerance of the integrator.
        :param atol: Absolute tolerance of the integrator.
        :return:
        """

        state_length = initial_state.size
        stm_shape = initial_stm.shape
        stm_length = initial_stm.size
        total_length = state_length + stm_length

        modelPlusSTMFunc = self._dynModel.getPropagationFunction()

        X0 = np.concatenate([initial_state, initial_stm.T.reshape(stm_length)]) # STM reshaped by columns
        X = odeint (modelPlusSTMFunc, X0, time_vec, args = (params,), rtol = rtol, atol = rtol)

        Xf = X[-1] # last state

        final_state = Xf[0:state_length:1]

        final_stm = Xf[state_length:total_length:1]
        final_stm = final_stm.reshape(stm_shape).T

        stms = np.zeros([X.shape[0], state_length, state_length])
        for i in range(0, X.shape[0]) :
            stms[i,:,:] = X[i,state_length:].reshape(stm_shape).T

        states = X[:,0:state_length]

        return (states, stms, time_vec, final_state, final_stm)

