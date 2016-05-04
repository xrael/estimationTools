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
from integrators import rk4Integrator, Integrator

class dynamicSimulator:
    """
    Class that simulates a system described by X_dot = F(X, t)
    """

    ## Constructor: DO NOT USE IT!
    def __init__(self):

        self._dynModel = None
        self._statesVec = None
        self._timeVec = None

        self._integrator = None
        self._integratorEventHandler = None

        return

    @classmethod
    def getDynamicSimulator(cls, dynModel, integrator = Integrator.ODEINT, integratorEventHandler = None):
        """
        Factory method used to get an instance of the class.
        :param dynModel: Interface dynModel.
        :param integrator: [Integrator] Enum that indicates the integrator to be used. Check the Integrator Enum to see what integrators are available.
        :param integratorEventHandler: [func] Function used by some integrators to handle events.
        :return: An instance of this class.
        """
        dynSim = dynamicSimulator()
        dynSim._dynModel = dynModel
        dynSim._integrator = integrator
        dynSim._integratorEventHandler = integratorEventHandler

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

    def propagate(self, initialState, params, t0, tf, dt, rtol = 1e-12, atol = 1e-12):
        """
        Simulate the model X_dot = F(X, t).
        Cannot be used to propagate X_dot = F(X, u, t) if u = u(t).
        :param initialState: Initial conditions.
        :param params: Variable parameters for the dynamic model. If u = const, params[0] = u.
        :param t0: Initial time.
        :param tf: Final time.
        :param dt: sample time.
        :param rtol: Relative tolerance of the integrator.
        :param atol: Absolute tolerance of the integrator.
        :return: An array of vectors with the sampled evolution X(t).
        """

        num = int((tf - t0)/dt) + 1
        tf = (num - 1) * dt + t0 # includes the last value
        time_vec = np.linspace(t0, tf, num)

        modelFunc = self._dynModel.getPropagationFunction()

        if self._integrator == Integrator.RK4:
            self._statesVec = rk4Integrator(modelFunc, initialState, time_vec, args = (params,), event = self._integratorEventHandler)
        else:
            self._statesVec = odeint(modelFunc, initialState, time_vec, args = (params,), rtol = rtol, atol = atol)
        self._timeVec = time_vec

        return (self._timeVec, self._statesVec)

    def propagateManyStateVectors(self, initialState, params, t0, dt, tf, rtol = 1e-12, atol = 1e-12):
        """
        Propagates p state vectors in parallel.
        :param initialState: [1-dimensional numpy array] Contains p initial state vectors in only 1 array.
        :param params: Variable parameters for the dynamic model. If u = const, params[0] = u.
        :param t0: Initial time.
        :param tf: Final time.
        :param dt: sample time.
        :param rtol: Relative tolerance of the integrator.
        :param atol: Absolute tolerance of the integrator.
        :return:
        """
        num = int((tf - t0)/dt) + 1
        tf = (num - 1) * dt + t0 # includes the last value
        time_vec = np.linspace(t0, tf, num)

        modelFunc = self._dynModel.getPropagationFunction()

        if self._integrator == Integrator.RK4:
            statesVec = rk4Integrator(modelFunc, initialState, time_vec, args = (params,), event = self._integratorEventHandler)
        else:
            statesVec = odeint(modelFunc, initialState, time_vec, args = (params,), rtol = rtol, atol = atol)
        timeVec = time_vec

        final_state = statesVec[-1]

        return (timeVec, statesVec, final_state)


    def propagateWithSTM(self, initial_state, initial_stm, params, t0, dt, tf, rtol = 1e-12, atol = 1e-12):
        """
        Simulate the model X_dot = F(X,t), propagating X(t) and the State Transition Matrix (STM).
        It can also simulate the model X_dot = F(X,u,t), propagating X(t), the STM=dX(t)/dX(t_0) and STM_input=dX(t)/du(t_0).
        The STM equation is STM_dot = A(t)*STM, where A(t) is the Jacobian of F.
        The STM input equation is STM_input_dot = A(t)*STM_input + B(t) where B(t)=dF/du.
        :param initialState: Initial state conditions.
        :param initial_stm: Initial STM + STM_input conditions. STM is a nxn matrix while STM_input is nxq (n: Nmbr of states, q: Nmbr of inputs).
        initial_stm is a nx(n+q) matrix.
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
        time_vec = np.linspace(t0, tf, num)

        state_length = initial_state.size
        stm_shape = initial_stm.shape
        stm_length = initial_stm.size
        total_length = state_length + stm_length

        modelPlusSTMFunc = self._dynModel.getPropagationFunction()
        #modelPlusSTMFunc = self._dynModel.getModelPlusSTMFunction()

        X0 = np.concatenate([initial_state, initial_stm.T.reshape(stm_length)]) # STM reshaped by columns

        if self._integrator == Integrator.RK4:
            X = rk4Integrator(modelPlusSTMFunc, X0, time_vec, args = (params,), event = self._integratorEventHandler)
        else:
            X = odeint(modelPlusSTMFunc, X0, time_vec, args = (params,), rtol = rtol, atol = atol)

        Xf = X[-1] # last state

        final_state = Xf[0:state_length:1]

        final_stm = Xf[state_length:total_length:1]
        final_stm = final_stm.reshape(stm_shape).T

        # stms = np.zeros([X.shape[0], state_length, state_length])
        # for i in range(0, X.shape[0]) :
        #     stms[i,:,:] = X[i,state_length:].reshape(stm_shape).T
        #
        # states = X[:,0:state_length]

        stms = np.zeros([X.shape[0], stm_shape[0], stm_shape[1]])
        for i in range(0, X.shape[0]) :
            stms[i,:,:] = X[i,state_length:].reshape(stm_shape).T

        states = X[:,0:state_length]

        return (states, stms, time_vec, final_state, final_stm)


    def propagateWithSTMtimeVec(self, initial_state, initial_stm, params, time_vec, rtol = 1e-12, atol = 1e-12):
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
        :param event: Function to handle events within the integrator
        :return:
        """

        state_length = initial_state.size
        stm_shape = initial_stm.shape
        stm_length = initial_stm.size
        total_length = state_length + stm_length

        modelPlusSTMFunc = self._dynModel.getPropagationFunction()

        X0 = np.concatenate([initial_state, initial_stm.T.reshape(stm_length)]) # STM reshaped by columns

        if self._integrator == Integrator.RK4:
            X = rk4Integrator(modelPlusSTMFunc, X0, time_vec, args = (params,), event = self._integratorEventHandler)
        else:
            X = odeint(modelPlusSTMFunc, X0, time_vec, args = (params,), rtol = rtol, atol = atol)

        Xf = X[-1] # last state

        final_state = Xf[0:state_length:1]

        final_stm = Xf[state_length:total_length:1]
        final_stm = final_stm.reshape(stm_shape).T

        # stms = np.zeros([X.shape[0], state_length, state_length])
        # for i in range(0, X.shape[0]) :
        #     stms[i,:,:] = X[i,state_length:].reshape(stm_shape).T

        stms = np.zeros([X.shape[0], stm_shape[0], stm_shape[1]])
        for i in range(0, X.shape[0]) :
            aux_stm = X[i,state_length:]
            aux = aux_stm.reshape((stm_shape[1], stm_shape[0])).T
            stms[i,:,:] = aux

        states = X[:,0:state_length]

        return (states, stms, time_vec, final_state, final_stm)