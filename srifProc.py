######################################################
# Square Root Information Filter Processor
#
# Manuel F. Diaz Ramos
#
# This class implements a SRIF processor.
# It relies on:
# 1) A dynamical model which implements
# the following interface:
# computeModel(state, time, params)
# computeJacobian(state, time, params)
# getNmbrOfStates()
# getParams()
# getProcessSTM()
# 2) An observation model which implements
# the following interface:
# computeModel(state, time, params)
# computeJacobian(state, time, params)
# getParams()
######################################################

import numpy as np
from sequentialFilter import sequentialFilterProc
import orthogonalTransformations as orTrans

class srifProc(sequentialFilterProc) :
    """
    Square Root Information Filter Processor.
    """

    ##-----------------Attributes---------------
    _xbar_0 = None
    _Xref_i_1 = None
    _xhat_i_1 = None
    _stm_i_1 = None
    _R_i_1 = None
    _b_i_1 = None
    _I = None

    # _iteration = 0
    # _nmbrIterations = 0
    ##------------------------------------------

    def __init__(self):
        sequentialFilterProc.__init__(self)

        self._xbar_0 = None
        self._Xref_i_1 = None
        self._xhat_i_1 = None
        self._stm_i_1 = None
        self._stm_i_1_0 = None
        self._R_i_1 = None
        self._b_i_1 = None
        self._I = None

        # self._iteration = 0
        # self._nmbrIterations = 0
        return

    def configureFilter(self, Xbar_0, Pbar_0, t_0):
        """
        Before computing the kalman solution, call this method.
        :param Xbar_0: [1-dimensional numpy array] Initial guess of the state.
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param t_0: [double] Initial time.
        :return:
        """
        sequentialFilterProc.configureFilter(self, Xbar_0, Pbar_0, t_0)

        L = np.linalg.cholesky(Pbar_0) # P = L*L^T

        self._xbar_0 = np.zeros(Xbar_0.size)
        self._xhat_i_1 = np.copy(self._xbar_0)

        self._Xref_i_1 = np.copy(Xbar_0)

        self._I = np.eye(self._dynModel.getNmbrOfStates())
        self._stm_i_1 = np.copy(self._I)

        self._R_i_1 = orTrans.backwardsSubstitutionInversion(L)
        self._b_i_1 = self._R_i_1.dot(self._xbar_0)

        self._xhat_vec = None
        self._stm_vec = None
        self._stm_t0_vec = None
        self._Xref_vec = None

        # # Default iterations
        # self._iteration = 0
        # self._nmbrIterations = 1

        return

    def computeNextEstimate(self, i, t_i, Y_i, obs_params, R_sqrt_inv, dt, rel_tol, abs_tol, refTrajectory = None, Q_sqrt_inv = None):
        """
        This method can be called in real time to get the next estimation deviation associated to the current observation.
        :param i: [int] Observation index.
        :param t_i: [double] Next observation time.
        :param Y_i: [1-dimension numpy array] Observations at time t_i.
        :param obs_params: [tuple] Non-static observation parameters.
        :param R_sqrt_inv: [2-dimensional numpy array] Observation covariance inverse square root: R_i = R^-1 R^-T
        :param dt: [double] time step in advancing from t_i_1 to t_i.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param refTrajectory: [tuple with numpy arrays] Reference trajectory and STMs. If None, the dynamic model should be integrated. It's used to avoid setting the integrator at every time step if all observation data is to be processed in batch.
        :param Q_sqrt_inv: [2-dimensional numpy array] Process noise covariance inverse square root: Q = Q_sqrt_inv^-1 * Q_sqrt_inv^-T
        :return:
        """
        params = ()
        n = self._dynModel.getNmbrOfStates()
        p = Y_i.size

        if t_i == self._t_i_1:
            Xref_i = self._Xref_i_1
            Rbar_i = np.copy(self._R_i_1)
            #Rbar_i = orTrans.householderTransformation(Rbar_i)
            bbar_i = self._b_i_1
            stm_ti_t0 = self._stm_i_1
            stm_i = self._I
        else:
            if refTrajectory is None: # Integrate
                (states, stms, time, Xref_i, stm_i)  = self._dynSim.propagateWithSTM(self._Xref_i_1, self._I, params,
                                                                                 self._t_i_1, dt, t_i, rel_tol, abs_tol)
                stm_ti_t0 = stm_i.dot(self._stm_i_1) # STM from t_0 to t_i
            else: # The whole batch has been processed and the reference trajectory is already available
                Xref_i = refTrajectory[0][i]
                stm_ti_t0 = refTrajectory[1][i]
                stm_ti_1_t0 = refTrajectory[1][i-1]
                stm_i = stm_ti_t0.dot(np.linalg.inv(stm_ti_1_t0)) # STM(t_i, t_i_1)

            # Time Update
            bbar_i = self._b_i_1
            Rbar_i = self._R_i_1.dot(np.linalg.inv(stm_i))
            #Rbar_i = orTrans.householderTransformation(Rbar_i)


            #Pbar_i = stm_i.dot(self._P_i_1).dot(stm_i.T)
            if self._dynModel.usingSNC() and Q_sqrt_inv is not None:
                q = Q_sqrt_inv.shape[1]
                gamma_i = self._dynModel.getPNSTM(self._t_i_1, t_i)
                A = np.zeros((n + q, q + n + 1))

                A[:q,:q] = Q_sqrt_inv
                A[q:,:q] = -Rbar_i.dot(gamma_i)
                A[q:,q:(q+n)] = Rbar_i
                A[q:,-1] = bbar_i

                A = orTrans.householderTransformation(A)

                Rbar_i = A[q:,q:(q+n)]
                bbar_i = A[q:,-1]

        # Read Observation
        Htilde_i = self._obsModel.computeJacobian(Xref_i, t_i, obs_params)
        y_i = Y_i - self._obsModel.computeModel(Xref_i, t_i, obs_params)
        # Whitened measurements
        y_i_white = R_sqrt_inv.dot(y_i)
        H_i_white = R_sqrt_inv.dot(Htilde_i)

        A = np.zeros((n + p, n + 1))
        A[0:n,0:n] = np.copy(Rbar_i)
        A[n:,0:n] = np.copy(H_i_white)
        A[0:n,-1] = np.copy(bbar_i)
        A[n:,-1] = np.copy(y_i_white)
        # for i in range(0, n+1):
        #     print np.linalg.norm(A[:,i])

        A = orTrans.householderTransformation(A)
        # for i in range(0, n+1):
        #     print np.linalg.norm(A[:,i])

        self._R_i_1 = A[0:n,0:n]
        self._b_i_1 = A[0:n,-1]

        self._xhat_i_1 = orTrans.backwardsSubstitution(self._R_i_1, self._b_i_1)
        self._Xhat_i_1 = Xref_i + self._xhat_i_1         # Non-linear estimate

        self._t_i_1 = t_i
        self._Xref_i_1 = Xref_i
        self._stm_i_1_0 = stm_ti_t0 # STM from t_0 to t_i
        self._stm_i_1 = stm_i
        self._prefit_residual = y_i
        self._postfit_residual = y_i - Htilde_i.dot(self._xhat_i_1)

        return

    def setMoreVectors(self, nmbrObs, nmbrStates, nmbrObsAtEpoch):
        """
        OVERLOAD THIS METHOD if more vectors are to be used inside processAllObservations().
        Example: deviations, STMs, a priori values, etc.
        :param nmbrObs: [int] Total number of obserfvation vectors.
        :param nmbrStates: [int] Number of states.
        :param nmbrObsAtEpoch: [int] Number of observations at each epoch (in each observation vector).
        :return: void
        """
        self._Xref_vec = np.zeros((nmbrObs, nmbrStates))
        self._xhat_vec = np.zeros((nmbrObs, nmbrStates))
        self._stm_vec = np.zeros((nmbrObs, nmbrStates, nmbrStates))
        self._stm_t0_vec = np.zeros((nmbrObs, nmbrStates, nmbrStates))
        return

    def assignMoreVectors(self, i):
        """
        OVERLOAD THIS METHOD if more vectors are to be used inside processAllObservations().
        Example: deviations, STMs, a priori values, etc.
        Use this method to assign other vector created in setMoreVectors()
        :param i: [int] Observation index.
        :return: void
        """
        self._Xref_vec[i, :] = self.getReferenceState()
        self._xhat_vec[i, :] = self.getStateDeviationEstimate()
        self._stm_vec[i,:,:] = self.getSTMfromLastState()
        self._stm_t0_vec[i,:,:] = self.getSTMfromt0()
        return

    # The following getters should be used after calling computeNextEstimate()

    def getStateDeviationEstimate(self):
        return self._xhat_i_1

    # Covariance matrix at current time
    def getCovarianceMatrix(self):
        Ri = orTrans.forwardSubstitutionInversion(self._R_i_1)
        self._P_i_1 = Ri.dot(Ri.T)
        return self._P_i_1

    def getReferenceState(self):
        return self._Xref_i_1

    # STM from t_0 to current time
    def getSTMfromt0(self):
        return self._stm_i_1_0

    def getSTMfromLastState(self):
        return self._stm_i_1

     #### The following getters should be used after calling processAllObservations()

     # Vector of all state estimates
    def getReferenceStateVector(self):
        return self._Xref_vec

    def getDeviationEstimateVector(self):
        return self._xhat_vec

    def getSTMMatrixFromLastStateVector(self):
        return self._stm_vec

    def getSTMMatrixfrom0Vector(self):
        return self._stm_t0_vec

    def processCovariances(self, R, Q):
        """
        Overriden method from sequentialFilterProc class.
        :param R: [2-dimensional numpy array] Observation covariance.
        :param Q: [2-dimensional numpy array] Noise covariance.
        :return: The inverse of the square root of R and Q.
        """
        LR = np.linalg.cholesky(R) # R = LR*LR^T
        R_o = orTrans.backwardsSubstitutionInversion(LR)

        if Q != None:
            LQ = np.linalg.cholesky(Q) # Q = LQ*LQ^T
            Q_o = orTrans.backwardsSubstitutionInversion(LQ)
        else:
            Q_o = None

        return (R_o, Q_o)

    def integrate(self, X_0, time_vec, rel_tol, abs_tol, params):
        """
        Integrate th whole batch. Possible for the SRIF since the reference trajectory does not change.
        :param X_0: [1-dimension numpy array] Initial state at t_0.
        :param time_vec: [1-dimensional numpy array] Time vector.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param params: [tuple] model parameters. Usually not used.
        :return: The trajectory data in a tuple (reference + STMs)
        """
        (states, stms, time, Xref_f, stm_f) = self._dynSim.propagateWithSTMtimeVec(X_0, self._I, params, time_vec, rel_tol, abs_tol)
        return (states, stms, time)

    def propagateForward(self, dtf, dt, rel_tol, abs_tol, params):
        """
        Propagates the filter forward without observations, only using teh model
        :param dtf: [double] interval of time to propagate forward (The estimation will be advanced from the last observation time in dtf).
        :param dt: [double] Time step. Should be smaller than dtf.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param params: [tuple] model parameters. Usually not used.
        :return:
        """
        #tf = self._t_i_1 + dtf
        #num = int((tf - self._t_i_1)/dt) + 1
        num = int(dtf/dt) + 1
        print "num: ", num
        tf = (num - 1) * dt + self._t_i_1 # includes the last value
        print "t_i: ", self._t_i_1
        print "t_f: ", tf
        time_vec = np.linspace(self._t_i_1, tf, num)
        print "time_vec: ", time_vec
        (states, stms, time, Xref_f, stm_f) = self._dynSim.propagateWithSTMtimeVec(self._Xref_i_1, self._I, params, time_vec, rel_tol, abs_tol)

        nmbrStates = self._dynModel.getNmbrOfStates()

        Xhat_vec_prop = np.zeros((num, nmbrStates))
        xhat_vec_prop = np.zeros((num, nmbrStates))
        P_vec_prop = np.zeros((num, nmbrStates, nmbrStates))
        for i in range(0, num):
            stm_ti_tobs = stms[i] # STM from the propagation initial time to ti
            #stm_ti_tobs = stms[i].dot(np.linalg.inv(self._stm_i_1)) # STM from the propagation initial time to ti
            xhat_vec_prop[i,:] = stm_ti_tobs.dot(self._xhat_i_1)
            Xhat_vec_prop[i,:] = states[i] + xhat_vec_prop[i]
            P_vec_prop[i,:,:] = stm_ti_tobs.dot(self._P_i_1.dot(stm_ti_tobs.T))

        self._Xref_i_1 = Xref_f
        #stm_tf_ti = stm_f.dot(np.linalg.inv(self._stm_i_1)) # STM from the propagation initial time to tf
        stm_tf_ti = stm_f
        self._stm_i_1 = stm_f.dot(self._stm_i_1)
        self._xhat_i_1 = stm_tf_ti.dot(self._xhat_i_1)
        self._Xhat_i_1 = self._Xref_i_1 + self._xhat_i_1
        self._P_i_1 = stm_tf_ti.dot(self._P_i_1.dot(stm_tf_ti.T))
        self._t_i_1 = tf
        return (Xhat_vec_prop, xhat_vec_prop, P_vec_prop, time_vec)


    def setNumberIterations(self, it):
        """

        :param it:
        :return:
        """
        self._nmbrIterations = it
        return

    def iterate(self):
        """
        This method defines the way the batch filter is going to iterate.
        Modify it if another iteration algorithm is desired.
        :return:
        """
        if self._iteration < self._nmbrIterations:
            self._iteration = self._iteration + 1
            xhat_0 = np.linalg.inv(self._stm_i_1).dot(self._xhat_i_1)
            Xbar_0 = self._Xhat_0 + xhat_0

            self._xbar_0 = self._xbar_0 - xhat_0
            self._Xhat_0 = np.copy(Xbar_0)

            self._t_i_1 = self._t_0
            self._Xhat_i_1 = np.copy(Xbar_0)
            self._P_i_1 = np.copy(self._P_0)

            L = np.linalg.cholesky(self._P_0) # P = L*L^T

            self._xhat_i_1 = np.copy(self._xbar_0)
            self._Xref_i_1 = np.copy(Xbar_0)
            self._stm_i_1 = np.copy(self._I)

            self._R_i_1 = orTrans.backwardsSubstitutionInversion(L)
            self._b_i_1 = self._R_i_1.dot(self._xbar_0)

            return True
        else:
            return False
