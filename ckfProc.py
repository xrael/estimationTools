######################################################
# Conventional Kalman Filter Processor
#
# Manuel F. Diaz Ramos
#
# This class implements a Kalman Filter processor.
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

class ckfProc(sequentialFilterProc) :
    """
    Conventional Kalman Filter Processor (CKF).
    """

    def __init__(self):
        sequentialFilterProc.__init__(self)

        self._xbar_0 = None
        self._Pbar_0 = None

        self._Xref_i_1 = None   # Reference
        self._xhat_i_1 = None   # Deviation estimate

        self._Pbar_i_1 = None   # a-priori covariance

        self._I = None

        self._stm_i_1 = None

        self._Xref_vec = None
        self._xhat_vec = None

        # self._iteration = 0
        # self._nmbrIterations = 0

        self._josephFormFlag = 0

        return

    #def configureFilter(self, Xref_0, xbar_0, Pbar_0, t_0, joseph_flag = False):
    def configureFilter(self, Xbar_0, Pbar_0, t_0):
        """
        Before computing the kalman solution, call this method.
        :param Xref_0: [1-dimensional numpy array] Initial guess of the state.
        :param xbar_0: [1-dimensional numpy array] Deviation from the initial guess (usually 0).
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param t_0: [double] Initial time.
        :param joseph_flag: [boolean] Set to true to propagate the covariance using Joseph Formulation.
        :return:
        """
        sequentialFilterProc.configureFilter(self, Xbar_0, Pbar_0, t_0)

        self._xbar_0 = np.zeros(Xbar_0.size)
        self._xhat_i_1 = np.copy(self._xbar_0)

        self._Xref_i_1 = np.copy(Xbar_0)

        self._I = np.eye(self._dynModel.getNmbrOfStates())
        self._stm_i_1 = np.copy(self._I)

        self._Pbar_i_1 = np.copy(Pbar_0)

        self._I = np.eye(self._dynModel.getNmbrOfStates())

        # # Default iterations
        # self._iteration = 0
        # self._nmbrIterations = 1

        return

    def josephFormulation(self, use_joseph):
        self._josephFormFlag = use_joseph

    def computeNextEstimate(self, i, t_i, Y_i, obs_params, R_i, dt, rel_tol, abs_tol, refTrajectory = None, Q_i_1 = None):
        """
        This method can be called in real time to get the next estimation deviation associated to the current observation.
        :param t_i: [double] Next observation time.
        :param Y_i: [1-dimension numpy array] Observations at time t_i.
        :param obs_params: [tuple] Non-static observation parameters.
        :param R_i: [2-dimensional numpy array] Observation covariance.
        :param dt: [double] time step in advancing from t_i_1 to t_i.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param Q_i_1: [2-dimensional numpy array] Process noise covariance.
        :return:
        """
        params = ()

        if t_i == self._t_i_1:
            Xref_i = self._Xref_i_1
            xbar_i = self._xhat_i_1
            Pbar_i = self._P_i_1
            stm_ti_t0 = self._stm_i_1
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
            xbar_i = stm_i.dot(self._xhat_i_1)
            Pbar_i = stm_i.dot(self._P_i_1).dot(stm_i.T)
            if self._dynModel.usingSNC() and Q_i_1 is not None:
                # Process Noise Transition Matrix with constant velocity approximation
                Q = self._dynModel.getSncCovarianceMatrix(self._t_i_1, t_i, Xref_i + xbar_i, Q_i_1) # xbar_i should be 0 in the EKF
                Pbar_i = Pbar_i + Q
            elif self._dynModel.usingDMC() and Q_i_1 is not None:
                Q = self._dynModel.getSmcCovarianceMatrix(self._t_i_1, t_i, Q_i_1)
                Pbar_i = Pbar_i + Q

        # Read Observation
        obP = obs_params
        Htilde_i = self._obsModel.computeJacobian(Xref_i, t_i, obP)
        y_i = Y_i - self._obsModel.computeModel(Xref_i, t_i, obP)

        K_i = Pbar_i.dot(Htilde_i.T).dot(self._invert(Htilde_i.dot(Pbar_i).dot(Htilde_i.T) + R_i))

        # Measurement Update
        predicted_residuals_i = y_i - Htilde_i.dot(xbar_i)
        xhat_i = xbar_i + K_i.dot(predicted_residuals_i)
        P_i = self._computeCovariance(Htilde_i, K_i, Pbar_i, R_i)

        self._t_i_1 = t_i
        self._Xref_i_1 = Xref_i
        self._xhat_i_1 = xhat_i
        self._Xhat_i_1 = Xref_i + xhat_i
        self._P_i_1 = P_i
        self._Pbar_i_1 = Pbar_i
        self._prefit_residual = y_i
        self._postfit_residual = y_i - Htilde_i.dot(xhat_i)

        self._stm_i_1 = stm_ti_t0 # STM from t_(i-1) to t_0

        return

    def integrateAllBatch(self, X_0, time_vec, rel_tol, abs_tol, params):
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

            self._xhat_i_1 = np.copy(self._xbar_0)
            self._Xref_i_1 = np.copy(Xbar_0)
            self._stm_i_1 = np.copy(self._I)

            return True
        else:
            return False

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
        self._xhat_vec[i, :] = self.getDeviationEstimate()
        return

    # def iterateSmoothedCKF(self, Xref_0, xbar_0, Pbar_0, t_0, joseph_flag, obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, iterations, Q = None):
    #
    #     for i in range(0, iterations):
    #         self.configureFilter(Xref_0, xbar_0, Pbar_0, t_0, joseph_flag)
    #
    #         self.processAllObservations(obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, Q)
    #
    #         (Xhat_ckf,
    #          xhat_ckf,
    #          Xref_ckf,
    #          P_ckf,
    #          prefit_ckf,
    #          postfit_ckf,
    #          Pbar_ckf,
    #          stm_ckf) = self.processAllObservations(obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, Q)
    #
    #         (Xhat_smoothed, xhat_ckf_smoothed, P_ckf_smoothed) = self.getSmoothedSolution(Xref_ckf, xhat_ckf, P_ckf, Pbar_ckf, stm_ckf)
    #
    #         Xref_0 = Xref_0 + xhat_ckf_smoothed[0] # Updating initial guess
    #
    #         xbar_0 = xbar_0 - xhat_ckf_smoothed[0]
    #
    #     return
    #
    # def getSmoothedSolution(self, Xref, xhat, P, Pbar, stm):
    #     """
    #     Smoother. Returns the smoothed solution using all the observations.
    #     It's not suited for real time processing. It should use all the observations.
    #     Use it after calling processAllObservations()
    #     :param xhat:
    #     :param P:
    #     :param Pbar:
    #     :param stm:
    #     :return:
    #     """
    #     xhat_shape = np.shape(xhat)
    #     obs_length = xhat_shape[0]
    #     state_length = xhat_shape[1]
    #
    #     # Smoothing
    #     l = obs_length - 1 # Smoothing using observations from 0 to l (the last one)
    #     S = np.zeros((l, state_length, state_length))
    #     xhat_smoothed = np.zeros((obs_length, state_length))
    #     P_smoothed = np.zeros((obs_length,state_length, state_length))
    #     Xhat_smoothed = np.zeros((obs_length, state_length))
    #     xhat_smoothed[l,:] = xhat[l]
    #     Xhat_smoothed[l,:] = Xref[l] + xhat[l]
    #     P_smoothed[l,:,:] = P[l]
    #     for k in range(l-1, -1, -1):
    #         phi = stm[k+1]
    #         S[k,:,:] = P[k].dot(phi.T).dot(np.linalg.inv(Pbar[k+1]))
    #         xhat_smoothed[k,:] = xhat[k] + S[k].dot(xhat_smoothed[k+1] - phi.dot(xhat[k]))
    #         P_smoothed[k,:,:] = P[k] + S[k].dot(P_smoothed[k+1] - Pbar[k+1]).dot(S[k].T)
    #
    #         Xhat_smoothed[k,:] = Xref[k] + xhat_smoothed[k]
    #     # end for
    #
    #     return (Xhat_smoothed, xhat_smoothed, P_smoothed)

    # The following getters should be used after calling computeNextEstimate()

    # Current deviation state estimate
    def getDeviationEstimate(self) :
        return self._xhat_i_1

    def getReferenceState(self):
        return self._Xref_i_1

    # A-priori covariance matrix (before processing observations) at current time
    def getAPrioriCovarianceMatrix(self):
        return self._Pbar_i_1

    # STM from t_0 to current time
    def getSTM(self):
        return self._stm_i_1

    #### The following getters should be used after calling processAllObservations()
    # Vector of all state estimates
    def getReferenceStateVector(self):
        return self._Xref_vec

    def getDeviationEstimateVector(self):
        return self._xhat_vec

    # Inversion method used
    def _invert(self, matrix):
        return np.linalg.inv(matrix)

    def _computeCovariance(self, Htilde_i, K_i, Pbar_i, R_i):
        if self._josephFormFlag == False:
            return ((self._I - K_i.dot(Htilde_i)).dot(Pbar_i))
        else: # Joseph Formulation
            aux = (self._I - K_i.dot(Htilde_i))
            return aux.dot(Pbar_i).dot(aux.T) + K_i.dot(R_i).dot(K_i.T)
