######################################################
# Extended Kalman Processor
#
# Manuel F. Diaz Ramos
#
# This class implements an Extended Kalman Filter (EKF) processor.
# It's an extension of the ckfProc class.
# It relies on:
# 1) A dynamical model which implements the interface dynamicModelBase.
# 2) An observation model which implements the interface observerModelBase
######################################################

import numpy as np
from ckfProc import ckfProc

class ekfProc(ckfProc) :
    """
    Extended Kalman Filter Processor (EKF).
    """

    def __init__(self):
        ckfProc.__init__(self)

        self._ckf_counter = 0
        self._start_using_EKF_at_obs = 0
        self._lastObsTime = 0
        return

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
        ckfProc.configureFilter(self, Xbar_0, Pbar_0, t_0)

        self._ckf_counter = 0
        self._start_using_EKF_at_obs = 0
        self._lastObsTime = t_0

        return

    def startEKFafter(self, n):
        """
        Usually, the CKF is used at the beginning, and thefilter switches to the EKF after n observations.
        :param start_using_EKF_at_obs:
        :return:
        """
        self._start_using_EKF_at_obs = n
        return

    def computeNextEstimate(self, i, t_i, Y_i, obs_params, R_i, dt, rel_tol, abs_tol, refTrajectory = None, Q_i_1 = None):
        """
        This works as an interface between the computeNextEstimate() interface defined in the sequential filter interface
        and the computeNextEstimateEKF().
        :param i: [int] Number of observation.
        :param t_i: [double] Next observation time.
        :param Y_i: [1-dimension numpy array] Observations at time t_i.
        :param obs_params: [tuple] Non-static observation parameters.
        :param R_i: [2-dimensional numpy array] Observation covariance.
        :param dt: [double] time step in advancing from t_i_1 to t_i.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param refTrajectory: NOT USED IN THE CKF.
        :param Q_i_1: [2-dimensional numpy array] Process noise covariance.
        :return:
        """
        if self._ckf_counter >= self._start_using_EKF_at_obs and i >= 1:
            if t_i - self._lastObsTime <= 100:
                useEKF = True  # Only use EKF after processing start_using_EKF_at_obs observations
            else :  # CHECK THIS!!!!! The ckf should be used for an interval of time
                self._ckf_counter = 0
                useEKF = False
        else:
            self._ckf_counter = self._ckf_counter + 1
            useEKF = False

        self.computeNextEstimateEKF(t_i, Y_i, obs_params, R_i, dt, rel_tol, abs_tol, useEKF, Q_i_1)

        self._lastObsTime = t_i

        return


    def computeNextEstimateEKF(self, t_i, Y_i, obs_params, R_i, dt, rel_tol, abs_tol, useEKF = True, Q_i_1 = None):
        """
        This method can be called in real time to get the next estimation deviation associated to the current observation.
        :param t_i: [double] Next observation time.
        :param Y_i: [1-dimension numpy array] Observations at time t_i.
        :param obs_params: [tuple] Non-static observation parameters.
        :param R_i: [2-dimensional numpy array] Observation covariance.
        :param dt: [double] time step in advancing from t_i_1 to t_i.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param useEKF: [Boolean] Used to switch between CKF (False) and EKF (True) propagation.
        :param Q_i_1: [2-dimensional numpy array] Process noise covariance.
        :return:
        """
        params = ()

        if t_i == self._t_i_1:
            Xref_i = self._Xref_i_1
            xbar_i = self._xhat_i_1
            Pbar_i = self._P_i_1
            stm_ti_t0 = self._stm_i_1_0
            stm_i = self._I
        else:
            (states, stms, time, Xref_i, stm_i)  = self._dynSim.propagateWithSTM(self._Xref_i_1, self._I, params,
                                                                                 self._t_i_1, dt, t_i, rel_tol, abs_tol)
            # Time Update
            #if useEKF == False: # use CKF instead
            xbar_i = stm_i.dot(self._xhat_i_1)
            #else:
                #xbar_i = np.zeros(np.shape(stm_i)[0])
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
        #if useEKF == False: # use CKF instead
        predicted_residuals_i = y_i - Htilde_i.dot(xbar_i)
        xhat_i = xbar_i + K_i.dot(predicted_residuals_i)
        # else:
            #predicted_residuals_i = y_i
            #xhat_i = K_i.dot(predicted_residuals_i)

        post_fit_residuals = y_i - Htilde_i.dot(xhat_i)

        P_i = self._computeCovariance(Htilde_i, K_i, Pbar_i, R_i)

        self._Xhat_i_1 = Xref_i + xhat_i         # Non-linear estimate

        if useEKF == True:
            Xref_i += xhat_i                       # Reference update
            xhat_i = np.zeros(np.size(Xref_i))     # New xhat

        self._t_i_1 = t_i
        self._Xref_i_1 = Xref_i
        self._xhat_i_1 = xhat_i  # The correction is stored even though it's zeroed after updating the reference
        self._P_i_1 = P_i
        self._Pbar_i_1 = Pbar_i
        self._prefit_residual = y_i
        self._postfit_residual = post_fit_residuals

        self._stm_i_1_0 = stm_i.dot(self._stm_i_1) # STM from t_(i-1) to t_0
        self._stm_i_1 = stm_i

        return


    def integrate(self, X_0, t_vec, rel_tol, abs_tol, params):
        """
        Integrate a trajectory using X_0 as initial condition and obtaining values at the time steps specified in the vector t_vec.
        OVERRIDE THIS METHOD IF THE INTEGRATION OF A WHOLE BATCH FEATURE IS TO BE USED.
        THE BATCH OF OBSERVATIONS CANNOT BE INTEGRATED ALL AT ONCE WHEN USING THE CKF BECAUSE
        THE REFERENCE TRAJECTORY IS MODIFIED.
        :param X_0:
        :param t_vec:
        :param rel_tol:
        :param abs_tol:
        :param params:
        :return:
        """
        return None




    def processAllObservations(self, obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, start_using_EKF_at_obs = 0, Q = None):
        """
        Process all observations together using this method.
        :param obs_vector:
        :param obs_time_vector:
        :param obs_params:
        :param R:
        :param dt:
        :param rel_tol:
        :param abs_tol:
        :param Q: [2-dim numpy array] Process noise covariance.
        :return:
        """
        nmbrObsAtEpoch = np.shape(obs_vector)[1]
        nmbrObs = np.size(obs_time_vector)
        nmbrStates = self._dynModel.getNmbrOfStates()

        xhat = np.zeros((nmbrObs, nmbrStates))
        Xhat = np.zeros((nmbrObs, nmbrStates))
        P = np.zeros((nmbrObs, nmbrStates, nmbrStates))
        prefit_res = np.zeros((nmbrObs, nmbrObsAtEpoch))
        postfit_res = np.zeros((nmbrObs, nmbrObsAtEpoch))

        ckf_counter = 0

        for i in range(0, nmbrObs): # Iteration for every observation
            t_i = obs_time_vector[i]
            Y_i = obs_vector[i]
            Q_i = None
            if Q is not None and i >= 1: # Only use process noise if the gap in time is not too big
                if t_i - obs_time_vector[i-1] <= 100:
                    Q_i = Q

            if ckf_counter >= start_using_EKF_at_obs and i >= 1:
                if t_i - obs_time_vector[i-1] <= 20:
                    useEKF = True  # Only use EKF after processing start_using_EKF_at_obs observations
                else :  # CHECK THIS!!!!! The ckf should be used for an interval of time
                    ckf_counter = 0
                    useEKF = False
            else:
                ckf_counter = ckf_counter + 1
                useEKF = False

            self.computeNextEstimate(t_i, Y_i, obs_params[i], R, dt, rel_tol, abs_tol, useEKF, Q_i)
            xhat[i, :] = self.getDeviationEstimate()
            Xhat[i, :] = self.getNonLinearEstimate()
            P[i, :, :] = self.getCovarianceMatrix()
            prefit_res[i,:] = self.getPreFitResidual()
            postfit_res[i,:] = self.getPostFitResidual()
        # End observation processing

        return (Xhat, xhat, P, prefit_res, postfit_res)


