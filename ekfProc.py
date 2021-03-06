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
        Usually, the CKF is used at the beginning, and the filter switches to the EKF after n observations.
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
        :param refTrajectory: NOT USED IN THE EKF.
        :param Q_i_1: [2-dimensional numpy array] Process noise covariance.
        :return:
        """
        if self._ckf_counter >= self._start_using_EKF_at_obs: # and i >= 1:
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
            Xbar_i = Xref_i+xbar_i
            #else:
                #xbar_i = np.zeros(np.shape(stm_i)[0])
            Pbar_i = stm_i.dot(self._P_i_1).dot(stm_i.T)
            if self._dynModel.usingNoiseCompensation():
                Q = self._dynModel.computeNoiseCovariance(self._t_i_1, t_i, Xbar_i, Q_i_1, params)
                Pbar_i = Pbar_i + Q

            # self._dynModel.normalizeCovariance(Xbar_i, Pbar_i)
            # self._dynModel.normalizeOutput(Xbar_i)
            # Xref_i = Xbar_i
            # xbar_i = Xbar_i - Xref_i


            # if self._dynModel.usingSNC() and Q_i_1 is not None:
            #     # Process Noise Transition Matrix with constant velocity approximation
            #     Q = self._dynModel.getSncCovarianceMatrix(self._t_i_1, t_i, Xref_i + xbar_i, Q_i_1) # xbar_i should be 0 in the EKF
            #     Pbar_i = Pbar_i + Q
            # elif self._dynModel.usingDMC() and Q_i_1 is not None:
            #     Q = self._dynModel.getSmcCovarianceMatrix(self._t_i_1, t_i, Q_i_1)
            #     Pbar_i = Pbar_i + Q

        # Read Observation
        Htilde_i = self._obsModel.computeJacobian(Xref_i, t_i, obs_params)
        Y_computed = self._obsModel.computeModel(Xref_i, t_i, obs_params)
        y_i = Y_i - Y_computed

        y_i_prime = self._obsModel.normalizePrefitResiduals(y_i, Y_i, Y_computed)
        if y_i_prime != None:
            y_i = y_i_prime

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

        Xhat_i = Xref_i + xhat_i         # Non-linear estimate

        if useEKF == True:
            Xref_i += xhat_i                       # Reference update
            xhat_i = np.zeros(np.size(Xref_i))     # New xhat

        # The following lines is useful when dynamics requires switching (e.g. MRPs)
        self._dynModel.normalizeCovariance(Xhat_i, P_i)
        self._dynModel.normalizeOutput(Xhat_i)

        self._t_i_1 = t_i
        self._Xref_i_1 = Xref_i
        self._xhat_i_1 = xhat_i  # The correction is stored even though it's zeroed after updating the reference
        self._Xhat_i_1 = Xhat_i
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