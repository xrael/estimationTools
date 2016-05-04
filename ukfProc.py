######################################################
# Unscented Kalman Processor
#
# Manuel F. Diaz Ramos
#
# This class implements an Unscented Kalman Filter (UKF) processor.
# It's an extension of the ckfProc class.
# It relies on:
# 1) A dynamical model which implements the interface dynamicModelBase.
# 2) An observation model which implements the interface observerModelBase
######################################################

import numpy as np
import numpy.matlib as matnp
import dynamicSimulator as dynSim
from sequentialFilter import sequentialFilterProc
import scipy.linalg as splin

class ukfProc(sequentialFilterProc):
    """
    Unscented Kalman Filter Processor (UKF).
    """

    ##-----------------Attributes---------------
    _weights_mean = None
    _weights_covariance = None
    _gamma_p = 0
    _sigma_points_i_1 = None
    ##------------------------------------------

    def __init__(self):
        sequentialFilterProc.__init__(self)

        self._weights_mean = None
        self._weights_covariance = None
        self._gamma_p = 0

        self._sigma_points_i_1 = None

        return

    def configureFilter(self, X_0, Pbar_0, t_0):
        """
        Before computing the UKF solution, call this method.
        :param Xbar_0: [1-dimensional numpy array] Initial guess of the state.
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param t_0: [double] Initial time.
        :return:
        """
        # Initial data
        sequentialFilterProc.configureFilter(self, X_0, Pbar_0, t_0)

        n = self._dynModel.getNmbrOfStates()
        self._weights_mean = np.zeros(2*n+1)
        self._weights_covariance = np.zeros(2*n+1)

        # DEFAULT alpha and beta
        alpha = 1.0
        beta = 2.0

        kappa_p = 3 - n
        lambda_p = alpha**2 * (n + kappa_p) - n
        self._gamma_p = np.sqrt(n + lambda_p)

        self._weights_mean[0] = lambda_p / (n + lambda_p)
        self._weights_covariance[0] = lambda_p / (n + lambda_p) + (1 - alpha**2  + beta)

        self._weights_mean[1:] = 1.0/(2*(n + lambda_p))
        self._weights_covariance[1:] = 1.0/(2*(n + lambda_p))

        return

    def setAlphaBeta(self, alpha, beta):
        """
        alpha = 1.0, beta = 2.0 are used as default. Use this method to change this values
        :param alpha:
        :param beta:
        :return:
        """
        n = self._dynModel.getNmbrOfStates()

        kappa_p = 3 - n
        lambda_p = alpha**2 * (n + kappa_p) - n
        self._gamma_p = np.sqrt(n + lambda_p)

        self._weights_mean[0] = lambda_p / (n + lambda_p)
        self._weights_covariance[0] = lambda_p / (n + lambda_p) + (1 - alpha**2  + beta)

        self._weights_mean[1:] = 1.0/(2*(n + lambda_p))
        self._weights_covariance[1:] = 1.0/(2*(n + lambda_p))

        return

    def computeNextEstimate(self, i, t_i, Y_i, obs_params, R_i, dt,  rel_tol = 1e-12, abs_tol = 1e-12, refTrajectory = None, Q_i_1 = None):
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
        :param refTrajectory: NOT USED IN THE UKF.
        :param Q_i_1: [2-dimensional numpy array] Process noise covariance (or Power Spectral Density, depends on the function used).
        :return:
        """
        params = ()
        n = self._dynModel.getNmbrOfStates()
        m = self._obsModel.getNmbrOutputs()
        s_nmbr = 2*n+1 # Number of sigma points

        Xbar_i = np.zeros(n)
        Pbar_i = np.zeros((n, n))
        Y_mean = np.zeros(m)
        sigma_points_i = np.zeros((s_nmbr, n))
        obs_sigma_points = np.zeros((s_nmbr, m))

        self._sigma_points_i_1 = self.computeSigmaPoints(self._Xhat_i_1, self._P_i_1, self._gamma_p)

        if t_i == self._t_i_1:
            sigma_points_i = self._sigma_points_i_1
            #sigma_points_mat = np.reshape(sigma_points_i,(2*n + 1, n)).T
            #Xbar_i = self._Xhat_i_1
            #Pbar_i = self._P_i_1
        else:
            # Time update
            # TODO: This is very inneficient!!! Just integrate all the sigma points together
            for i in range(0,s_nmbr):
                x = self._sigma_points_i_1[i]
                (time, states) = self._dynSim.propagate(x, params, self._t_i_1, t_i, dt, rel_tol, abs_tol)
                sigma_points_i[i,:] = states[-1]
                # (states, stms, time, sigma_points_i[i*6:(i+1)*6], stm)  = self._dynSim.propagateWithSTM(self._sigma_points_i_1[i*6:(i+1)*6], np.eye(6), params,
                #                                                                   self._t_i_1, dt, t_i, rel_tol, abs_tol)


            #(timeVec, statesVec, sigma_points_i) = self._dynSim.propagateManyStateVectors(self._sigma_points_i_1, params, self._t_i_1, dt, t_i, rel_tol, abs_tol)

        #sigma_points_mat = np.reshape(sigma_points_i,(2*n + 1, n)).T
        for i in range(0, s_nmbr):
            Xbar_i = Xbar_i + sigma_points_i[i] * self._weights_mean[i]

        #Xbar_i = sigma_points_mat.dot(self._weights_mean)

        #Xbar_i_mat = np.matlib.repmat(Xbar_i, 2*n+1, 1).T

        #aux = sigma_points_mat - Xbar_i_mat

        # Pbar_i = np.zeros((n, n))
        # for i in range(0, 2*n+1):
        #     aux_col = sigma_points_mat[:] - Xbar_i
        #     Pbar_i = Pbar_i + np.outer(aux_col, aux_col) * self._weights_covariance[i]


        for i in range(0, s_nmbr):
            aux_col = sigma_points_i[i] - Xbar_i
            Pbar_i = Pbar_i + np.outer(aux_col, aux_col) * self._weights_covariance[i]

        if self._dynModel.usingNoiseCompensation():
                Q = self._dynModel.computeNoiseCovariance(self._t_i_1, t_i, Xbar_i, Q_i_1, params)
                Pbar_i = Pbar_i + Q

        # if self._dynModel.usingSNC() and Q_i_1 is not None:
        #     # Process Noise Transition Matrix with constant velocity approximation
        #     Q = self._dynModel.getSncCovarianceMatrix(self._t_i_1, t_i, Xbar_i, Q_i_1) # xbar_i should be 0 in the EKF
        #     Pbar_i = Pbar_i + Q
        #     # sigma_points_i = self.computeSigmaPoints(Xbar_i, Pbar_i, self._gamma_p)
        #     # sigma_points_mat = np.reshape(sigma_points_i,(2*n + 1, n)).T
        # elif self._dynModel.usingDMC() and Q_i_1 is not None:
        #     Q = self._dynModel.getSmcCovarianceMatrix(self._t_i_1, t_i, Q_i_1)
        #     Pbar_i = Pbar_i + Q

        # Recompute the sigma points to incorporate process noise
        sigma_points_i = self.computeSigmaPoints(Xbar_i, Pbar_i, self._gamma_p)
        #sigma_points_mat = np.reshape(sigma_points_i,(2*n + 1, n)).T

        # Read Observation

        # Measurement update
        # obs_sigma_points = self._obsModel.computeModelFromManyStates(sigma_points_i, t_i, obP)

        for i in range(0, s_nmbr):
            obs_sigma_points[i,:] = self._obsModel.computeModel(sigma_points_i[i], t_i, obs_params)

        #obs_sigma_points_mat = np.reshape(obs_sigma_points,(2*n+1, m)).T # Each sigma point in a column
        for i in range(0, s_nmbr):
            Y_mean = Y_mean + obs_sigma_points[i] * self._weights_mean[i]

        #Y_mean = obs_sigma_points_mat.dot(self._weights_mean) # Each column is multiplied by the weight and all are summed up

        #Y_mean_mat = np.matlib.repmat(Y_mean, 2*n+1, 1).T # Y_mean repeated in columns

        #aux_y = obs_sigma_points_mat - Y_mean_mat # Difference in columns

        #Xbar_i_mat = np.matlib.repmat(Xbar_i, 2*n+1, 1).T
        #aux_x = sigma_points_mat - Xbar_i_mat # Difference in columns

        Pyy = np.array(R_i)
        Pxy = np.zeros((n, m))
        # for i in range(0, 2*n+1):
        #     aux_y_col = obs_sigma_points_mat[:,i] - Y_mean
        #     aux_x_col = sigma_points_mat[:,i] - Xbar_i
        #     Pyy = Pyy + np.outer(aux_y_col, aux_y_col) * self._weights_covariance[i]
        #     Pxy = Pxy + np.outer(aux_x_col, aux_y_col) * self._weights_covariance[i]

        for i in range(0, 2*n+1):
            aux_y_col = obs_sigma_points[i,:] - Y_mean
            aux_x_col = sigma_points_i[i,:] - Xbar_i
            Pyy = Pyy + np.outer(aux_y_col, aux_y_col) * self._weights_covariance[i]
            Pxy = Pxy + np.outer(aux_x_col, aux_y_col) * self._weights_covariance[i]

        K_i = Pxy.dot(np.linalg.inv(Pyy))

        # New state
        self._prefit_residual = Y_i - Y_mean
        self._Xhat_i_1 = Xbar_i + K_i.dot(self._prefit_residual)
        self._P_i_1 = Pbar_i - K_i.dot(Pyy.dot(K_i.T))
        self._sigma_points_i_1 = sigma_points_i
        self._t_i_1 = t_i

        self._postfit_residual = Y_i - self._obsModel.computeModel(self._Xhat_i_1, t_i, obP)

        return self._Xhat_i_1

    # def processAllObservations(self, obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, Q = None):
    #     """
    #     Process all observations together using this method.
    #     :param obs_vector:
    #     :param obs_time_vector:
    #     :param obs_params:
    #     :param R:
    #     :param dt:
    #     :param rel_tol:
    #     :param abs_tol:
    #     :param Q: [2-dim numpy array] Process noise covariance.
    #     :return:
    #     """
    #     nmbrObsAtEpoch = np.shape(obs_vector)[1]
    #     nmbrObs = np.size(obs_time_vector)
    #     nmbrStates = self._dynModel.getNmbrOfStates()
    #
    #     Xhat = np.zeros((nmbrObs, nmbrStates))
    #     P = np.zeros((nmbrObs, nmbrStates, nmbrStates))
    #     prefit_res = np.zeros((nmbrObs, nmbrObsAtEpoch))
    #     postfit_res = np.zeros((nmbrObs, nmbrObsAtEpoch))
    #
    #     for i in range(0, nmbrObs): # Iteration for every observation
    #         t_i = obs_time_vector[i]
    #         Y_i = obs_vector[i]
    #         Q_i = None
    #         if Q is not None and i >= 1: # Only use process noise if the gap in time is not too big
    #             if t_i - obs_time_vector[i-1] <= 100:
    #                 Q_i = Q
    #
    #         self.computeNextEstimate(t_i, Y_i, obs_params[i], R, dt, rel_tol, abs_tol, Q_i)
    #         Xhat[i, :] = self.getNonLinearEstimate()
    #         P[i, :, :] = self.getCovarianceMatrix()
    #         prefit_res[i,:] = self.getPreFitResidual()
    #         postfit_res[i,:] = self.getPostFitResidual()
    #     # End observation processing
    #
    #     return (Xhat, P, prefit_res, postfit_res)


    def computeSigmaPoints(cls, X, P, gamma):
        n = X.size
        #sqrt_P = np.linalg.cholesky(P)
        sqrt_P = splin.sqrtm(P)
        #aux = matnp.repmat(X, n, 1)
        sigma_points = np.zeros((2*n+1, n))
        sigma_points[0,:] = X
        for i in range(0, n):
            sigma_points[1+i,:] = X + gamma * sqrt_P[:,i]
        for i in range(0,n):
            sigma_points[1+n+i,:] = X - gamma * sqrt_P[:,i]

        # aux1 = np.ndarray.flatten(aux + gamma * sqrt_P)
        # aux2 = np.ndarray.flatten(aux - gamma * sqrt_P)
        # sigma_points = np.concatenate((X, aux1, aux2))

        return sigma_points

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