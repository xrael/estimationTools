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
import dynamicSimulator as dynSim

class ckfProc :
    """
    Conventional Kalman Filter Processor (CKF).
    """

    def __init__(self):

        self._dynModel = None
        self._obsModel = None
        self._dynSim = None

        self._Xref_0 = None
        self._xbar_0 = None
        self._Pbar_0 = None
        self._t_0 = 0

        self._t_i_1 = 0
        self._Xref_i_1 = None
        self._xhat_i_1 = None
        self._P_i_1 = None

        self._I = None

        self._stm_i_1 = None

        self._prefit_residual = None
        self._postfit_residual = None

        self._josephFormFlag = 0

        return

    # Factory method. Use this to get an instance!
    @classmethod
    def getCkfProc(cls, dynModel, obsModel):
        """
        Factory method to get an instantiation of the class.
        :param dynModel: [dynamicModelBase] Object that implements a dynamic model interface.
        :param obsModel: [observerModelBase] Object that implements an observer model interface.
        :return:
        """
        proc = ckfProc()

        proc._dynModel = dynModel
        proc._obsModel = obsModel

        proc._dynSim = dynSim.dynamicSimulator.getDynamicSimulator(dynModel)

        return proc

    def configureCkf(self, Xref_0, xbar_0, Pbar_0, t_0, joseph_flag):
        """
        Before computing the kalman solution, call this method.
        :param Xref_0: [1-dimensional numpy array] Initial guess of the state.
        :param xbar_0: [1-dimensional numpy array] Deviation from the initial guess (usually 0).
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param t_0: [double] Initial time.
        :param joseph_flag: [boolean] Set to true to propagate the covariance using Joseph Formulation.
        :return:
        """
        # Initial data
        self._Xref_0 = Xref_0
        self._xbar_0 = xbar_0
        self._Pbar_0 = Pbar_0
        self._t_0 = t_0

        self._t_i_1 = t_0
        self._Xref_i_1 = Xref_0
        self._xhat_i_1 = xbar_0
        self._P_i_1 = Pbar_0

        self._I = np.eye(self._dynModel.getNmbrOfStates())

        self._stm_i_1 = self._I # STM matrix from t_0 to t_(i-1)

        self._prefit_residual = None
        self._postfit_residual = None

        self._josephFormFlag = joseph_flag

        return

    def computeNextEstimate(self, t_i, Y_i, obs_params, R_i, dt, rel_tol, abs_tol, Q_i_1 = None):
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
            stm_i = self._I
            Xref_i = self._Xref_i_1
            xbar_i = self._xhat_i_1
            Pbar_i = self._P_i_1
        else:
            (states, stms, time, Xref_i, stm_i)  = self._dynSim.propagateWithSTM(self._Xref_i_1, self._I, params,
                                                                                  self._t_i_1, dt, t_i, rel_tol, abs_tol)

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
        self._P_i_1 = P_i
        self._prefit_residual = y_i
        self._postfit_residual = y_i - Htilde_i.dot(xhat_i)

        self._stm_i_1 = stm_i.dot(self._stm_i_1)

        return xhat_i

    def processAllObservations(self, obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, Q = None):
        """
        Process all observations together using this method.
        :param obs_vector:
        :param obs_time_vector:
        :param obs_params:
        :param R:
        :param dt:
        :param rel_tol:
        :param abs_tol:
        :param Q:
        :return:
        """
        nmbrObsAtEpoch = np.shape(obs_vector)[1]
        nmbrObs = np.size(obs_time_vector)
        nmbrStates = self._dynModel.getNmbrOfStates()

        xhat = np.zeros((nmbrObs, nmbrStates))
        Xhat = np.zeros((nmbrObs, nmbrStates))
        Xref = np.zeros((nmbrObs, nmbrStates))
        P = np.zeros((nmbrObs, nmbrStates, nmbrStates))
        prefit_res = np.zeros((nmbrObs, nmbrObsAtEpoch))
        postfit_res = np.zeros((nmbrObs, nmbrObsAtEpoch))

        for i in range(0, nmbrObs): # Iteration for every observation
            t_i = obs_time_vector[i]
            Y_i = obs_vector[i]
            Q_i = None
            if Q is not None and i >= 1:
                # Only use process noise if the gap in time is not too big
                if t_i - obs_time_vector[i-1] <= 100:
                    Q_i = Q

            self.computeNextEstimate(t_i, Y_i, obs_params[i], R, dt, rel_tol, abs_tol, Q_i)
            xhat[i, :] = self.getDeviationEstimate()
            Xhat[i, :] = self.getNonLinearEstimate()
            Xref[i,:] = self.getReference()
            P[i, :, :] = self.getCovarianceMatrix()
            prefit_res[i,:] = self.getPreFitResidual()
            postfit_res[i,:] = self.getPostFitResidual()
        # End observation processing

        return (Xhat, xhat, Xref, P, prefit_res, postfit_res)

    # The following getters should be used after calling computeNextEstimate()
    # Current state estimate
    def getNonLinearEstimate(self) :
        return (self._Xref_i_1 + self._xhat_i_1)

    # Current deviation state estimate
    def getDeviationEstimate(self) :
        return self._xhat_i_1

    def getReference(self):
        return self._Xref_i_1

    # Current pre-fit residual
    def getPreFitResidual(self):
        return self._prefit_residual

    # Current post-fit residual
    def getPostFitResidual(self):
        return self._postfit_residual

    # Covariance matrix at current time
    def getCovarianceMatrix(self):
        return self._P_i_1

    # STM from t0 to current time
    def getSTM(self):
        return self._stm_i_1

    # Inversion method used
    def _invert(self, matrix):
        return np.linalg.inv(matrix)

    def _computeCovariance(self, Htilde_i, K_i, Pbar_i, R_i):
        if self._josephFormFlag == False:
            return ((self._I - K_i.dot(Htilde_i)).dot(Pbar_i))
        else: # Joseph Formulation
            aux = (self._I - K_i.dot(Htilde_i))
            return aux.dot(Pbar_i).dot(aux.T) + K_i.dot(R_i).dot(K_i.T)