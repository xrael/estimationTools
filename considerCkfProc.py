######################################################
# CKF with consider parameters analysis Processor
#
# Manuel F. Diaz Ramos
#
# This class implements a Kalman Filter with consider parameters.
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
from ckfProc import ckfProc
import matplotlib.pyplot as plt

class considerParamsCkfProc(ckfProc) :
    """
    Conventional Kalman Filter Processor (CKF) with consider parameters.
    """

    def __init__(self):
        ckfProc.__init__(self)

        self._S_i_1 = None      # Sensitivity matrix
        self._xhatc_i_1 = None  # State deviation with consider parameters

        self._P_c_i_1 = None
        self._P_xc_i_1 = None
        self._Xhat_c_i_1 = None

        self._Pbar_cc = None
        self._Cref = None       # Parameters reference
        self._cbar = None       # Parameters' deviation

        self._posfit_consider_residual = None

        self._theta_i_1_0 = None  # From t0 to t_i
        self._theta_i_1 = None    # From t_i_i to t_i

        self._zeroMat = None

        self._xhat_c_vec = None
        self._Xhat_c_vec = None

        self._Pc_vec = None
        self._Pxc_vec = None

        self._theta_vec = None
        self._theta_t0_vec = None

        self._psi_t0_vec = None

        self._postfit_consider_res_vec = None

        return

    #def configureFilter(self, Xref_0, xbar_0, Pbar_0, t_0, joseph_flag = False):
    def considerAnalysis(self, Xbar_0, Pbar_0, t_0, C_ref, cbar, Pbar_cc):
        """
        Before computing the kalman solution, call this method.
        :param Xref_0: [1-dimensional numpy array] Initial guess of the state.
        :param xbar_0: [1-dimensional numpy array] Deviation from the initial guess (usually 0).
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param t_0: [double] Initial time.
        :param C_ref:
        :param cbar:
        :param Pbar_cc
        :return:
        """
        ckfProc.configureFilter(self, Xbar_0, Pbar_0, t_0)

        q = C_ref.size
        n = Xbar_0.size

        self._S_i_1 = np.zeros((n,q))
        self._xhatc_i_1 = np.zeros(n)

        self._P_c_i_1 = np.copy(Pbar_0)
        self._P_xc_i_1 = np.zeros((n,q))
        self._Xhat_c_i_1 = np.zeros(n)

        self._Pbar_cc = np.copy(Pbar_cc)
        self._Cref = C_ref       # Parameters reference
        self._cbar = cbar       # Parameters' deviation

        self._posfit_consider_residual = None

        self._theta_i_1 = np.zeros((n,q))
        self._theta_i_1_0 = np.zeros((n,q))
        self._zeroMat = np.zeros((n,q))

        return

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
            stm_ti_t0 = self._stm_i_1_0
            stm_i = self._I #self._stm_i_1
            theta_i = self._zeroMat #self._theta_i_1
            theta_ti_t0 = self._theta_i_1_0

            Sbar_i = self._S_i_1
        else:
            n = self._Xref_i_1.size # nmbr of states
            if refTrajectory is None: # Integrate
                stms_i_1 = np.concatenate((self._I, self._zeroMat), axis=1) # [STM | STM_input]
                (states, stms, time, Xref_i, stms_i)  = self._dynSim.propagateWithSTM(self._Xref_i_1, stms_i_1, params,
                                                                                 self._t_i_1, dt, t_i, rel_tol, abs_tol)

                stm_i = stms_i[:n,:n]
                theta_i = stms_i[n:,n:]

                stm_ti_t0 = stm_i.dot(self._stm_i_1_0) # STM from t_0 to t_i
                theta_ti_t0 = theta_i + stm_i.dot(self._theta_i_1_0)
            else: # The whole batch has been processed and the reference trajectory is already available
                Xref_i = refTrajectory[0][i]

                aux_i = refTrajectory[1][i]
                aux_i_1 = refTrajectory[1][i-1]

                stm_ti_t0 = aux_i[:,:n]
                theta_ti_t0 = aux_i[:,n:]

                stm_ti_1_t0 = aux_i_1[:,:n]
                theta_ti_1_t0 = aux_i_1[:,n:]

                stm_i = stm_ti_t0.dot(np.linalg.inv(stm_ti_1_t0)) # STM(t_i, t_i_1)
                theta_i = theta_ti_t0 - stm_i.dot(theta_ti_1_t0)

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

            Sbar_i = stm_i.dot(self._S_i_1) + theta_i

        #xbar_c_i = xbar_i + Sbar_i.dot(self._cbar)
        #Pbar_c_i = Pbar_i + Sbar_i.dot(self._Pbar_cc).dot(Sbar_i.T)
        #Pbar_xc_i = Sbar_i.dot(self._Pbar_cc)

        # Read Observation
        Htilde_i = self._obsModel.computeJacobian(Xref_i, t_i, obs_params, self._Cref)
        Htilde_c_i = self._obsModel.computeInputJacobian(Xref_i, t_i, obs_params, self._Cref)
        y_i = Y_i - self._obsModel.computeModel(Xref_i, t_i, obs_params, self._Cref)

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

        # Consider parameters values
        self._S_i_1 = (self._I - K_i.dot(Htilde_i)).dot(Sbar_i) - K_i.dot(Htilde_c_i)
        self._xhatc_i_1 = self._xhat_i_1 + np.squeeze(self._S_i_1.dot(self._cbar))
        self._P_c_i_1 = self._P_i_1 + self._S_i_1.dot(self._Pbar_cc.dot(self._S_i_1.T))
        self._P_xc_i_1 = self._S_i_1.dot(self._Pbar_cc)
        self._Xhat_c_i_1 = Xref_i + self._xhatc_i_1

        self._prefit_residual = y_i
        self._postfit_residual = y_i - Htilde_i.dot(xhat_i)
        self._posfit_consider_residual = y_i - Htilde_i.dot(self._xhatc_i_1) #- np.squeeze(Htilde_c_i.dot(self._cbar))

        self._stm_i_1_0 = stm_ti_t0 # STM from t_(i-1) to t_0
        self._stm_i_1 = stm_i

        self._theta_i_1_0 = theta_ti_t0
        self._theta_i_1 = theta_i

        return

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
        initial_stms = np.concatenate((self._I, self._zeroMat), axis=1) # [STM | STM_input]
        (states, stms, time, Xref_f, stm_f) = self._dynSim.propagateWithSTMtimeVec(X_0, initial_stms, params, time_vec, rel_tol, abs_tol)
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
        self._stm_i_1_0 = stm_f.dot(self._stm_i_1_0)
        self._stm_i_1 = stm_f
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
            xhat_0 = np.linalg.inv(self._stm_i_1_0).dot(self._xhat_i_1)
            Xbar_0 = self._Xhat_0 + xhat_0

            self._xbar_0 = self._xbar_0 - xhat_0
            self._Xhat_0 = np.copy(Xbar_0)

            self._t_i_1 = self._t_0
            self._Xhat_i_1 = np.copy(Xbar_0)
            self._P_i_1 = np.copy(self._P_0)

            self._xhat_i_1 = np.copy(self._xbar_0)
            self._Xref_i_1 = np.copy(Xbar_0)
            self._stm_i_1_0 = np.copy(self._I)
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
        super(considerParamsCkfProc,self).setMoreVectors(nmbrObs, nmbrStates, nmbrObsAtEpoch)

        self._xhat_c_vec = np.zeros((nmbrObs, nmbrStates))
        self._Xhat_c_vec = np.zeros((nmbrObs, nmbrStates))

        self._Pc_vec = np.zeros((nmbrObs, nmbrStates, nmbrStates))
        self._Pxc_vec = np.zeros((nmbrObs, nmbrStates, self._Cref.size))

        self._postfit_consider_res_vec = np.zeros((nmbrObs, nmbrObsAtEpoch))

        q = self._Cref.size # Number of consider parameters

        self._theta_vec = np.zeros((nmbrObs, nmbrStates, q))
        self._theta_t0_vec = np.zeros((nmbrObs, nmbrStates, q))

        self._psi_t0_vec = np.zeros((nmbrObs, nmbrStates + q, nmbrStates + q))

        return

    def assignMoreVectors(self, i):
        """
        OVERLOAD THIS METHOD if more vectors are to be used inside processAllObservations().
        Example: deviations, STMs, a priori values, etc.
        Use this method to assign other vector created in setMoreVectors()
        :param i: [int] Observation index.
        :return: void
        """
        super(considerParamsCkfProc,self).assignMoreVectors(i)

        self._xhat_c_vec[i,:] = self.getConsiderDeviationEstimate()
        self._Xhat_c_vec[i,:] = self.getConsiderEstimate()

        self._Pc_vec[i,:,:] = self.getConsiderCovariance()
        self._Pxc_vec[i,:,:] = self.getCrossConsiderCovariance()

        self._postfit_consider_res_vec[i,:] = self.getConsiderPostfitResidual()

        self._theta_vec[i,:,:] = self.getThetafromLastState()
        self._theta_t0_vec[i,:,:] = self.getThetafromt0()

        q = self._Cref.size # Number of consider parameters
        n = self._xbar_0.size   # Number of states

        self._psi_t0_vec[i,:n,:n] = self.getSTMfromt0()
        self._psi_t0_vec[i,:n,n:] = self.getThetafromt0()
        self._psi_t0_vec[i,n:,:n] = np.zeros((q,n))
        self._psi_t0_vec[i,n:,n:] = np.eye(q)

        return

    # The following getters should be used after calling computeNextEstimate()

    # Current deviation state estimate
    def getConsiderDeviationEstimate(self) :
        return self._xhatc_i_1

    def getConsiderEstimate(self):
        return self._Xhat_c_i_1

    def getConsiderCovariance(self):
        return self._P_c_i_1

    def getCrossConsiderCovariance(self):
        return self._P_xc_i_1

    def getConsiderPostfitResidual(self):
        return self._posfit_consider_residual

    # Theta from t_0 to current time
    def getThetafromt0(self):
        return self._theta_i_1_0

    def getThetafromLastState(self):
        return self._theta_i_1


    #### The following getters should be used after calling processAllObservations()
    # Vector of all state estimates
    def getConsiderEstimateVector(self):
        return self._Xhat_c_vec

    def getConsiderDeviationEstimateVector(self):
        return self._xhat_c_vec

    # Vector of all covariance estimates
    def getConsiderCovarianceMatrixVector(self):
        return self._Pc_vec

    def getCrossConsiderCovarianceMatrixVector(self):
        return self._Pxc_vec

    # Vector of all postfit residuals
    def getConsiderPostfitResidualsVector(self):
        return self._postfit_consider_res_vec

    def getThetaMatrixFromLastStateVector(self):
        return self._theta_vec

    def getThetaMatrixfrom0Vector(self):
        return self._theta_t0_vec

    # Complete STM matrix
    def getPsiFromt0Vector(self):
        return self._psi_t0_vec

    def plotConsiderPostfitResiduals(self, R_obs_noise, labels, units, colors, filename):
        """
        Plot the postfit residuals after using processAllObservations() using the consider parameter analysis.
        :param R_obs_noise: [2-dimensional numpy array] Observation covariance.
        :param labels: [list of strings] List with the name of each observation.
        :param units: [List of strings] List with the units of each obserfvations.
        :param colors: [List of strings] List with the colors for each observation.
        :param filename: [string] Path to save the figure.
        :return:
        """
        obs_time_vec = self._obs_time_vec
        postfit = self._postfit_consider_res_vec

        nmbrTotalObs = obs_time_vec.size
        nmbrObs = np.shape(R_obs_noise)[0] # Number of observations per unit time

        plt.figure()
        plt.hold(True)
        for i in range(0, nmbrObs):
            postfit_obs_i = postfit[:,i]
            postfit_RMS = np.sqrt(np.sum(postfit_obs_i**2)/nmbrTotalObs)
            plt.plot(obs_time_vec/3600, postfit_obs_i/np.sqrt(R_obs_noise[i,i]), '.', color=colors[i], label= labels[i] + ' RMS = ' + str(round(postfit_RMS,3)) + ' ' +  units[i])
        plt.axhline(3, color='k',linestyle='--')
        plt.axhline(-3, color='k',linestyle='--')
        plt.legend(prop={'size':8})
        plt.xlim([0, obs_time_vec[-1]/3600])
        plt.ylim([-6,6])
        plt.xlabel('Observation Time $[h]$')
        plt.ylabel('Normalized Consider Post-fit Residuals')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

        return