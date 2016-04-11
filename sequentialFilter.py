######################################################
# Sequential Filter Processor
#
# Manuel F. Diaz Ramos
#
# This class is the base class for every sequential filter.
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

from abc import ABCMeta, abstractmethod
import numpy as np
import dynamicSimulator as dynSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class sequentialFilterProc :
    """
    Base class for every sequential filter.
    """
    __metaclass__ = ABCMeta

    ##-----------------Attributes---------------
    _dynModel = None   # Dynamic Model
    _obsModel = None   # Observation Model
    _dynSim = None     # Dynamic Simulator (integrates the equations

    # Initial values
    _t_0 = 0
    _Xhat_0 = None
    _P_0 = None

    _t_i_1 = 0
    _Xhat_i_1 = None   # Non-linear estimate at t_(i-1)
    _P_i_1 = None      # Covariance estimate at t_(i-1)

    _prefit_residual = None
    _postfit_residual = None

    # If all observations are processed in batch
    _Xhat_vec = None
    _P_vec = None
    _prefit_res_vec = None
    _postfit_res_vec = None
    _obs_vec = None
    _obs_time_vec = None

    # Iteration counters
    _iteration = 0
    _nmbrIterations = 0
    ##------------------------------------------

    ##---------------Constructor----------------
    def __init__(self):

        self._dynModel = None   # Dynamic Model
        self._obsModel = None   # Observation Model
        self._dynSim = None     # Dynamic Simulator (integrates the equations

        self._t_i_1 = 0
        self._Xhat_i_1 = None   # Non-linear estimate
        self._P_i_1 = None      # Covariance estimate
        self._prefit_residual = None
        self._postfit_residual = None

        self._Xhat_vec = None
        self._P_vec = None
        self._prefit_res_vec = None
        self._postfit_res_vec = None
        self._obs_vec = None
        self._obs_time_vec = None

        self._iteration = 0
        self._nmbrIterations = 0

        return

    # Factory method. Use this to get an instance!
    @classmethod
    def getFilter(cls, dynModel, obsModel):
        """
        Factory method to get an instantiation of the class.
        :param dynModel: [dynamicModelBase] Object that implements a dynamic model interface.
        :param obsModel: [observerModelBase] Object that implements an observer model interface.
        :return:
        """
        proc = cls()

        obsModel.defineSymbolicState(dynModel.getSymbolicState()) # Redefines the state of the observer
        obsModel.defineSymbolicInput(dynModel.getSymbolicInput())

        proc._dynModel = dynModel
        proc._obsModel = obsModel

        proc._dynSim = dynSim.dynamicSimulator.getDynamicSimulator(dynModel)

        return proc

    def configureFilter(self, Xbar_0, Pbar_0, t_0):
        """
        Before computing the kalman solution, call this method.
        :param Xbar_0: [1-dimensional numpy array] Initial guess of the state.
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param t_0: [double] Initial time.
        :return:
        """

        self._t_0 = t_0
        self._Xhat_0 = np.copy(Xbar_0)
        self._P_0 = np.copy(Pbar_0)

        self._t_i_1 = t_0
        self._Xhat_i_1 = np.copy(Xbar_0)
        self._P_i_1 = np.copy(Pbar_0)
        self._prefit_residual = None
        self._postfit_residual = None

        self._Xhat_vec = None
        self._P_vec = None
        self._prefit_res_vec = None
        self._postfit_res_vec = None
        self._obs_vec = None
        self._obs_time_vec = None

        # Default iterations
        self._iteration = 0
        self._nmbrIterations = 1

        return

    def processAllObservations(self, obs_vector, obs_time_vector, obs_params, R, dt, rel_tol, abs_tol, Q = None):
        """
        Process all observations together using this method.
        :param obs_vector: [2-dimensional numpy array] Each row is an observation vector.
        :param obs_time_vector: [1-dimensional numpy array] Time associated to every observation vector.
        :param obs_params: [2-dimensional numpy array] Each row contains parameters of every observations (like Ground station coordinates or index).
        :param R: [2-dimensional numpy array] Covariance matrix of all the observations.
        :param dt: [double] integration time step.
        :param rel_tol: [double] relative tolerance of the integrator.
        :param abs_tol: [double] absolute tolerance of the integrator.
        :param Q: [2-dimensional numpy array] Process Noise covariance matrix.
        :return:
        """
        nmbrObsAtEpoch = np.shape(obs_vector)[1]
        nmbrObs = np.size(obs_time_vector)
        nmbrStates = self._dynModel.getNmbrOfStates()

        self._obs_vec = obs_vector
        self._obs_time_vec = obs_time_vector

        self._Xhat_vec = np.zeros((nmbrObs, nmbrStates))
        self._P_vec = np.zeros((nmbrObs, nmbrStates, nmbrStates))
        self._prefit_res_vec = np.zeros((nmbrObs, nmbrObsAtEpoch))
        self._postfit_res_vec = np.zeros((nmbrObs, nmbrObsAtEpoch))

        self.setMoreVectors(nmbrObs, nmbrStates, nmbrObsAtEpoch) # Define more vectors to hold data here

        (R_o,Q_o) = self.processCovariances(R, Q) # Change the covariance format if necessary

        while self.iterate():
            print "Iteration number", self._iteration
            # Integrate the whole batch, if possible
            refTraj = self.integrateAllBatch(self._Xhat_i_1, obs_time_vector, rel_tol, abs_tol, ())

            for i in range(0, nmbrObs): # Iteration for every observation
                t_i = obs_time_vector[i]
                Y_i = obs_vector[i]
                Q_i = None
                if Q_o is not None and i >= 1:
                    # Only use process noise if the gap in time is not too big
                    if t_i - obs_time_vector[i-1] <= 100:
                        Q_i = Q_o

                self.computeNextEstimate(i, t_i, Y_i, obs_params[i], R_o, dt, rel_tol, abs_tol, refTraj, Q_i)
                self._Xhat_vec[i, :] = self.getStateEstimate()
                self._P_vec[i, :, :] = self.getCovarianceMatrix()
                self._prefit_res_vec[i,:] = self.getPreFitResidual()
                self._postfit_res_vec[i,:] = self.getPostFitResidual()
                self.assignMoreVectors(i) # Save other data here
            # End observation processing
        # End of iterations

        return

    def integrateAllBatch(self, X_0, time_vec, rel_tol, abs_tol, params):
        """
        This is useful for filters which do not modify the reference trajectory.
        It's useful when using with processAllObservations() to integrate the whole batch of observations.
        Override the integrate() method if the integration of all the batch is to be performed.
        :param X_0: [1-dimension numpy array] Initial state at t_0.
        :param time_vec: [1-dimensional numpy array] Time vector.
        :param rel_tol: relative tolerance of the integrator.
        :param abs_tol: absolute tolerance of the integrator.
        :param params: [tuple] model parameters. Usually not used.
        :return: The trajectory data in a tuple (reference + STMs)
        """
        time0added = False
        if self._t_i_1 < time_vec[0]: # Add the initial time if it's not part of the time vector
            t_vec = np.concatenate((np.array([self._t_i_1]), time_vec))
            time0added = True
        else:
            t_vec = np.copy(time_vec)
        refTraj = self.integrate(X_0, t_vec, rel_tol, abs_tol, params)
        if time0added == True:
            # The initial time is not part of the observation vector and it was added only to integrate
            # It must be removed.
            for i in range(0, len(refTraj)):
                refTraj[i] = refTraj[i][1:]
        return refTraj

    #-----------------------------Getter Methods-----------------------------#
    #### The following getters should be used after calling computeNextEstimate()
    # Current state estimate
    def getStateEstimate(self) :
        return self._Xhat_i_1

    # Covariance matrix at current time
    def getCovarianceMatrix(self):
        return self._P_i_1

    # Current pre-fit residual
    def getPreFitResidual(self):
        return self._prefit_residual

    # Current post-fit residual
    def getPostFitResidual(self):
        return self._postfit_residual

    #### The following getters should be used after calling processAllObservations()
    # Vector of all state estimates
    def getEstimateVector(self):
        return self._Xhat_vec

    # Vector of all covariance estimates
    def getCovarianceMatrixVector(self):
        return self._P_vec

    # Vector of all prefit residuals
    def getPrefitResidualsVector(self):
        return self._prefit_res_vec

    # Vector of all postfit residuals
    def getPostfitResidualsVector(self):
        return self._postfit_res_vec

    #-----------------------------Abstract Methods----------------------------#
    """
    This method can be called in real time to get the next estimation deviation associated to the current observation.
    :param i: [int] Observation index.
    :param t_i: [double] Next observation time.
    :param Y_i: [1-dimension numpy array] Observations at time t_i.
    :param obs_params: [tuple] Non-static observation parameters.
    :param R_i: [2-dimensional numpy array] Observation covariance (or covariance square root. Depends on how computeNextEstimate() is implemented).
    :param dt: [double] time step in advancing from t_i_1 to t_i.
    :param rel_tol: relative tolerance of the integrator.
    :param abs_tol: absolute tolerance of the integrator.
    :param refTrajectory: [tuple with numpy arrays] Reference trajectory and STM (or sigma points) at time t_i. If None, the dynamic model should be integrated. It's used to avoid setting the integrator at every time step if all observation data is to be processed in batch.
    :param Q_i_1: [2-dimensional numpy array] Process noise covariance.
    :return:
    """
    @abstractmethod
    def computeNextEstimate(self, i, t_i, Y_i, obs_params, R_i, dt, rel_tol, abs_tol, refTrajectory = None, Q_i_1 = None): pass


    def integrate(self, X_0, t_vec, rel_tol, abs_tol, params):
        """
        Integrate a trajectory using X_0 as initial condition and obtaining values at the time steps specified in the vector t_vec.
        OVERRIDE THIS METHOD IF THE INTEGRATION OF A WHOLE BATCH FEATURE IS TO BE USED.
        :param X_0:
        :param t_vec:
        :param rel_tol:
        :param abs_tol:
        :param params:
        :return:
        """
        return None

    """
    This method uses the last estimate and propagates forward only using the dynamic model without observations.
    :return:
    """
    @abstractmethod
    def propagateForward(self, tf, dt, rel_tol, abs_tol, params): pass


    """
    This method defines the way the filter is going to be iterated.
    :return:
    """
    @abstractmethod
    def iterate(self) : pass

    def processCovariances(self, R, Q):
        """
        OVERLOAD THIS METHOD if the covariances are to be processed.
        Example: square root methods need the inverse of the square root: R = M^-1*M^-T
        :param R: [2-dimensional numpy array] Observation covariance.
        :param Q: [2-dimensional numpy array] Noise covariance.
        :return: The two matrices modfied if needed
        """
        return (R,Q)

    def setMoreVectors(self, nmbrObs, nmbrStates, nmbrObsAtEpoch):
        """
        OVERLOAD THIS METHOD if more vectors are to be used inside processAllObservations().
        Example: deviations, STMs, a priori values, etc.
        :param nmbrObs: [int] Total number of obserfvation vectors.
        :param nmbrStates: [int] Number of states.
        :param nmbrObsAtEpoch: [int] Number of observations at each epoch (in each observation vector).
        :return: void
        """
        return

    def assignMoreVectors(self, i):
        """
        OVERLOAD THIS METHOD if more vectors are to be used inside processAllObservations().
        Example: deviations, STMs, a priori values, etc.
        Use this method to assign other vector created in setMoreVectors()
        :param i: [int] Observation index.
        :return: void
        """
        return

    #-----------------------------Plotting Methods----------------------------#
    # Only after calling processAllObservations()

    def plotPostfitResiduals(self, R_obs_noise, labels, units, colors, filename):
        """
        Plot the postfit residuals after using processAllObservations()
        :param R_obs_noise: [2-dimensional numpy array] Observation covariance.
        :param labels: [list of strings] List with the name of each observation.
        :param units: [List of strings] List with the units of each obserfvations.
        :param colors: [List of strings] List with the colors for each observation.
        :param filename: [string] Path to save the figure.
        :return:
        """
        obs_time_vec = self._obs_time_vec
        postfit = self._postfit_res_vec

        nmbrTotalObs = obs_time_vec.size
        nmbrObs = np.shape(R_obs_noise)[0] # Number of observations per unit time

        plt.figure()
        plt.hold(True)
        for i in range(0, nmbrObs):
            postfit_obs_i = postfit[:,i]
            postfit_RMS = np.sqrt(np.sum(postfit_obs_i**2)/nmbrTotalObs)
            plt.plot(obs_time_vec/3600, postfit_obs_i/np.sqrt(R_obs_noise[i,i]), '.', color=colors[i], label= labels[i] + ' RMS = ' + str(round(postfit_RMS,4)) + ' ' +  units[i])
        plt.axhline(3, color='k',linestyle='--')
        plt.axhline(-3, color='k',linestyle='--')
        plt.legend(prop={'size':8})
        plt.xlim([obs_time_vec[0]/3600, obs_time_vec[-1]/3600])
        #plt.ylim([-6,6])
        plt.xlabel('Observation Time $[h]$')
        plt.ylabel('Normalized Post-fit Residuals')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

        return

    def plotPrefitResiduals(self, R_obs_noise, labels, units, colors, filename):
        """
        Plot the prefit residuals after using processAllObservations()
        :param R_obs_noise: [2-dimensional numpy array] Observation covariance.
        :param labels: [list of strings] List with the name of each observation.
        :param units: [List of strings] List with the units of each obserfvations.
        :param colors: [List of strings] List with the colors for each observation.
        :param filename: [string] Path to save the figure.
        :return:
        """
        obs_time_vec = self._obs_time_vec
        prefit = self._prefit_res_vec

        nmbrTotalObs = obs_time_vec.size
        nmbrObs = np.shape(R_obs_noise)[0] # Number of observations per unit time

        plt.figure()
        plt.hold(True)
        for i in range(0, nmbrObs):
            prefit_obs_i = prefit[:,i]
            prefit_RMS = np.sqrt(np.sum(prefit_obs_i**2)/nmbrTotalObs)
            plt.plot(obs_time_vec/3600, prefit_obs_i/np.sqrt(R_obs_noise[i,i]), '.', color=colors[i], label= labels[i] + ' RMS = ' + str(round(prefit_RMS,4)) + ' ' +  units[i])
        plt.axhline(3, color='k',linestyle='--')
        plt.axhline(-3, color='k',linestyle='--')
        plt.legend(prop={'size':8})
        plt.xlim([obs_time_vec[0]/3600, obs_time_vec[-1]/3600])
        #plt.ylim([-6,6])
        plt.xlabel('Observation Time $[h]$')
        plt.ylabel('Normalized Pre-fit Residuals')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

        return

    def plotCovarianceEnvelope(self, labels, filename_pos, filename_vel, filename_rest, dividing_factor = 1):
        obs_time_vec = self._obs_time_vec
        P = self._P_vec

        nmbrStates = self._dynModel.getNmbrOfStates()

        plt.figure()
        for i in range(0, 3): # Position
            subp = int(str(3) + '1' + str(i + 1))

            plt.subplot(subp)
            plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P[:,i,i]))/dividing_factor, '--k')
            plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P[:,i,i]))/dividing_factor, '--k')
            plt.xlim([obs_time_vec[0]/3600, obs_time_vec[-1]/3600])
            plt.ylabel(labels[i])
            #plt.legend(prop={'size':8})
        plt.xlabel('Observation Time $[h]$')
        plt.savefig(filename_pos, bbox_inches='tight', dpi=300)
        plt.close()

        plt.figure()
        for i in range(3, 6): # Velocity
            subp = int(str(3) + '1' + str(i-3 + 1))

            plt.subplot(subp)
            plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P[:,i,i]))/dividing_factor, '--k')
            plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P[:,i,i]))/dividing_factor, '--k')
            plt.xlim([obs_time_vec[0]/3600, obs_time_vec[-1]/3600])
            plt.ylabel(labels[i])
            plt.legend(prop={'size':8})
        plt.xlabel('Observation Time $[h]$')
        plt.savefig(filename_vel, bbox_inches='tight', dpi=300)
        plt.close()

        if nmbrStates > 6: # Rest
            plt.figure()
            for i in range(6, nmbrStates):
                subp = int(str(nmbrStates-6) + '1' + str(i-6 + 1))

                plt.subplot(subp)
                plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P[:,i,i])), '--k')
                plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P[:,i,i])), '--k')
                plt.xlim([obs_time_vec[0]/3600, obs_time_vec[-1]/3600])
                plt.ylabel(labels[i])
                plt.legend(prop={'size':8})
            plt.xlabel('Observation Time $[h]$')
            plt.savefig(filename_rest, bbox_inches='tight', dpi=300)
            plt.close()

        return

    def plotStateEstimates(self, labels, filename, state_vec):

        obs_time_vec = self._obs_time_vec
        fig = plt.figure()
        i = 0
        for nmb in state_vec:
            subp = int(str(len(state_vec)) + '1' + str(i + 1))
            plt.subplot(subp)
            plt.plot(obs_time_vec/3600, self._Xhat_vec[:,nmb], '-b')
            plt.plot(obs_time_vec/3600, self._Xhat_vec[:,nmb], '.r')
            plt.xlim([obs_time_vec[0]/3600, obs_time_vec[-1]/3600])
            plt.ylabel(labels[i])
            #plt.legend(prop={'size':8})
            i += 1
        plt.xlabel('Observation Time $[h]$')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()


    def plotTrajectory(self,labels, filename, dividing_factor = 1):

        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.plot(self._Xhat_vec[:,0]/dividing_factor, self._Xhat_vec[:,1]/dividing_factor, self._Xhat_vec[:,2]/dividing_factor, label='Estimated Trajectory')
        ax.legend(prop={'size':8})
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        return