######################################################
# Observation simulator
#
# Manuel F. Diaz Ramos
#
# Class to be used along with a dynamic model and
# an observation model to simulate observations.
######################################################

from dynamicSimulator import dynamicSimulator
from integrators import Integrator
import numpy as np

class observationSimulator:
    """
    Class that simulates observations for a given dynamics and observation models.
    It can be used with additive noise.
    """

    ## Constructor: DO NOT USE IT!
    def __init__(self):

        self._obsModels = None
        self._dynSimulator = None

        self._observations_time_vec = []
        self._observers = []
        self._observations = []
        self._observed_states = []

        self._lastObsIndex = 0

        # self._whiteNoiseFlag = False
        # self._whiteNoiseMean = 0
        # self._whiteNoiseCovariance = 0

        # self._randomWalkNoiseFlag = False
        # self._randomWalkNoiseMean = 0
        # self._randomWalkNoiseCovariance = 0
        return

    @classmethod
    def getObservationSimulator(cls, dynModel, obsModels, integrator = Integrator.ODEINT, integratorEventHandler = None):
        """
        Factory method used to get an instance of the class.
        :param dynModel: Interface dynModel.
        :param integrator: [Integrator] Enum that indicates the integrator to be used. Check the Integrator Enum to see what integrators are available.
        :param integratorEventHandler: [func] Function used by some integrators to handle events.
        :return: An instance of this class.
        """
        obsSim = observationSimulator()

        for obsModel in obsModels:
            obsModel.defineSymbolicState(dynModel.getSymbolicState()) # Redefines the state of the observer
            obsModel.defineSymbolicInput(dynModel.getSymbolicInput())

        obsSim._obsModels = obsModels
        obsSim._dynSimulator = dynamicSimulator.getDynamicSimulator(dynModel, integrator, integratorEventHandler)

        # obsSim._whiteNoiseFlag = False
        # obsSim._whiteNoiseMean = 0
        # obsSim._whiteNoiseCovariance = 0

        # obsSim._randomWalkNoiseFlag = False
        # obsSim._randomWalkNoiseMean = 0
        # obsSim._randomWalkNoiseCovariance = 0

        return obsSim

    # def addWhiteNoise(self, flag, noise_mean, noise_covariance):
    #     """
    #     Use it to add white gaussian noise to the measurements.
    #     :param flag:
    #     :param noise_mean:
    #     :param noise_covariance:
    #     :return:
    #     """
    #     nmbrObs = self._obsModel.getNmbrOutputs()
    #
    #     if flag == True and noise_mean.size == nmbrObs and noise_covariance.shape == (nmbrObs, nmbrObs):
    #         self._whiteNoiseFlag = True
    #         self._whiteNoiseMean = noise_mean
    #         self._whiteNoiseCovariance = noise_covariance
    #         ret = True
    #     else:
    #         self._whiteNoiseFlag = False
    #         self._whiteNoiseMean = 0
    #         self._whiteNoiseCovariance = 0
    #         ret = False
    #
    #     return ret
    #
    # def addRandomWalkNoise(self, flag, noise_mean, noise_covariance):
    #     """
    #     Use it to add random walk gaussian noise to the measurements.
    #     :param flag:
    #     :param noise_mean:
    #     :param noise_covariance:
    #     :return:
    #     """
    #     nmbrObs = self._obsModel.getNmbrOutputs()
    #
    #     if flag == True and noise_mean.size == nmbrObs and noise_covariance.shape == (nmbrObs, nmbrObs):
    #         self._randomWalkNoiseFlag = True
    #         self._randomWalkNoiseMean = noise_mean
    #         self._randomWalkNoiseCovariance = noise_covariance
    #         ret = True
    #     else:
    #         self._randomWalkNoiseFlag = False
    #         self._randomWalkNoiseMean = 0
    #         self._randomWalkNoiseCovariance = 0
    #         ret = False
    #
    #     return ret

    def getObservations(self):
        """
        Getter of the observations computed by simulate().
        :return:
        """
        return (self._observers, self._observations)

    def getTimeVector(self):
        """
        Getter of the time vector used in simulate()
        :return:
        """
        return self._observations_time_vec

    def getObservedStates(self):
        """
        Getter of the observed states computed by simulate().
        :return:
        """
        return self._observed_states

    def getDynamicSimulator(self):
        """
        Retrieves the dynamic simulator.
        :return:
        """
        return self._dynSimulator

    def simulate(self, initialState, dynamic_params, t0, tf, dt,  rtol = 1e-12, atol = 1e-12):
        """

        :param initialState:
        :param dynamic_params:
        :param t0:
        :param tf:
        :param dt:
        :param rtol:
        :param atol:
        :param event:
        :return:
        """
        (timeVec, statesVec) = self._dynSimulator.propagate(initialState, dynamic_params, t0, tf, dt, rtol, atol)

        self._observations_time_vec = []
        self._observers = []
        self._observations = []
        self._observed_states = []

        self._lastObsIndex = -1*np.ones(len(self._obsModels))

        for obsModel in self._obsModels: # For each observation model
            observed_time_vec = []
            observers = []
            observations = []
            observed_states = []
            for j in range(0, timeVec.size):
                nmbr_coordinates = obsModel.getNmbrObservers()
                t_j = timeVec[j]
                state_j = statesVec[j]
                for k in range(0, nmbr_coordinates): # For each observer (each ground station or each star tracker)
                    params = (k,)
                    if obsModel.isObservable(state_j, t_j, params):
                        obs = obsModel.computeModel(state_j, t_j, params)
                        if obsModel.addAdditiveNoise():
                            obs += obsModel.noiseModel()

                        obsModel.normalizeOutput(obs)

                        observed_time_vec.append(t_j)
                        observers.append((k,))
                        observations.append(obs)
                        observed_states.append(state_j)

            self._observations_time_vec.append(np.array(observed_time_vec))
            self._observers.append(np.array(observers))
            self._observations.append(np.array(observations))
            self._observed_states.append(np.array(observed_states))

        return (self._observations_time_vec, self._observers, self._observations, self._observed_states)


    def getNextObservation(self, t, obsNmbr):

        obs_time_vec = self._observations_time_vec[obsNmbr]
        obs_vec = self._observations[obsNmbr]

        if self._lastObsIndex[obsNmbr] + 1 < obs_time_vec.size:
            while t >= obs_time_vec[self._lastObsIndex[obsNmbr] + 1]: # There's a new observation available
                self._lastObsIndex[obsNmbr] += 1
                if self._lastObsIndex[obsNmbr] + 1 >= obs_time_vec.size:
                    break
        return obs_vec[self._lastObsIndex[obsNmbr]]

