

from dynamicSimulator import dynamicSimulator
import numpy as np

class observationSimulator:
    """
    Class that simulates observations for a given dynamics and observation models.
    It can be used with additive noise.
    """

    ## Constructor: DO NOT USE IT!
    def __init__(self):

        self._obsModel = None
        self._dynSimulator = None

        self._observations_time_vec = []
        self._observers = []
        self._observations = []
        self._observed_states = []

        self._noiseFlag = False
        self._noiseMean = 0
        self._noiseCovariance = 0
        return

    @classmethod
    def getObservationSimulator(cls, dynModel, obsModel):
        """
        Factory method used to get an instance of the class.
        :param dynModel: Interface dynModel.
        :return: An instance of this class.
        """
        obsSim = observationSimulator()

        obsSim._obsModel = obsModel
        obsSim._dynSimulator = dynamicSimulator.getDynamicSimulator(dynModel)

        obsSim._noiseFlag = False
        obsSim._noiseMean = 0
        obsSim._noiseCovariance = 0

        return obsSim

    def addNoise(self, flag, noise_mean, noise_covariance):
        """
        Use it to add gaussian noise to the measurements.
        :param flag:
        :param noise_mean:
        :param noise_covariance:
        :return:
        """
        nmbrObs = self._obsModel.getNmbrOutputs()

        if flag == True and noise_mean.size == nmbrObs and noise_covariance.shape == (nmbrObs, nmbrObs):
            self._noiseFlag = True
            self._noiseMean = noise_mean
            self._noiseCovariance = noise_covariance
            ret = True
        else:
            self._noiseFlag = False
            self._noiseMean = 0
            self._noiseCovariance = 0
            ret = False

        return ret

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

    def simulate(self, initialState, dynamic_params, t0, tf, dt, rtol, atol):

        (timeVec, statesVec) = self._dynSimulator.propagate(initialState, dynamic_params, t0, tf, dt, rtol, atol)

        nmbr_coordinates = self._obsModel.getNmbrObservers()

        observed_time_vec = []
        observers = []
        observations = []
        observed_states = []

        for i in range(0, timeVec.size):
            for j in range(0, nmbr_coordinates):
                params = (j,)
                if self._obsModel.isObservable(statesVec[i], timeVec[i], params):
                    obs = self._obsModel.computeModel(statesVec[i], timeVec[i], params)
                    if self._noiseFlag:
                        obs += np.random.multivariate_normal(self._noiseMean, self._noiseCovariance)

                    observed_time_vec.append(timeVec[i])
                    observers.append(j)
                    observations.append(obs)
                    observed_states.append(statesVec[i])

        self._observations_time_vec = np.array(observed_time_vec)
        self._observers = np.array(observers)
        self._observations = np.array(observations)
        self._observed_states = observed_states

        return (self._observations_time_vec, self._observers, self._observations, self._observed_states)
