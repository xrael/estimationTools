######################################################
# Batch Processor
#
# Manuel F. Diaz Ramos
#
# This class implements a batch processor.
# It relies on:
# 1) A dynamical model which implements
# the interface dynamicModelBase.
# 2) An observation model which implements
# the interface observerModelBase
######################################################

import numpy as np
import dynamicSimulator as dynSim

class batchProc :
    
    def __init__(self):
        self._dynModel = None
        self._obsModel = None
        self._dynSim = None

        self._obsVector = None
        self._obsTime = None
        self._obsParams = None
        self._nmbrObs = 0
        
        self._Xref_0 = None
        
        self._xbar_0 = None
        self._Pbar_0 = None
        
        self._stm0 = None
        #self._stms = None
        
        self._R = None
        
        self._normalMatrix = None
        self._infoMatrix = None
        self._Xhat_0 = None
        self._xhat_0 = None
        self._prefit_residuals = None
        self._postfit_residuals = None
        self._nonLinearEstimates = None
        self._deviationEstimates = None
        self._covarianceMatrices = None
        #self._Xrefs = None
        
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
        proc = batchProc()

        obsModel.defineSymbolicState(dynModel.getSymbolicState()) # Redefines the state of the observer
        
        proc._dynModel = dynModel
        proc._obsModel = obsModel

        proc._dynSim = dynSim.dynamicSimulator.getDynamicSimulator(dynModel)
        
        return proc
     
    # Before computing the batch solution, call this method
    def configureFilter(self, obs_vector, obs_time, obs_params, Xref_0, xbar_0, Pbar_0, R, iterations):
        """
        Before computing the batch solution, call this method.
        :param obs_vector: [2-dimensional numpy array] Observations. rows: observations at time t_i. Columns: observations of a given type.
        :param obs_time: [1-dimensional numpy array] Time of observations.
        :param obs_params: [1-dimensional numpy array] Additional data at time t_i. Example: Ground Station coordinates.
        :param Xref_0: [1-dimensional numpy array] Initial guess of the state.
        :param xbar_0: [1-dimensional numpy array] Deviation from the initial guess (usually 0).
        :param Pbar_0: [2-dimensional numpy array] A-priori covariance.
        :param iterations: [int] Number of iterations to be carried out by the filter.
        :return:
        """
        self._obsVector = obs_vector
        self._obsTime = obs_time
        self._obsParams = obs_params
        self._nmbrObs = np.size(obs_time)
        
        self._Xref_0 = Xref_0
        
        self._xbar_0 = xbar_0
        self._Pbar_0 = Pbar_0
        
        self._stm0 = np.eye(self._dynModel.getNmbrOfStates())
        
        self._R = R
        
        self._iteration = 0
        self._nmbrIterations = iterations
        
        # For the following, there's one element per iteration
        self._normalMatrix = []
        self._infoMatrix = []
        self._Xhat_0 = []
        self._xhat_0 = []
        self._prefit_residuals = []
        self._postfit_residuals = []
        self._nonLinearEstimates = []
        self._deviationEstimates = []
        self._covarianceMatrices = []
        #self._stms = []
        #self._Xrefs = []

        return
         
    def computeEstimate(self, t0, tf, dt, rel_tol, abs_tol):
        
        nmbrStates = self._dynModel.getNmbrOfStates()
        nmbrObs = self._nmbrObs
        
        Xref_0 = self._Xref_0            # Non-linear reference
        xbar_0 = self._xbar_0            # Linear a priori value

        params = ()

        Pbar_0_inv = self.invert(self._Pbar_0) # Invert the way you want
        R_inv = self.invert(self._R)
        
        while self.iterateBatch() :            # Iterate the way you want
            H = np.zeros((nmbrObs, np.size(self._obsVector[0]), nmbrStates))
            y = np.zeros((nmbrObs, np.size(self._obsVector[0])))

            stms_from_t0 = np.zeros((nmbrObs, nmbrStates, nmbrStates)) # all STMs from t0 to the current time
            Xrefs = np.zeros((nmbrObs, nmbrStates))
        
            initial_t = t0
            initial_Xref = Xref_0
            initial_stm = self._stm0
            
            info_matrix = Pbar_0_inv
            normal_matrix = Pbar_0_inv.dot(xbar_0)

            for i in range(0, nmbrObs): # Iteration for every observation
                ti = self._obsTime[i]
                
                if ti == t0:
                    stm = self._stm0
                    Xref = Xref_0
                else:
                    (states, stms, time, Xref, stm)  = self._dynSim.propagateWithSTM(initial_Xref, initial_stm, params,
                                                                                  initial_t, dt, ti, rel_tol, abs_tol)

                obP = (self._obsParams[i],)
                H_tilde_i = self._obsModel.computeJacobian(Xref, ti, obP)
                Y_i = self._obsVector[i]
                y[i, :] = Y_i - self._obsModel.computeModel(Xref, ti, obP)
                H[i, :, :] = H_tilde_i.dot(stm)
                info_matrix = info_matrix + ((H[i].T).dot(R_inv)).dot(H[i])
                normal_matrix = normal_matrix + ((H[i].T).dot(R_inv)).dot(y[i])
                
                initial_t = ti
                initial_Xref = Xref
                initial_stm = stm

                stms_from_t0[i,:,:] = stm
                Xrefs[i,:] = Xref # Reference trajectory for t_i
            # End observation processing
                
            # Solve the normal equation
            xhat_0 = self.solveNormalEq(info_matrix, normal_matrix)
            
            Xref_0 = Xref_0 + xhat_0 # Updating initial guess
            
            xbar_0 = xbar_0 - xhat_0
            
            self._normalMatrix.append(normal_matrix)
            self._infoMatrix.append(info_matrix)
            self._Xhat_0.append(Xref_0)
            self._xhat_0.append(xhat_0)
            self._prefit_residuals.append(y)
            #self._stms.append(stms_from_t0)
            #self._Xrefs.append(Xrefs)
            
            postfit_residuals = np.zeros((nmbrObs, np.shape(y)[1]))
            Xhat = np.zeros((nmbrObs, nmbrStates))
            xhat = np.zeros((nmbrObs, nmbrStates))
            P = np.zeros((nmbrObs, nmbrStates, nmbrStates))
            P_0 = self.invert(info_matrix)
            for i in range(0, nmbrObs):
                postfit_residuals[i,:] = y[i,:] - H[i,:,:].dot(xhat_0)
                xhat[i,:] = stms_from_t0[i].dot(xhat_0)
                Xhat[i,:] = Xrefs[i] + xhat[i]
                P[i,:,:] = stms_from_t0[i].dot(P_0).dot(stms_from_t0[i].T)

            self._postfit_residuals.append(postfit_residuals)
            self._nonLinearEstimates.append(Xhat)
            self._deviationEstimates.append(xhat)
            self._covarianceMatrices.append(P)
        # End Iteration        
        
        return

    # def propagateState(self):
    #     stms = self._stms[-1] # Last iteration STM
    #     Xrefs = self._Xrefs[-1]
    #     P_0 = self.getCovariance()[-1]
    #     xhat_0 = self._xhat_0[-1]
    #
    #     Xref_shape = np.shape(Xrefs)
    #     nmbrObs = Xref_shape[0]
    #     nmbrStates = Xref_shape[1]
    #
    #     Xhat = np.zeros((nmbrObs, nmbrStates))
    #     P = np.zeros((nmbrObs, nmbrStates, nmbrStates))
    #
    #     for i in range(0, nmbrObs):
    #         Xhat[i,:] = Xrefs[i] + stms[i].dot(xhat_0)
    #         P[i,:,:] = stms[i].dot(P_0).dot(stms[i].T)
    #
    #     return  (Xhat, P)

    # The following getters should be used after calling computeEstimate()       
    def getInformationMatrix(self) :
        return self._infoMatrix
        
    def getNormalMatrix(self) :
        return self._normalMatrix
    
    def getNonLinearEstimate(self) :
        return self._Xhat_0
        
    def getDeviationEstimate(self) :
        return self._xhat_0
        
    def getPreFitResiduals(self):
        return self._prefit_residuals

    def getPostFitResiduals(self):
        return self._postfit_residuals

    def getNonLinearEstimates(self):
        return  self._nonLinearEstimates

    def getDeviationEstimates(self):
        return self._deviationEstimates

    def getCovarianceMatrices(self):
        return self._covarianceMatrices
    
    def getCovariance(self):
        covariance = []
        for inf in self._infoMatrix:
            covariance.append(self.invert(inf))
        return covariance
        
    # def getSTMsFromInitialTime(self):
    #     return self._stms
        
    def solveNormalEq(self, infoMat, normalMat) :
        #return np.linalg.inv(infoMat).dot(normalMat)
        return self.choleskyAlgorithm(infoMat, normalMat)

    def iterateBatch(self) :
        """
        This method defines the way the batch filter is going to iterate.
        Modify it if another iteration algorithm is desired.
        :return:
        """
        if self._iteration < self._nmbrIterations:
            self._iteration = self._iteration + 1
            return True
        else:
            return False  

    def invert(self, matrix):
        """
        Inverts a matrix using a given method: numerical inversion or Cholesky inversion.
        :param matrix:
        :return:
        """
        if np.size(matrix) == 1:
            return 1.0/matrix
        else:
            #return np.linalg.inv(matrix)
            return self.choleskyInversion(matrix)

    def choleskyAlgorithm(self, A, b) :
        """
        Solves a system A*x = b where A is positive-definite using the Cholesky algorithm.
        :param A:  [2-dimensional numpy array] Matrix to invert.
        :param b:  [1-dimensional numpy array] Vector.
        :return: Solution of the linear system.
        """
        L = np.linalg.cholesky(A) # Lower triangular
        R = L.T
        
        nmbrStates = np.size(b)
        
        z = np.zeros(nmbrStates)
        for i in range(0, nmbrStates):
            aux = 0
            for j in range(0, i):
                aux += R[j,i] * z[j]
            z[i] = (b[i] - aux)/R[i,i]
            
        x = np.zeros(nmbrStates)
        for i in range(nmbrStates-1, -1, -1):
            aux = 0
            for j in range(i, nmbrStates):
                aux += R[i,j] * x[j]
            x[i] = (z[i] - aux)/R[i,i]
        
        return x

    def choleskyInversion(self, mat):
        """
        Implements the Cholesky inversion of a positive-definite matrix.
        :param mat: [2-dimensional numpy array] Matrix to invert.
        :return:
        """
        L = np.linalg.cholesky(mat) # Lower triangular
        R = L.T
        
        dim = mat.shape[0] # Square-matrix assumed

        S = np.zeros([dim, dim])        
        for i in range(0, dim):
            S[i,i] = 1.0/R[i,i]
            
            for j in range(i+1, dim):
                aux = 0
                for k in range(i, j):
                    aux -= S[i,k]*R[k,j]
                S[i,j] = aux/R[j,j]
                
        return S.dot(S.T)
                    
            