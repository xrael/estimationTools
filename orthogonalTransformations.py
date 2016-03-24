######################################################
# orthogonalTransformations
#
# Manuel F. Diaz Ramos
#
# - Functions to perform orthogonal transformations into a matrix.
# Givens, Householder.
# - Whitening Transformation.
# - Additional function to invert matrices and solve linear systems:
# Cholesky, backwards and forward substitution.
######################################################

import numpy as np

def whiteningTransformation(P):
    """
    If P is a covariance matrix associated to a vector e, then: P = L L^T.
    L^-1*e has a covariance I. R = L^-1
    :param P: [2-dimensional numpy array] Covariance matrix.
    :return: R = L^-1
    """
    L = np.linalg.cholesky(P) # P = L * L^T
    R = backwardsSubstitutionInversion(L) #P^-1 = L^-T * L^-1 = R^T* R

    return R

def choleskyAlgorithm(A, b) :
    """
    Solves a system A*x = b where A is positive-definite using the Cholesky algorithm.
    :param A:  [2-dimensional numpy array] Matrix to invert.
    :param b:  [1-dimensional numpy array] Vector.
    :return: Solution of the linear system.
    """
    L = np.linalg.cholesky(A) # Lower triangular
    R = L.T
    z = forwardSubstitution(L, b)
    x = backwardsSubstitution(R, z)

    return x

def backwardsSubstitution(R, b):
    """

    :param A:
    :param b:
    :return:
    """
    n = b.size
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        aux = 0
        for j in range(i, n):
            aux += R[i,j] * x[j]
        x[i] = (b[i] - aux)/R[i,i]

    return x

def forwardSubstitution(L, b):
    """

    :param L:
    :param b:
    :return:
    """
    n = b.size
    z = np.zeros(b)
    for i in range(0, n):
        aux = 0
        for j in range(0, i):
            aux += L[i,j] * z[j]
        z[i] = (b[i] - aux)/L[i,i]

    return z

def choleskyInversion(mat):
    """
    Implements the Cholesky inversion of a positive-definite matrix.
    :param mat: [2-dimensional numpy array] Matrix to invert.
    :return:
    """
    L = np.linalg.cholesky(mat) # Lower triangular
    R = L.T
    S = forwardSubstitutionInversion(R)
    return S.dot(S.T)

def forwardSubstitutionInversion(U):
    """
    For an upper triangular matrix L,
    this function computes the inverse S using forward substitution
    by analyzing the product S*U=I
    :param L: [2-dimensional numpy array] Upper triangular matrix.
    :return: [2-dimensional numpy array] Inverse (also upper triangular).
    """
    n = np.shape(U)[0]
    S = np.zeros((n,n))
    for i in range(0, n):
        S[i,i] = 1.0/U[i,i]
        for j in range(i+1, n):
            aux = 0.0
            for k in range(i, j):
                aux -= S[i,k] * U[k,j]
            S[i,j] = aux/U[j,j]

    return S

def backwardsSubstitutionInversion(L):
    """
    For a lower triangular matrix L,
    this functions computes the inverse S using backwards substitution
    by analizyng the product S*L=I
    :param L: [2-dimensional numpy array] Lower triangular matrix.
    :return: [2-dimensional numpy array] Inverse (also lower triangular).
    """
    n = np.shape(L)[0]
    S = np.zeros((n,n))
    for i in range(n-1,-1,-1):
        # Starts at the last S row: S[n-1,:]
        # i is the number of row in S
        S[i,i] = 1.0 / L[i,i]
        for j in range(i-1,-1,-1):
            # j is the number of column in L
            aux = 0.0
            for k in range(j+1, n):
                aux -= S[i,k] * L[k,j]
            S[i,j] = aux/L[j,j]

    return S


def householderTransformation(A):
    """
    Performs the orthogonal transformation using Househoulder algorithm.
    A is a (n+m)x(n+1) matrix, where
    n: number of states.
    m: number of observations (pxl).
    p: observations per observation vector.
    l: number of observation vectors.
    :param A: [2-dimensional numpy arra] Matrix to be orthogonalized.
    :return: [2-dimensional numpy arra] Matrix orthogonalized.
    """

    A_dim = np.shape(A)
    nmbrRows = A_dim[0]
    n = A_dim[1] - 1

    u = np.zeros(nmbrRows)
    #A_new = np.zeros((nmbrRows, n+1))

    for k in range(0, n):
        # Zeroing elements in column k
        col_k = np.copy(A[k:,k]) #column that will be zeroed (from row k to the end)
        A_kk = A[k,k]
        sigma = np.sign(A_kk) * np.linalg.norm(col_k)
        u[k] = A_kk + sigma
        #A[k,k] = -sigma
        u[(k+1):] = col_k[1:]
        beta = 1.0/(sigma*u[k])
        for j in range(k+1, n+1):
            # Multiplying the following columns by the orthogonal matrix
            gamma = beta * u[k:].dot(A[k:,j])
            A[k:,j] = A[k:,j] - gamma * u[k:]

        A[k,k] = -sigma
        A[(k+1):,k] = 0 # the column is zeroed

    return A
