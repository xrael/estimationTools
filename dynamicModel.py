######################################################
# ASEN 5070: Statistical Orbit Determination I
# Dynamic Model
#
# Manuel F. Diaz Ramos
#
# This class could be interpreted as an implementation of a
# a DynModel Interface with the following methods:
# getF()
# getA()
# getNmbrOfStates()
# getParams()
# getProcessSTM()
######################################################

import sympy as sp
import numpy as np

# Function passed to the propagator.
# Contains the dynamic model.
def dynamic_model(X, t, params):

    dynModel = params[0]

    dX = dynModel.getF(X, t, params[1:])

    A = dynModel.getA(X, t, params[1:])

    nmbrStates = dynModel.getNmbrOfStates()

    #dX = dX[0:6]

    for i in range(0, nmbrStates):  # loop for each phi vector
        phi = X[(nmbrStates + nmbrStates*i):(nmbrStates + nmbrStates + nmbrStates*i):1]
        dphi = A.dot(phi)
        dX = np.concatenate([dX, dphi])

    return dX

# The class dynModel computes symbollically and numerically
# the dynamic model and its derivatives.
# Use the factory method to construct an instance.
# Use getF() and getA() to get numerical values.
# getA() depends on how the state is defined.
class dynModel:

    # Constructors and factory methods--------------------
    # Constructor: do not use it. Use the factory methods instead
    def __init__(self):
        self._gravity = None
        self._gravityLambdas = None
        self._potential = None
        self._potentialLambda = None
        self._accelModel = None
        self._accelLambdas = None

        self._A_pos = None
        self._Alambda_pos = None

        self._A_vel = None
        self._Alambda_vel = None

        self._A_params = None
        self._Alambda_params = None

        self._include_J2 = False
        self._include_drag = False

        self._params = None

        return

    # Factory method: Use this to get a two-body model
    @classmethod
    def getTwoBodyModel(cls, mu):
        mod = dynModel()

        mod._include_J2 = False
        mod._include_drag = False
        mod._getAccelModel()
        mod._getModelDerivative()
        mod._getAccelDerivFromPos()
        mod._getAccelDerivFromVel()
        mod._getAccelDerivFromParams()

        mod._params = (mu,)

        return mod

    # Factory method: Use this to get a two-body + J2 model
    @classmethod
    def getJ2Model(cls, mu, J2, Req):
        mod = dynModel()

        mod._include_J2 = True
        mod._include_drag = False
        mod._getAccelModel()
        mod._getAccelDerivFromPos()
        mod._getAccelDerivFromVel()
        mod._getAccelDerivFromParams()

        mod._params = (mu, J2, Req)

        return mod

    # Factory method: Use this to get a two-body + J2 + Drag model
    @classmethod
    def getJ2DragModel(cls, mu, Req, J2, CD_drag, A_drag, mass_sat, rho_0_drag,
                            r0_drag, H_drag, theta_dot):
        mod = dynModel()

        mod._include_J2 = True
        mod._include_drag = True
        mod._getAccelModel()
        mod._getAccelDerivFromPos()
        mod._getAccelDerivFromVel()
        mod._getAccelDerivFromParams()

        mod._params = (mu, Req, J2, CD_drag, A_drag, mass_sat, rho_0_drag,
                            r0_drag, H_drag, theta_dot)

        return mod

    # Private methods--------------------------------------

    # Generates the Lambda functions of the model
    def _getAccelModel(self) :
        x, y, z = sp.symbols('x y z')
        r = sp.sqrt(x**2 + y**2 + z**2)

        mu = sp.symbols('mu')

        U_pm = mu/r # Point mass potential

        self._potential = U_pm
        du_x = sp.diff(U_pm,x).simplify()
        du_y = sp.diff(U_pm,y).simplify()
        du_z = sp.diff(U_pm,z).simplify()

        # Gravity models
        if self._include_J2 == True:
            R_E, J2 = sp.symbols('R_E J2')
            U_J2 = - mu/r * J2 * (R_E/r)**2 * (sp.Rational(3,2) * (z/r)**2 - sp.Rational(1,2))
            self._potential = self._potential + U_J2
            du_x = du_x + sp.diff(U_J2,x).simplify()
            du_y = du_y + sp.diff(U_J2,y).simplify()
            du_z = du_z + sp.diff(U_J2,z).simplify()

        self._gravity = [du_x, du_y, du_z]
        self._accelModel = [du_x, du_y, du_z]

        # Perturbation models
        if self._include_drag == True:
            x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')
            C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag = sp.symbols('C_D_drag A_drag m_drag rho_0_drag r_0_drag H_drag theta_dot_drag')

            Va = sp.sqrt((x_dot + theta_dot_drag * y)**2 + (y_dot - theta_dot_drag * x)**2 + z_dot**2)

            rho_A_drag = rho_0_drag*sp.exp(-(r-r_0_drag)/H_drag)
            aux = -sp.Rational(1,2) * C_D_drag * A_drag/m_drag  * rho_A_drag * Va

            drag_acc1 = aux * (x_dot + theta_dot_drag * y)
            drag_acc2 = aux * (y_dot - theta_dot_drag * x)
            drag_acc3 = aux * (z_dot)

            self._accelModel[0] = self._accelModel[0] + drag_acc1
            self._accelModel[1] = self._accelModel[1] + drag_acc2
            self._accelModel[2] = self._accelModel[2] + drag_acc3

        g1 = None
        g2 = None
        g3 = None
        U = None
        a1 = None
        a2 = None
        a3 = None
        if self._include_J2 == False and self._include_drag == False :
            g1 = sp.lambdify((x, y, z, mu), self._gravity[0], "numpy")
            g2 = sp.lambdify((x, y, z, mu), self._gravity[1], "numpy")
            g3 = sp.lambdify((x, y, z, mu), self._gravity[2], "numpy")

            U = sp.lambdify((x, y, z, mu), self._potential, "numpy")

            a1 = sp.lambdify((x, y, z, mu), self._accelModel[0], "numpy")
            a2 = sp.lambdify((x, y, z, mu), self._accelModel[1], "numpy")
            a3 = sp.lambdify((x, y, z, mu), self._accelModel[2], "numpy")
        elif self._include_J2 == True and self._include_drag == False :
            g1 = sp.lambdify((x, y, z, mu, R_E, J2), self._gravity[0], "numpy")
            g2 = sp.lambdify((x, y, z, mu, R_E, J2), self._gravity[1], "numpy")
            g3 = sp.lambdify((x, y, z, mu, R_E, J2), self._gravity[2], "numpy")

            U = sp.lambdify((x, y, z, mu, R_E, J2), self._potential, "numpy")

            a1 = sp.lambdify((x, y, z, mu, R_E, J2), self._accelModel[0], "numpy")
            a2 = sp.lambdify((x, y, z, mu, R_E, J2), self._accelModel[1], "numpy")
            a3 = sp.lambdify((x, y, z, mu, R_E, J2), self._accelModel[2], "numpy")
        elif self._include_J2 == False and self._include_drag == True :
            g1 = sp.lambdify((x, y, z, mu), self._gravity[0], "numpy")
            g2 = sp.lambdify((x, y, z, mu), self._gravity[1], "numpy")
            g3 = sp.lambdify((x, y, z, mu), self._gravity[2], "numpy")

            U = sp.lambdify((x, y, z, mu), self._potential, "numpy")

            a1 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), self._accelModel[0], "numpy")
            a2 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), self._accelModel[1], "numpy")
            a3 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), self._accelModel[2], "numpy")
        else : # include_J2 == True and include_drag == True
            g1 = sp.lambdify((x, y, z, mu, R_E, J2), self._gravity[0], "numpy")
            g2 = sp.lambdify((x, y, z, mu, R_E, J2), self._gravity[1], "numpy")
            g3 = sp.lambdify((x, y, z, mu, R_E, J2), self._gravity[2], "numpy")

            U = sp.lambdify((x, y, z, mu, R_E, J2), self._potential, "numpy")

            a1 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), self._accelModel[0], "numpy")
            a2 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), self._accelModel[1], "numpy")
            a3 = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), self._accelModel[2], "numpy")

        self._accelLambdas = [a1, a2, a3]

        self._gravityLambdas = [g1, g2, g3]
        self._potentialLambda = U

        return self._accelLambdas

    # Returns the Jacobian matrix of the acceleration model
    # with respect to position (a 3x3 matrix)
    def _getAccelDerivFromPos(self) :

        acc = self._accelModel

        x, y, z = sp.symbols('x, y, z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        mu = sp.symbols('mu')
        R_E, J2 = sp.symbols('R_E J2')
        C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag = sp.symbols('C_D_drag A_drag m_drag rho_0_drag r_0_drag H_drag theta_dot_drag')

        dF = [[0 for i in range(3)] for i in range(3)]
        A_lambda = [[0 for i in range(3)] for i in range(3)]

        for i in range(0,3) :
            dF[i][0] = sp.diff(acc[i],x)
            dF[i][1] = sp.diff(acc[i],y)
            dF[i][2] = sp.diff(acc[i],z)

        for i in range(0, 3) :
            for j in range(0, 3) :
                if self._include_J2 == False and self._include_drag == False :
                    A_lambda[i][j] = sp.lambdify((x, y, z, mu), dF[i][j], "numpy")
                elif self._include_J2 == True and self._include_drag == False :
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2), dF[i][j], "numpy")
                elif self._include_J2 == False and self._include_drag == True :
                    A_lambda[i][j] =  sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), dF[i][j], "numpy")
                else : # include_J2 == True and include_drag == True
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), dF[i][j], "numpy")

        self._A_pos = dF
        self._Alambda_pos = A_lambda

        return

    # Returns the Jacobian matrix of the acceleration model
    # with respect to velocity (a 3x3 matrix)
    def _getAccelDerivFromVel(self) :

        acc = self._accelModel

        x, y, z = sp.symbols('x, y, z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        mu = sp.symbols('mu')
        R_E, J2 = sp.symbols('R_E J2')
        C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag = sp.symbols('C_D_drag A_drag m_drag rho_0_drag r_0_drag H_drag theta_dot_drag')

        dF = [[0 for i in range(3)] for i in range(3)]
        A_lambda = [[0 for i in range(3)] for i in range(3)]

        for i in range(0,3) :
            dF[i][0] = sp.diff(acc[i],x_dot)
            dF[i][1] = sp.diff(acc[i],y_dot)
            dF[i][2] = sp.diff(acc[i],z_dot)

        for i in range(0, 3) :
            for j in range(0, 3) :
                if self._include_J2 == False and self._include_drag == False :
                    A_lambda[i][j] = sp.lambdify((x, y, z, mu), dF[i][j], "numpy") # Should be 0
                elif self._include_J2 == True and self._include_drag == False :
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2), dF[i][j], "numpy") # Should be 0
                elif self._include_J2 == False and self._include_drag == True :
                    A_lambda[i][j] =  sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), dF[i][j], "numpy")
                else : # include_J2 == True and include_drag == True
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), dF[i][j], "numpy")

        self._A_vel = dF
        self._Alambda_vel = A_lambda

        return

    # Returns the Jacobian matrix of the acceleration model
    # with respect to Parameters (a 3x3 matrix)
    def _getAccelDerivFromParams(self) :

        acc = self._accelModel

        x, y, z = sp.symbols('x, y, z')
        x_dot, y_dot, z_dot = sp.symbols('x_dot y_dot z_dot')

        mu = sp.symbols('mu')
        R_E, J2 = sp.symbols('R_E J2')
        C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag = sp.symbols('C_D_drag A_drag m_drag rho_0_drag r_0_drag H_drag theta_dot_drag')

        dF = [[0 for i in range(3)] for i in range(3)]
        A_lambda = [[0 for i in range(3)] for i in range(3)]

        for i in range(0,3) :
            dF[i][0] = sp.diff(acc[i],mu)
            dF[i][1] = sp.diff(acc[i],J2)
            dF[i][2] = sp.diff(acc[i],C_D_drag)

        for i in range(0, 3) :
            for j in range(0, 3) :
                if self._include_J2 == False and self._include_drag == False :
                    A_lambda[i][j] = sp.lambdify((x, y, z, mu), dF[i][j], "numpy")
                elif self._include_J2 == True and self._include_drag == False :
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2), dF[i][j], "numpy")
                elif self._include_J2 == False and self._include_drag == True :
                    A_lambda[i][j] =  sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), dF[i][j], "numpy")
                else : # include_J2 == True and include_drag == True
                    A_lambda[i][j] = sp.lambdify((x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag), dF[i][j], "numpy")

        self._A_params = dF
        self._Alambda_params = A_lambda

        return

    # Public Interface--------------------------------------
    # This is the important interface por the estimation processors!

    # Number of States
    def getNmbrOfStates(self) :
        return 18

    def getParams(self) :
        return self._params

    # Computes the dynamic model function F: X_dot = F(X,t)
    def getF(self, X, t, params) :
        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]
        mu = X[6] #params[0]
        R_E = params[1]
        J2 = X[7] #params[2]
        C_D_drag = X[8] #params[3]
        A_drag = params[4]
        m_drag = params[5]
        rho_0_drag = params[6]
        r_0_drag = params[7]
        H_drag = params[8]
        theta_dot_drag = params[9]

        F = np.array([x_dot,
                      y_dot,
                      z_dot,
                      self._accelLambdas[0](x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag),
                      self._accelLambdas[1](x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag),
                      self._accelLambdas[2](x, y, z, x_dot, y_dot, z_dot, mu, R_E, J2, C_D_drag, A_drag, m_drag, rho_0_drag, r_0_drag, H_drag, theta_dot_drag),
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0])

        return F

    # Returns A matrix
    # This is application-specific. It depends on how state vector is defined.
    def getA(self, X, t, params):

        # state: [x, y, z, x_dot, y_dot, z_dot, mu, J2, CD, x_gs1, y_gs1, z_gs1,
        #         x_gs2, y_gs2, z_gs2, x_gs3, y_gs3, z_gs3]

        x = X[0]
        y = X[1]
        z = X[2]
        x_dot = X[3]
        y_dot = X[4]
        z_dot = X[5]
        mu = X[6] #params[0]
        R_E = params[1]
        J2 = X[7] #params[2]
        C_D_drag = X[8] #params[3]
        A_drag = params[4]
        m_drag = params[5]
        rho_0_drag = params[6]
        r_0_drag = params[7]
        H_drag = params[8]
        theta_dot_drag = params[9]

        A = np.zeros([18,18])

        for i in range(0,3):
            for j in range(3,6):
                if i == j-3:
                    A[i][j] = 1

        for i in range(3,6):
            for j in range(0,3):
                A[i][j] = self._Alambda_pos[i-3][j](x, y, z, x_dot, y_dot, z_dot,
                                mu, R_E, J2, C_D_drag, A_drag, m_drag,
                                rho_0_drag, r_0_drag, H_drag, theta_dot_drag)


        for i in range(3,6):
            for j in range(3,6):
                A[i][j] = self._Alambda_vel[i-3][j-3](x, y, z, x_dot, y_dot, z_dot,
                                mu, R_E, J2, C_D_drag, A_drag, m_drag,
                                rho_0_drag, r_0_drag, H_drag, theta_dot_drag)

        for i in range(3,6):
            for j in range(6,9):
                A[i][j] = self._Alambda_params[i-3][j-6](x, y, z, x_dot, y_dot, z_dot,
                                mu, R_E, J2, C_D_drag, A_drag, m_drag,
                                rho_0_drag, r_0_drag, H_drag, theta_dot_drag)

        return A

    # Obtains the Process Noise Transition matrix model
    def getProcessSTM(self, t_i_1, t_i):
        delta_t = t_i - t_i_1

        # Process Noise Transition Matrix with constant velocity approximation
        pnstm = np.zeros((np.shape(self.getNmbrOfStates())[0], 3))
        pnstm[0:3, :] = delta_t**2/2 * np.eye(3)
        pnstm[3:6, :] = delta_t * np.eye(3)


        return pnstm