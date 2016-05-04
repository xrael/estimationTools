######################################################
# attitudeKinematics
#
# Manuel F. Diaz Ramos
#
# Functions to transform attitude kinematic representations.
######################################################

import numpy as np
from coordinateTransformations import ROT1, ROT2, ROT3
from scipy.integrate import odeint


def getSkewSymmetrixMatrix(vector):
    """
    Compute the skew-symmetric matrix associated to a vector
    :param vector:
    :return:
    """
    v1 = vector[0]
    v2 = vector[1]
    v3 = vector[2]
    vector_tilde = np.array([[0,    -v3,  v2],
                             [v3,    0,  -v1],
                             [-v2,   v1,   0]])

    return vector_tilde

def getVectorFromSkewSymmetricMatrix(skew):
    """
    Returns the vector that parameterizes a skew symmetric matrix.
    :param skew:
    :return:
    """
    return np.array([skew[2,1], skew[0,2], skew[1,0]])

###### DCM
def DCMrate2angVel(dcm, dcm_dot):
    """
    Computes the angular velocity from the DCM and its derivative.
    :param dcm:
    :param dcm_dot:
    :return:
    """
    w_tilde = -dcm_dot.dot(dcm.T)
    return getVectorFromSkewSymmetricMatrix(w_tilde)



###### (3-2-1) Euler angles (yaw-pitch-roll) #########
def dcm2eulerAngles321(DCM):
    """
    Computes the (3-2-1) Euler angles from the DCM.
    :param DCM: [2-dimension numpy array] DCM matrix.
    :return: [1-dimension numpy array] (3-2-1) Euler angles.
    """
    theta_1 = np.arctan2(DCM[0,1], DCM[0,0])
    theta_2 = -np.arcsin(DCM[0,2])
    theta_3 = np.arctan2(DCM[1,2], DCM[2,2])

    return np.array([theta_1, theta_2, theta_3])

def eulerAngles3212dcm(angles321):
    """
    Computes the DCM from the (3-2-1) Euler angles representation.
    :param angles321: [1-dimension numpy array] (3-2-1) Euler angles (rad).
    :return: [2-dimension numpy array] DCM.
    """
    return ROT1(angles321[2]).dot(ROT2(angles321[1]).dot(ROT3(angles321[0])))

def angularVelocity2eulerAngles321rate(eulerang, w):
    """
    Computes the Euler angles (3-2-1) rate from the angles and the angular velocity in body frame.
    :param eulerang: (3-2-1) yaw-pitch-roll Euler angles.
    :param w:
    :return:
    """
    pitch = eulerang[1]
    roll = eulerang[2]

    if np.abs(pitch) > np.deg2rad(90 - 0.1):
        return np.nan

    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)

    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)

    yaw_dot = (sin_roll*w[1] + cos_roll*w[2])/cos_pitch
    pitch_dot = (cos_roll*w[1] - sin_roll*w[2])
    roll_dot = (w[0] + sin_pitch*sin_roll/cos_pitch * w[1] + sin_pitch*cos_roll/cos_pitch * w[2])

    return np.array([yaw_dot, pitch_dot, roll_dot])


def eulerAngles321rate2angularVelocity(eulerang, eulerang_dot):
    """
    Computes the angular velocity in body given the Euler angles (3-2-1) and its rates.
    :param eulerang: (3-2-1) yaw-pitch-roll Euler angles.
    :param eulerang_dot: (3-2-1) yaw-pitch-roll rates.
    :return:
    """
    pitch = eulerang[1]
    roll = eulerang[2]

    yaw_dot = eulerang_dot[0]
    pitch_dot = eulerang_dot[1]
    roll_dot = eulerang_dot[2]

    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)

    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)

    w1 = -sin_pitch*yaw_dot + roll_dot
    w2 = sin_roll*cos_pitch*yaw_dot + cos_roll*pitch_dot
    w3 = cos_roll*cos_pitch*yaw_dot - sin_roll*pitch_dot

    w_B = np.array([w1, w2, w3])

    return w_B

def kinematicEquationEuler321(eulerang, t, getw_callback):
    w = getw_callback(t)
    return angularVelocity2eulerAngles321rate(eulerang,w)


def solveKinematicEquationEuler321(eulerang_initial, getw_callback, t0, tf, dt, a_tol, r_tol):

    num = int((tf - t0)/dt) + 1
    tf = (num - 1) * dt + t0 # includes the last value
    time = np.linspace(t0, tf, num)

    eulerang = odeint(kinematicEquationEuler321, eulerang_initial, time , args= (getw_callback,), atol = a_tol, rtol= r_tol)

    # for angles in eulerang:
    #     print angles
    #     angles[0] = np.mod(angles[0],2*np.pi)
    #     angles[2] = np.mod(angles[2],2*np.pi)

    return (time, eulerang)


############### (3-1-3) Euler angles #####################

def eulerAngles3132dcm(angles313):
    """
    Computes the DCM from the (3-1-3) Euler angles representation.
    :param angles313: [1-dimension numpy array] (3-1-3) Euler angles (rad).
    :return: [2-dimension numpy array] DCM.
    """
    return ROT3(angles313[2]).dot(ROT1(angles313[1]).dot(ROT3(angles313[0])))

############### Principal Rotation Vector #####################
def dcm2prv(DCM):
    """
    Computes the rotation angle (from 0 to 180) and the rotation axis
    given the Direction Cosine Matrix.
    :param DCM: [two-dimension numpy array].
    :return: [tuple] angle from 0 to 180 and rotation axis.
    """
    angle = np.arccos(0.5*(np.trace(DCM) - 1))

    if np.mod(angle, np.pi) == 0:
        axis = np.array([0,0,0])
    else:
        axis = 1.0/(2*np.sin(angle)) * np.array([DCM[1,2] - DCM[2,1],
                                                 DCM[2,0] - DCM[0,2],
                                                 DCM[0,1] - DCM[1,0]])
    return (angle, axis)

def prv2dcm(axis, angle):
    """
    Computes the DCM from the rotation axis and rotation angle.
    :param axis: [1-dimension numpy array] Axis of rotation.
    :param angle: Angle of rotation.
    :return: [2-dimension numpy array] DCM.
    """
    c_phi = np.cos(angle)
    s_phi = np.sin(angle)
    sigma = 1 - c_phi

    e1 = axis[0]
    e2 = axis[1]
    e3 = axis[2]

    C = np.zeros((3,3))

    C[0,0] = e1**2 * sigma + c_phi
    C[1,1] = e2**2 * sigma + c_phi
    C[2,2] = e3**2 * sigma + c_phi

    C[1,0] = e1*e2 * sigma - e3*s_phi
    C[0,1] = e1*e2 * sigma + e3*s_phi

    C[2,0] = e3*e1 * sigma + e2*s_phi
    C[0,2] = e3*e1 * sigma - e2*s_phi

    C[2,1] = e3*e2 * sigma - e1*s_phi
    C[1,2] = e3*e2 * sigma + e1*s_phi

    return C

############### Quaternion #####################
def prv2quat(angle, axis):
    """
    Computes the quaternion given the principal rotation.
    :param angle: [double] In radians.
    :return: [1-dimension numpy array] Quaternion (scalar: first parameter)
    """
    q0 = np.cos(angle/2)
    sin_ang = np.sin(angle/2)
    q1 = axis[0] * sin_ang
    q2 = axis[1] * sin_ang
    q3 = axis[2] * sin_ang

    if q0 < 0:  # Use the short rotation
        q0 = -q0
        q1 = -q1
        q2 = -q2
        q3 = -q3

    return np.array([q0, q1, q2, q3])

def quat2dcm(quat):
    """
    Computes the DCM associated to the quaternion quat.
    :param quat: [1-dimension numpy array] Quaterion.
    :return: [2-dimension numpy array] DCM.
    """
    b_0 = quat[0]
    e = quat[1:]

    dcm = (b_0**2 - np.inner(e,e))*np.eye(3) + 2 * np.outer(e,e) - 2 * b_0 * getSkewSymmetrixMatrix(e)
    return dcm

def dcm2quat(dcm):
    """
    Computes the DCM from the quaternion using Sheppard's method
    :param dcm:
    :return:
    """
    beta_2 = np.zeros(4)
    beta = np.zeros(4)
    dcm_trace = np.trace(dcm)
    beta_2[0] = (1.0 + dcm_trace)/4
    beta_2[1] = (1.0 + 2*dcm[0,0] - dcm_trace)/4
    beta_2[2] = (1.0 + 2*dcm[1,1] - dcm_trace)/4
    beta_2[3] = (1.0 + 2*dcm[2,2] - dcm_trace)/4

    max = np.argmax(beta_2)

    if max == 0:
        beta[0] = np.sqrt(beta_2[0])
        beta[1] = (dcm[1,2]-dcm[2,1])/(4*beta[0])
        beta[2] = (dcm[2,0]-dcm[0,2])/(4*beta[0])
        beta[3] = (dcm[0,1]-dcm[1,0])/(4*beta[0])
    elif max == 1:
        beta[1] = np.sqrt(beta_2[1])
        beta[0] = (dcm[1,2]-dcm[2,1])/(4*beta[1])
        beta[2] = (dcm[0,1]+dcm[1,0])/(4*beta[1])
        beta[3] = (dcm[2,0]+dcm[0,2])/(4*beta[1])
    elif max == 2:
        beta[2] = np.sqrt(beta_2[2])
        beta[0] = (dcm[2,0]-dcm[0,2])/(4*beta[2])
        beta[1] = (dcm[0,1]+dcm[1,0])/(4*beta[2])
        beta[3] = (dcm[1,2]+dcm[2,1])/(4*beta[2])
    else: #max==3
        beta[3] = np.sqrt(beta_2[3])
        beta[0] = (dcm[0,1]-dcm[1,0])/(4*beta[3])
        beta[1] = (dcm[2,0]+dcm[0,2])/(4*beta[3])
        beta[2] = (dcm[1,2]+dcm[2,1])/(4*beta[3])

    # C =np.array([[beta[0]**2+beta[1]**2-beta[2]**2-beta[3]**2, 2*(beta[1]*beta[2]+beta[0]*beta[3]), 2*(beta[1]*beta[3]-beta[0]*beta[2])],
    #           [2*(beta[1]*beta[2]-beta[0]*beta[3]), beta[0]**2-beta[1]**2+beta[2]**2-beta[3]**2, 2*(beta[2]*beta[3]-beta[0]*beta[1])],
    #           [2*(beta[1]*beta[3]+beta[0]*beta[2]), 2*(beta[2]*beta[3]-beta[0]*beta[1]), beta[0]**2-beta[1]**2-beta[2]**2+beta[3]**2]])

    if beta[0] < 0: # use the short rotation
        beta[0] = -beta[0]
        beta[1] = -beta[1]
        beta[2] = -beta[2]
        beta[3] = -beta[3]

    return beta


def angularVelocity2QuaternionRate(quat, w):
    """
    Compute the quaternion rate using the kinematic differential equation for quaternions.
    :param quat: [1-dimensional numpy array] quaternion.
    :param w: [1-dimentsion numpy array] Angular velocity.
    :return: [1-dimentsion numpy array] Quaternion rate.
    """
    quat_rate = np.zeros(4)

    quat_rate[0] = 0.5*(-quat[1] * w[0] - quat[2] * w[1] - quat[3] * w[2])
    quat_rate[1] = 0.5*(quat[0] * w[0] - quat[3] * w[1] + quat[2] * w[2])
    quat_rate[2] = 0.5*(quat[3] * w[0] + quat[0] * w[1] - quat[1] * w[2])
    quat_rate[3] = 0.5*(-quat[2] * w[0] + quat[1] * w[1] + quat[0] * w[2])

    return quat_rate


def quatKinematicEquation(q_k_1, w_k_1, dt):
    """
    Advances a time step dt using the quaternion kinematic differential equation.
    Uses the Euler method of integration.
    :param q_k_1: [1-dimensional numpy array] quaternion at time t_k_1.
    :param w_k_1: [1-dimentsion numpy array] Angular velocity at time t_k_1.
    :param dt: [double] Time step.
    :return: [1-dimentsion numpy array] Quaternion at time t_k.
    """
    q_k = np.zeros(4)

    quat_rate = angularVelocity2QuaternionRate(q_k_1, w_k_1)

    q_k[0] = q_k_1[0] + dt*quat_rate[0]
    q_k[1] = q_k_1[1] + dt*quat_rate[1]
    q_k[2] = q_k_1[2] + dt*quat_rate[2]
    q_k[3] = q_k_1[3] + dt*quat_rate[3]

    q_k = q_k/np.linalg.norm(q_k)

    return q_k


############### Classical Rodrigues Parameters #####################
def prv2crp(angle, axis):
    """
    Computes the CRP given the principal rotation.
    :param angle: [double] In radians.
    :param axis: [1-dimension numpy array] Rotation axis.
    :return: [1-dimension numpy array] CRP.
    """
    tan_ang = np.tan(angle/2)
    crp1 = axis[0] * tan_ang
    crp2 = axis[1] * tan_ang
    crp3 = axis[2] * tan_ang

    return np.array([crp1, crp2, crp3])

def quat2crp(beta):
    """
    Computes the CRP given the quaternion.
    :param beta: [1-dimension numpy array] Quaternion.
    :return: [1-dimension numpy array] CRP.
    """
    beta_0 = beta[0]
    if abs(beta_0) < 1e-6:
        return [0,0,0] # not-defined

    crp1 = beta[1]/beta_0
    crp2 = beta[2]/beta_0
    crp3 = beta[3]/beta_0

    return np.array([crp1, crp2, crp3])

def crp2dcm(crp):
    """
    Computes the DCM associated to a CRP.
    :param crp: [1-dimension numpy array] CRP.
    :return: [2-dimension numpy array] The 3x3 DCM matrix/
    """
    qtq = np.inner(crp, crp)
    qqt = np.outer(crp, crp)
    q_tilde = getSkewSymmetrixMatrix(crp)

    if (qtq + 1) < 1e-6:
        return np.zeros((3,3)) # not defined
    else:
        return 1.0/(1+qtq) * ((1-qtq) * np.eye(3) + 2*qqt - 2*q_tilde)

def cayleyTransformation(matrix):
    """
    Performs the Cayley transformation. Useful to obtain C from Q or Q from C:
    C = (I - Q)*(I + Q)^-1
    Q = (I - C)*(I + C)^-1
    Both transformations are the same.
    :param matrix: [2-dimension numpy array] A orthogonal rotation matrix (C) or a skew-symmetric matrix (Q).
    :return:[2-dimension numpy array] A skew-symmetric matrix (Q) or a orthogonal rotation matrix (C).
    """
    I = np.eye(3)
    return (I - matrix).dot(np.linalg.inv(I + matrix))


############### Modified Rodrigues Parameters #####################
def mrp2dcm(mrp):
    """
    Computes the DCM from the MRP.
    :param dcm:
    :return:
    """
    mrp_tilde = getSkewSymmetrixMatrix(mrp)
    mrp_inner = np.inner(mrp, mrp)

    I = np.eye(3)

    dcm = I + (8*mrp_tilde.dot(mrp_tilde) - 4*(1-mrp_inner)*mrp_tilde)/(1+mrp_inner)**2
    return dcm


def prv2mrp(angle, axis):
    """
    Computes the MRP given the principal rotation.
    :param angle: [double] In radians.
    :param axis: [1-dimension numpy array] Rotation axis.
    :return: [1-dimension numpy array] MRP.
    """
    tan_ang = np.tan(angle/4)
    mrp1 = axis[0] * tan_ang
    mrp2 = axis[1] * tan_ang
    mrp3 = axis[2] * tan_ang

    mrp = switchMRPrepresentation(np.array([mrp1, mrp2, mrp3])) # Use the short rotation

    return mrp

def quat2mrp(beta):
    """
    Computes the mrp given the quaternion.
    :param beta: [1-dimension numpy array] Quaternion.
    :return: [1-dimension numpy array] MRP.
    """
    beta_0 = beta[0]
    if abs(beta_0 + 1) < 1e-6:
        beta = -beta # Switch to the other set
        beta_0 = beta[0]

    mrp1 = beta[1]/(1+beta_0)
    mrp2 = beta[2]/(1+beta_0)
    mrp3 = beta[3]/(1+beta_0)

    mrp = np.array([mrp1, mrp2, mrp3])

    mrp = switchMRPrepresentation(mrp) # use the short rotation

    return mrp

def angVel2mrpRate(mrp, w):
    """
    Computes the MRP rate from the MRP and the angular velocity in B frame.
    :param mrp: [1-dimension numpy array] MRP representing rotation BN (from N to B).
    :param w: [1-dimension numpy array] angular velocity expressed in B frame.
    :return: [1-dimension numpy array] MRP rate.
    """
    mrp_inner = np.inner(mrp, mrp)
    mrp_outer = np.outer(mrp, mrp)
    mrp_tilde = getSkewSymmetrixMatrix(mrp)
    B = ((1 - mrp_inner) * np.eye(3) + 2 * mrp_tilde + 2 * mrp_outer)
    return (0.25*np.dot(B, w))

def mrpRate2AngVel(mrp, mrp_dot):
    """
    Computes the angular velocity from the MRP rates.
    :param mrp:
    :param mrp_dot:
    :return:
    """
    mrp_inner = np.inner(mrp, mrp)
    mrp_outer = np.outer(mrp, mrp)
    mrp_tilde = getSkewSymmetrixMatrix(mrp)
    BT = ((1 - mrp_inner) * np.eye(3) - 2 * mrp_tilde + 2 * mrp_outer)

    return (4.0/(1 + mrp_inner)**2 * BT.dot(mrp_dot))


def switchMRPrepresentation(state):
    """
    Returns the mrp representation that has norm(mrp)<= 1 (Equivalent to a short rotation)
    :param state: Attitude state (it might include velocity).
    :return:
    """
    mrp = state[0:3]
    s = np.inner(mrp, mrp)
    if s > 1:
        state[0] = -state[0]/s
        state[1] = -state[1]/s
        state[2] = -state[2]/s

    return state

def switchMRPcovariance(state, P):
    """
    Computes the covariance switching for Kalman filters using the algorithm on Karlgaard-Schaub paper and
    on "Analytical Mechanics of Space Systems", by Schaub (Appendix I).
    :param state:
    :param P:
    :return:
    """
    mrp = state[0:3]
    mrp_inner = np.inner(mrp, mrp)
    mrp_outer = np.outer(mrp, mrp)

    if mrp_inner > 1:
        lambd = np.eye(state.size)

        lambd[0:3,0:3] = 2*mrp_outer/mrp_inner**2 - np.eye(3)/mrp_inner

        P[:,:] = lambd.dot(P.dot(lambd.T))

    return P



def computeErrorMRP(mrp_BN, mrp_RN):
    """
    Computes the "difference" mrp: mrp_BR.
    :param mrp_BN:
    :param mrp_RN:
    :return:
    """
    mrp_RN2 = np.inner(mrp_RN, mrp_RN)
    mrp_BN2 = np.inner(mrp_BN, mrp_BN)

    den = (1 + mrp_RN2*mrp_BN2 + 2*np.dot(mrp_RN, mrp_BN))

    if den < 1e-2: # Difference of almost 360 deg
        mrp_RN = -mrp_RN/mrp_RN2 # switch one mrp to make the difference close to 0 deg
        mrp_RN2 = np.inner(mrp_RN, mrp_RN)

    return ((1-mrp_RN2) * mrp_BN - (1-mrp_BN2) * mrp_RN + 2 * np.cross(mrp_BN, mrp_RN))/(1 + mrp_RN2*mrp_BN2 + 2*np.dot(mrp_RN, mrp_BN))


def solveKinematicEquationMRP(mrp_0, getw_callback, t0, tf, dt):
    """

    :param mrp_0:
    :param getw_callback:
    :param t0:
    :param tf:
    :param dt:
    :return:
    """
    num = int((tf - t0)/dt) + 1
    tf = (num - 1) * dt + t0 # includes the last value
    time = np.linspace(t0, tf, num)
    l = time.size

    # Enforcing norm(mrp) <= 1
    s = np.inner(mrp_0, mrp_0)
    if s > 1:
        mrp_0 = -mrp_0 / s

    mrp = np.zeros((l, 3))
    mrp[0,:] = mrp_0

    for i in range(0, l-1):
        t_i = time[i]
        w_i = getw_callback(t_i)
        mrp_i = mrp[i,:]
        mrp_ip1 = mrp_i + dt*angVel2mrpRate(mrp_i, w_i) # Euler method

        # Enforcing norm(mrp) <= 1
        s = np.inner(mrp_ip1, mrp_ip1)
        if s > 1:
            mrp_ip1 = -mrp_ip1 / s

        mrp[i+1,:] = mrp_ip1

    return(time, mrp)



