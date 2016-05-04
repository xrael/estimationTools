######################################################
# Integrators
#
# Manuel F. Diaz Ramos
#
# Functions to perform numerical integration.
######################################################

import numpy as np
from enum import Enum

class Integrator(Enum):
    ODEINT = 1
    RK4 = 2

def rk4Step(func, t_i_1, t_i, state_i_1, params, event = None):
    """
    Advances time from t_i_1 to t_i numerically integrating a first order sistem X_dot = F(X, t).
    :param func: [Pointer to function] Function F(X,t)
    :param t_i_1:
    :param t_i:
    :param state_i_1:
    :param params:
    :param event:
    :return:
    """

    dt = t_i - t_i_1
    # RK4

    # k1
    k1 = func(state_i_1, t_i_1, *params)

    # k2
    t_k23 = t_i_1 + 0.5 * dt
    state_k2 = state_i_1 + 0.5 * k1 * dt
    k2 = func(state_k2, t_k23, *params)

    # k3
    state_k3 = state_i_1 + 0.5 * k2 * dt
    k3 = func(state_k3, t_k23, *params)

    # k4
    t_k4 = t_i_1 + dt
    state_k4 = state_i_1 + k3 * dt
    k4 = func(state_k4, t_k4, *params)

    state_i = state_i_1 + 1.0/6.0 * (k1 + 2*k2 + 2*k3 + k4) * dt

    if event != None: # Apply corrections
        event(state_i)

    return state_i

def rk4Integrator(func, X0, time_vec, args = (), event = None):
    """
    RK-4 fixed-step integrator of a first order ODE: X_dot = F(X,t)
    :param func: [Pointer to function] Function F(X,t)
    :param X0: [1-dimensional numpy array] Initial state.
    :param time_vec: [1-dimensional numpy array] Vector with the times where X will be computed.
    :param args: [tuple] Optional arguments.
    :param event: [Pointer to function] Function to process the state after advancing one time step.
    :return:
    """

    l = time_vec.size

    state_vec = np.zeros((l, X0.size))
    state_vec[0,:] = X0

    for i in range(0, l-1):
        state_vec[i+1]= rk4Step(func, time_vec[i], time_vec[i+1], state_vec[i], args, event)

    return state_vec