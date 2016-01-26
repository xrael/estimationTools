######################################################
# ASEN 6080: Statistical Orbit Determination II
#
# Manuel F. Diaz Ramos
#
# Homework 01.
######################################################

from dynModels import zonalHarmonicsModel
from obsModels import *
from observationSimulator import observationSimulator
import coordinateTransformations
import satellite
from ckfProc import ckfProc
from ekfProc import  ekfProc
import numpy as np
import matplotlib.pyplot as plt


###### Select observation model
# Selects the observation set to use
# 1: range-rate
# 2: right ascension-declination
# other: range + range_rate
RANGE_RANGE_RATE = 1
RA_DEC = 2
select_obs = RANGE_RANGE_RATE

###### Parameters
mu = 398600.4415e9        # [m^3/s^2] Earths gravitational parameter
R_E = 6378.1363e3         # [m] Earth's mean equatorial radius
e_earth = 0.0             # Earth eccentricity (ellipsoid model)

J_2 = 0.0010826269        # Earth J2 parameter
J_3 = -0.0000025323       # Earth J3 parameter

theta_0 = 0.0                   # [rad] Greenwhich Mean Sidereal time at t=0
theta_dot = 7.2921158553e-5     # [rad/sec] Earth's rotation angular velocity
w_matrix = np.array([[0,        -theta_dot, 0],
                     [theta_dot, 0,         0],
                     [0,         0,         0]])

####### Ground Station Coordinates
X_GS1 = -5127510.0 # [m]
Y_GS1 = -3794160.0 # [m]
Z_GS1 = 0.0        # [m]

X_GS2 = 3860910.0  # [m]
Y_GS2 = 3238490.0  # [m]
Z_GS2 = 3898094.0  # [m]

X_GS3 = 549505.0   # [m]
Y_GS3 = -1380872.0 # [m]
Z_GS3 = 6182197.0  # [m]

GS_coord = np.array([[X_GS1, Y_GS1, Z_GS1],
                    [X_GS2, Y_GS2, Z_GS2],
                    [X_GS3, Y_GS3, Z_GS3]])

####### Initial State
a = 10000e3                         # [m] Semi-major axis
e = 0.05                            # Eccentricity
inc = np.deg2rad(80.0)              # [rad] Inclination
raan = np.deg2rad(20.0)             # [rad] Right Ascension of Ascending Node
w = np.deg2rad(10.0)                # [rad] Argument of Perigee
nu = np.deg2rad(00.0)               # [rad] True Anomaly
period = 2*np.pi*np.sqrt(a**3/mu)   # [sec] Period


####### Simulation Parameters
rtol = 3e-13        # Relative Tolerance
atol = 1e-14        # Absolute tolerance

t0 = 0.0            # [sec] Initial Time
dt = 10.0           # [sec] Time step
tf = 3*period       # [sec] Final time

initState_orbEl = satellite.orbitalElements.getOrbitalElementsObj(a, e, inc, raan, w, nu, 0, 0, 0)
initialState = satellite.satelliteState.getSatelliteStateObjFromOrbElem(mu,initState_orbEl,t0)
X_0 = np.concatenate((initialState.getPosition(), initialState.getVelocity()))

####### Estimation Parameters
# Noise model
noise_flag = True
if select_obs == RA_DEC:
    mean_noise_obs = np.array([0,0])
    right_as_std = np.deg2rad(0.02/3600)            # [rad] Right Ascension Standard Deviation
    declination_std = np.deg2rad(0.02/3600)         # [rad] Declination Standard Deviation
    R_noise_obs = np.diag([right_as_std**2, declination_std**2])
else:
    mean_noise_obs = np.array([0,0])
    range_std = 1.0                 # [m] Range Standard Deviation
    range_rate_std = 1.0            # [m/s] Range-rate Standard Deviation
    R_noise_obs = np.diag([range_std**2, range_rate_std**2])

# Estimation Initial Conditions
Pbar_0 = np.diag([10,10,10,10,10,10])**2
Xref_0 = X_0 + np.random.multivariate_normal(np.zeros(6), Pbar_0)
xbar_0 = np.zeros(Xref_0.size)

######## Observation model
# Get observation model
if select_obs == RA_DEC:
    observationModel = rightAscensionDeclinationObsModel.getObserverModel(theta_0, theta_dot, R_E, e_earth, GS_coord)
    obs_used = '_ra_dec'
else:
    observationModel = rangeRangeRateObsModel.getObserverModel(theta_0, theta_dot, R_E, e_earth, GS_coord)
    obs_used = ''

######## Orbit simulation
J_sim = np.array([0, 0, J_2, J_3]) # Vector with the J params to simulate

orbModelGenerator = zonalHarmonicsModel.getDynamicModel(mu, R_E, J_sim)
orbModel_params = ()

obsGen = observationSimulator.getObservationSimulator(orbModelGenerator, observationModel)
obsGen.addNoise(noise_flag, mean_noise_obs, R_noise_obs)

obsGen.simulate(X_0, orbModel_params, t0, tf, dt, rtol, atol)
(observers, observations) = obsGen.getObservations() # observers: contains the number of observer. observations: contains the observations
obs_time_vec = obsGen.getTimeVector()
obs_states_vec = obsGen.getObservedStates() # Vector with the observed states
dynGen = obsGen.getDynamicSimulator() # all the states.

nmbrObs = obs_time_vec.size

state_list = satellite.stateList.getFromVectors(mu, dynGen.getStatesVector(), dynGen.getTimeVector())
state_time_vec = state_list.getTimeVector()

# J3 acceleration
J3_model = zonalHarmonicsModel.getDynamicModel(mu, R_E, np.array([0,0,0,J_3]), False)
J3_accel = J3_model.computeModelFromVector(dynGen.getStatesVector(), dynGen.getTimeVector(), ())[:,3:6]

plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, J3_accel[:,0], 'r', label='x_ECI component')
plt.plot(state_time_vec/3600, J3_accel[:,1], 'g', label='y_ECI component')
plt.plot(state_time_vec/3600, J3_accel[:,2], 'b', label='z_ECI component')
plt.plot(state_time_vec/3600, np.sqrt(J3_accel[:,0]**2+J3_accel[:,1]**2+J3_accel[:,2]**2), 'k', label='Magnitude')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.legend()
plt.xlabel('Simulation Time [h]')
plt.ylabel('J3 acceleration [m/s^2]')
plt.savefig('../report/include/J3_acceleration.png', bbox_inches='tight', dpi=300)

# Simulated orbit plots
plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, state_list.getSemimajorAxisVector()/1000, 'b')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.xlabel('Simulation Time [h]')
plt.ylabel('Semi-major axis [km]')
plt.savefig('../report/include/semi_major_axis.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, state_list.getEccentricityVector(), 'b')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.xlabel('Simulation Time [h]')
plt.ylabel('Eccentricity')
plt.savefig('../report/include/eccentricity.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, state_list.getInclinationVector()*180/np.pi, 'b')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.xlabel('Simulation Time [h]')
plt.ylabel('Inclination [deg]')
plt.savefig('../report/include/inclination.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, state_list.getRAANVector()*180/np.pi, 'b')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.xlabel('Simulation Time [h]')
plt.ylabel('Right ascension of ascending node [deg]')
plt.savefig('../report/include/raan.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, state_list.getArgumentOfPeriapsisVector()*180/np.pi, 'b')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.xlabel('Simulation Time [h]')
plt.ylabel('Argument of Perigee [deg]')
plt.savefig('../report/include/arg_perigee.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.hold(True)
plt.plot(state_time_vec/3600, state_list.getTrueAnomalyVector()*180/np.pi, 'b')
plt.xlim([0, state_time_vec[-1]/3600])
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.xlabel('Simulation Time [h]')
plt.ylabel('Right ascension of ascending node [deg]')
plt.savefig('../report/include/true_anomaly.png', bbox_inches='tight', dpi=300)


for i in range(0, obs_time_vec.size):
    if select_obs == RA_DEC:
        r_eci = obs_states_vec[i][0:3]
        (rightAs, dec, rang) = coordinateTransformations.eci2RightAscensionDeclinationRange(r_eci)
        print (rightAs - observations[i][0])
        print (dec - observations[i][1])
    else:
        GMST = theta_0 + theta_dot * obs_time_vec[i]
        r_eci = obs_states_vec[i][0:3]
        v_eci = obs_states_vec[i][3:6]
        range_vec = coordinateTransformations.eci2ecef(r_eci, GMST) - GS_coord[observers[i]]
        range_norm = np.linalg.norm(range_vec)

        R_EN = coordinateTransformations.ROT3(GMST)
        range_rate_vec = -w_matrix.dot(R_EN).dot(r_eci) + R_EN.dot(v_eci)

        range_rate = range_vec.dot(range_rate_vec)/range_norm

        print (range_norm - observations[i][0])
        print (range_rate - observations[i][1])


print obs_time_vec

print observers

print observations

print obs_states_vec

final_state = satellite.satelliteState.getSatelliteStateObjFromRV(mu,obs_states_vec[-1][0:3],obs_states_vec[-1][3:6],tf)

final_state.printOrbitalElements()


# Estimation
J_est = np.array([0, 0, J_2]) # Vector with the J params to estimate
orbModel = zonalHarmonicsModel.getDynamicModel(mu, R_E, J_est)

############################ Kalman processing#################################
joseph_formulation = True
ckfP = ckfProc.getCkfProc(orbModel, observationModel)
ckfP.configureCkf(Xref_0, xbar_0, Pbar_0, t0, joseph_formulation)

(Xhat_kalman,
 xhat_kalman,
 Xref_kalman,
 P_kalman,
 prefit_kalman,
 postfit_kalman) = ckfP.processAllObservations(observations, obs_time_vec, observers, R_noise_obs, dt, rtol, atol)

stm_final_kalman = ckfP.getSTM()
stm_inv_final_kalman = np.linalg.inv(stm_final_kalman)
xhat_final_kalman = xhat_kalman[-1,:]
xhat_0_kalman = stm_inv_final_kalman.dot(xhat_final_kalman)    # Estimation of the initial deviation
Xhat_0_kalman = Xref_kalman[0] + xhat_0_kalman
P_final_kalman = P_kalman[-1,:,:]

initial_satellite_state_kalman = satellite.satelliteState.getSatelliteStateObjFromRV(mu, Xhat_0_kalman[0:3], Xhat_0_kalman[3:6])

##### Trace of the covariance
P_kalman_pos_trace = np.zeros(nmbrObs)
P_kalman_vel_trace = np.zeros(nmbrObs)
negative_pos_trace_index = []
negative_vel_trace_index = []
for i in range(0, nmbrObs):
    P_kalman_pos_trace[i] = np.trace(P_kalman[i][0:3,0:3])
    P_kalman_vel_trace[i] = np.trace(P_kalman[i][3:6,3:6])

    if P_kalman_pos_trace[i] < 0:
       negative_pos_trace_index.append(i)

    if P_kalman_vel_trace[i] < 0:
       negative_vel_trace_index.append(i)

plt.figure()
plt.hold(True)
plt.semilogy(obs_time_vec/3600, np.abs(P_kalman_pos_trace), '.b')
plt.semilogy(obs_time_vec[negative_pos_trace_index]/3600, np.abs(P_kalman_pos_trace[negative_pos_trace_index]), '.r')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Position Covariance Trace (log) [m^2]')
plt.savefig('../report/include/kalman_position_covariance' + obs_used  + '.png', bbox_inches='tight', dpi=300)


plt.figure()
plt.hold(True)
plt.semilogy(obs_time_vec/3600, np.abs(P_kalman_vel_trace), '.b')
plt.semilogy(obs_time_vec[negative_vel_trace_index]/3600, np.abs(P_kalman_vel_trace[negative_vel_trace_index]), '.r')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Velocity Covariance Trace (log) [m^2/s^2]')
plt.savefig('../report/include/kalman_velocity_covariance' + obs_used  + '.png', bbox_inches='tight', dpi=300)

##### Errors
estimation_errors_ckf = Xhat_kalman - obs_states_vec
plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,0], '.', color='r')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_kalman[:,0,0])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_kalman[:,0,0])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('X Error [m]')
plt.subplot(312)
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,1], '.', color='g')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_kalman[:,1,1])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_kalman[:,1,1])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Y Error [m]')
plt.subplot(313)
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,2], '.', color='b')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_kalman[:,2,2])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_kalman[:,2,2])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Z Error [m]')
plt.legend()
plt.xlabel('Observation Time [h]')
plt.savefig('../report/include/errors_position_ckf' + obs_used  + '.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,3], '.', color='r')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_kalman[:,3,3])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_kalman[:,3,3])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('X_dot Error [m/s]')
plt.subplot(312)
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,4], '.', color='g')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_kalman[:,4,4])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_kalman[:,4,4])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Y_dot Error [m/s]')
plt.subplot(313)
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,5], '.', color='b')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_kalman[:,5,5])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_kalman[:,5,5])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Z_dot Error [m/s]')
plt.legend()
plt.xlabel('Observation Time [h]')
plt.savefig('../report/include/errors_velocity_ckf' + obs_used  + '.png', bbox_inches='tight', dpi=300)

##### Pre-fit residuals
prefit_RMS_kalman_firstobs = np.sqrt(np.sum(prefit_kalman[:,0]**2)/nmbrObs)
prefit_RMS_kalman_secondobs = np.sqrt(np.sum(prefit_kalman[:,1]**2)/nmbrObs)
plt.figure()
if select_obs == RA_DEC:
    plt.plot(obs_time_vec/3600, prefit_kalman[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Right Ascension RMS = ' + str(np.rad2deg(prefit_RMS_kalman_firstobs)) + 'deg')
    plt.plot(obs_time_vec/3600, prefit_kalman[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Declination RMS = ' + str(np.rad2deg(prefit_RMS_kalman_secondobs)) + ' deg')
else:
    plt.plot(obs_time_vec/3600, prefit_kalman[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Range RMS = ' + str(round(prefit_RMS_kalman_firstobs,3)) + ' m')
    plt.plot(obs_time_vec/3600, prefit_kalman[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Range-Rate RMS = ' + str(round(prefit_RMS_kalman_secondobs,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Range Pre-fit Residuals')
plt.savefig('../report/include/prefit_ckf' + obs_used  + '.png', bbox_inches='tight', dpi=300)


##### Post-fit residuals
postfit_RMS_kalman_firstobs = np.sqrt(np.sum(postfit_kalman[:,0]**2)/nmbrObs)
postfit_rate_RMS_kalman_secondobs = np.sqrt(np.sum(postfit_kalman[:,1]**2)/nmbrObs)
plt.figure()
if select_obs == RA_DEC:
    plt.plot(obs_time_vec/3600, postfit_kalman[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Right Ascension RMS = ' + str(np.rad2deg(postfit_RMS_kalman_firstobs)) + ' deg')
    plt.plot(obs_time_vec/3600, postfit_kalman[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Declination RMS = ' + str(np.rad2deg(postfit_rate_RMS_kalman_secondobs)) + ' deg')
else:
    plt.plot(obs_time_vec/3600, postfit_kalman[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Range RMS = ' + str(round(postfit_RMS_kalman_firstobs,3)) + ' m')
    plt.plot(obs_time_vec/3600, postfit_kalman[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Range-Rate RMS = ' + str(round(postfit_rate_RMS_kalman_secondobs,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Range Post-fit Residuals')
plt.savefig('../report/include/postfit_ckf' + obs_used  + '.png', bbox_inches='tight', dpi=300)

############################ EKF processing#################################
ekfP = ekfProc.getExtendedKalmanFilterProc(orbModel, observationModel)
ekfP.configureExtendedKalmanFilter(Xref_0, xbar_0, Pbar_0, t0, joseph_formulation)

if select_obs == RA_DEC:
    switch_to_ekf_after = 10
else:
    switch_to_ekf_after = 20

(Xhat_ekf,
 xhat_ekf,
 P_ekf,
 prefit_ekf,
 postfit_ekf) = ekfP.processAllObservations(observations, obs_time_vec, observers, R_noise_obs, dt, rtol, atol,switch_to_ekf_after, None)

stm_final_ekf = ekfP.getSTM()
stm_inv_final_ekf = np.linalg.inv(stm_final_ekf)
xhat_final_ekf = xhat_ekf[-1,:]
xhat_0_ekf = stm_inv_final_ekf.dot(xhat_final_ekf)    # Estimation of the initial deviation
P_final_ekf = P_ekf[-1,:,:]
#P_0_ekf = stm_inv_final_ekf.dot(P_ekf[-1,:,:]).dot(stm_inv_final_ekf)

##### Trace of the covariance
P_ekf_pos_trace = np.zeros(nmbrObs)
P_ekf_vel_trace = np.zeros(nmbrObs)
negative_pos_trace_index = []
negative_vel_trace_index = []
for i in range(0, nmbrObs):
    P_ekf_pos_trace[i] = np.trace(P_ekf[i][0:3,0:3])
    P_ekf_vel_trace[i] = np.trace(P_ekf[i][3:6,3:6])

    if P_ekf_pos_trace[i] < 0:
       negative_pos_trace_index.append(i)

    if P_ekf_vel_trace[i] < 0:
       negative_vel_trace_index.append(i)

plt.figure()
plt.hold(True)
plt.semilogy(obs_time_vec/3600, np.abs(P_ekf_pos_trace), '.b')
plt.semilogy(obs_time_vec[negative_pos_trace_index]/3600, np.abs(P_ekf_pos_trace[negative_pos_trace_index]), '.r')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Position Covariance Trace (log) [m^2]')
plt.savefig('../report/include/ekf_position_covariance' + obs_used  + '.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.semilogy(obs_time_vec/3600, np.abs(P_ekf_vel_trace), '.b')
plt.semilogy(obs_time_vec[negative_vel_trace_index]/3600, np.abs(P_ekf_vel_trace[negative_vel_trace_index]), '.r')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Velocity Covariance Trace (log) [m^2/s^2]')
plt.savefig('../report/include/ekf_velocity_covariance' + obs_used  + '.png', bbox_inches='tight', dpi=300)


##### Errors
estimation_errors_ekf = Xhat_ekf - obs_states_vec
plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,0], '.', color='r')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_ekf[:,0,0])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_ekf[:,0,0])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('X Error [m]')
plt.subplot(312)
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,1], '.', color='g')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_ekf[:,1,1])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_ekf[:,1,1])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Y Error [m]')
plt.subplot(313)
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,2], '.', color='b')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_ekf[:,2,2])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_ekf[:,2,2])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Z Error [m]')
plt.legend()
plt.xlabel('Observation Time [h]')
plt.savefig('../report/include/errors_position_ekf' + obs_used  + '.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,3], '.', color='r')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_ekf[:,3,3])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_ekf[:,3,3])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('X_dot Error [m/s]')
plt.subplot(312)
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,4], '.', color='g')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_ekf[:,4,4])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_ekf[:,4,4])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Y_dot Error [m/s]')
plt.subplot(313)
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,5], '.', color='b')
plt.plot(obs_time_vec/3600, 3*np.abs(np.sqrt(P_ekf[:,5,5])), '-k')
plt.plot(obs_time_vec/3600, -3*np.abs(np.sqrt(P_ekf[:,5,5])), '-k')
plt.xlim([0, obs_time_vec[-1]/3600])
plt.ylabel('Z_dot Errors [m/s]')
plt.legend()
plt.xlabel('Observation Time [h]')
plt.savefig('../report/include/errors_velocity_ekf' + obs_used  + '.png', bbox_inches='tight', dpi=300)

##### Pre-fit residuals
prefit_RMS_ekf_firstobs = np.sqrt(np.sum(prefit_ekf[:,0]**2)/nmbrObs)
prefit_RMS_ekf_secondobs = np.sqrt(np.sum(prefit_ekf[:,1]**2)/nmbrObs)
plt.figure()
if select_obs == RA_DEC:
    plt.plot(obs_time_vec/3600, prefit_ekf[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Right Ascension RMS = ' + str(np.rad2deg(prefit_RMS_ekf_firstobs)) + ' deg')
    plt.plot(obs_time_vec/3600, prefit_ekf[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Declination RMS = ' + str(np.rad2deg(prefit_RMS_ekf_secondobs)) + ' deg')
else:
    plt.plot(obs_time_vec/3600, prefit_ekf[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Range RMS = ' + str(round(prefit_RMS_ekf_firstobs,3)) + ' m')
    plt.plot(obs_time_vec/3600, prefit_ekf[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Range-Rate RMS = ' + str(round(prefit_RMS_ekf_secondobs,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Pre-fit Residuals')
plt.savefig('../report/include/prefit_ekf' + obs_used  + '.png', bbox_inches='tight', dpi=300)

##### Post-fit residuals
postfit_RMS_ekf_firstobs = np.sqrt(np.sum(postfit_ekf[:,0]**2)/nmbrObs)
postfit_RMS_ekf_secondobs = np.sqrt(np.sum(postfit_ekf[:,1]**2)/nmbrObs)
plt.figure()
if select_obs == RA_DEC:
    plt.plot(obs_time_vec/3600, postfit_ekf[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Right Ascension RMS = ' + str(np.rad2deg(postfit_RMS_ekf_firstobs)) + ' deg')
    plt.plot(obs_time_vec/3600, postfit_ekf[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Declination RMS = ' + str(np.rad2deg(postfit_RMS_ekf_secondobs)) + ' deg')
else:
    plt.plot(obs_time_vec/3600, postfit_ekf[:,0]/np.sqrt(R_noise_obs[0,0]), '.', color='b', label='Range RMS = ' + str(round(postfit_RMS_ekf_firstobs,3)) + ' m')
    plt.plot(obs_time_vec/3600, postfit_ekf[:,1]/np.sqrt(R_noise_obs[1,1]), '.', color='g', label='Range-Rate RMS = ' + str(round(postfit_RMS_ekf_secondobs,3)) + ' m/s')

plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Post-fit Residuals')
plt.savefig('../report/include/postfit_ekf' + obs_used  + '.png', bbox_inches='tight', dpi=300)
