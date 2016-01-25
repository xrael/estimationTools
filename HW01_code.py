

from dynModels import zonalHarmonicsModel
from obsModels import rangeRangeRateObsModel
from observationSimulator import observationSimulator
import coordinateTransformations
import satellite
from ckfProc import ckfProc
from ekfProc import  ekfProc
import numpy as np
import matplotlib.pyplot as plt


# Parameters
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

# Ground Station Coordinates
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

# Initial State
a = 10000e3                         # [m] Semi-major axis
e = 0.05                            # Eccentricity
inc = np.deg2rad(90.0)              # [rad] Inclination
raan = np.deg2rad(20.0)                          # [rad] Right Ascension of Ascending Node
w = 0.0                             # [rad] Argument of Perigee
nu = 0.0                            # [rad] True Anomaly
period = 2*np.pi*np.sqrt(a**3/mu)   # [sec] Period


# Simulation Parameters
rtol = 3e-13        # Relative Tolerance
atol = 1e-14        # Absolute tolerance

t0 = 0.0            # [sec] Initial Time
dt = 10.0           # [sec] Time step
tf = 10*period       # [sec] Final time


initState_orbEl = satellite.orbitalElements.getOrbitalElementsObj(a, e, inc, raan, w, nu, 0, 0, 0)
initialState = satellite.satelliteState.getSatelliteStateObjFromOrbElem(mu,initState_orbEl,t0)
X_0 = np.concatenate((initialState.getPosition(), initialState.getVelocity()))

print X_0

# Estimation Parameters
# Noise model
noise_flag = True
mean_noise_obs = np.array([0,0])
range_std = 1.0                 # [m] Range Standard Deviation
range_rate_std = 1.0            # [m/s] Range-rate Standard Deviation
R_noise_obs = np.diag([range_std**2, range_rate_std**2])

# Estimation Initial Conditions
Pbar_0= np.diag([100,100,100,100,100,100])**2
Xref_0 = X_0 + np.random.multivariate_normal(np.zeros(6), Pbar_0)
print Xref_0
xbar_0 = np.zeros(Xref_0.size)


# Observation model
observationModel = rangeRangeRateObsModel.getObserverModel(theta_0, theta_dot, R_E, e_earth, GS_coord)

# Orbit simulation
J_sim = np.array([0, 0]) # Vector with the J params to simulate

orbModelGenerator = zonalHarmonicsModel.getDynamicModel(mu, R_E, J_sim)
orbModel_params = ()

obsGen = observationSimulator.getObservationSimulator(orbModelGenerator, observationModel)
obsGen.addNoise(noise_flag, mean_noise_obs, R_noise_obs)

obsGen.simulate(X_0, orbModel_params, t0, tf, dt, rtol, atol)
(observers, observations) = obsGen.getObservations() # observers: contains the number of observer. observations: contains the observations
obs_time_vec = obsGen.getTimeVector()
obs_states_vec = obsGen.getObservedStates() # Vector with the observed states
nmbrObs = obs_time_vec.size

for i in range(0, obs_time_vec.size):
    GMST = theta_0 + theta_dot * obs_time_vec[i]
    r_eci = obs_states_vec[i][0:3]
    v_eci = obs_states_vec[i][3:6]
    range_vec = coordinateTransformations.eci2ecef(r_eci, GMST) - GS_coord[observers[i]]
    range = np.linalg.norm(range_vec)

    R_EN = coordinateTransformations.ROT3(GMST)
    range_rate_vec = -w_matrix.dot(R_EN).dot(r_eci) + R_EN.dot(v_eci)

    range_rate = range_vec.dot(range_rate_vec)/range

    print (range - observations[i][0])
    print (range_rate - observations[i][1])


print obs_time_vec

print observers

print observations

print obs_states_vec

final_state = satellite.satelliteState.getSatelliteStateObjFromRV(mu,obs_states_vec[-1][0:3],obs_states_vec[-1][3:6],tf)

final_state.printOrbitalElements()


# Estimation
J_est = np.array([0, 0]) # Vector with the J params to estimate
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

##### Errors
estimation_errors_ckf = Xhat_kalman - obs_states_vec
plt.figure()
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,0], '.', color='r')
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,1], '.', color='g')
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,2], '.', color='b')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Position Errors [m]')
plt.savefig('../report/include/errors_position_ckf.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,3], '.', color='r')
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,4], '.', color='g')
plt.plot(obs_time_vec/3600, estimation_errors_ckf[:,5], '.', color='b')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Velocity Errors [m/s]')
plt.savefig('../report/include/errors_velocity_ckf.png', bbox_inches='tight', dpi=300)

##### Pre-fit residuals
prefit_range_RMS_kalman = np.sqrt(np.sum(prefit_kalman[:,0]**2)/nmbrObs)
prefit_range_rate_RMS_kalman = np.sqrt(np.sum(prefit_kalman[:,1]**2)/nmbrObs)
plt.figure()
plt.plot(obs_time_vec/3600, prefit_kalman[:,0]/range_std, '.', color='b', label='Range RMS = ' + str(round(prefit_range_RMS_kalman,3)) + ' m')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Range Pre-fit Residuals')
plt.savefig('../report/include/prefit_range_kalman.png', bbox_inches='tight', dpi=300)


plt.figure()
plt.plot(obs_time_vec/3600, prefit_kalman[:,1]/range_rate_std, '.', color='g', label='Range-Rate RMS = ' + str(round(prefit_range_rate_RMS_kalman,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Range-rate Pre-fit Residuals')
plt.savefig('../report/include/prefit_range_rate_kalman.png', bbox_inches='tight', dpi=300)


##### Post-fit residuals
postfit_range_RMS_kalman = np.sqrt(np.sum(postfit_kalman[:,0]**2)/nmbrObs)
postfit_range_rate_RMS_kalman = np.sqrt(np.sum(postfit_kalman[:,1]**2)/nmbrObs)
plt.figure()
plt.plot(obs_time_vec/3600, postfit_kalman[:,0]/range_std, '.', color='b', label='Range RMS = ' + str(round(postfit_range_RMS_kalman,3)) + ' m')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Normalized Range Post-fit Residuals')
plt.savefig('../report/include/postfit_range_kalman.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(obs_time_vec/3600, postfit_kalman[:,1]/range_rate_std, '.', color='g', label='Range-Rate RMS = ' + str(round(postfit_range_rate_RMS_kalman,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('NormalizedRange-rate Post-fit Residuals')
plt.savefig('../report/include/postfit_range_rate_kalman.png', bbox_inches='tight', dpi=300)



############################ EKF processing#################################
ekfP = ekfProc.getExtendedKalmanFilterProc(orbModel, observationModel)
ekfP.configureExtendedKalmanFilter(Xref_0, xbar_0, Pbar_0, t0, joseph_formulation)

(Xhat_ekf,
 xhat_ekf,
 P_ekf,
 prefit_ekf,
 postfit_ekf) = ekfP.processAllObservations(observations, obs_time_vec, observers, R_noise_obs, dt, rtol, atol,100, None)

stm_final_ekf = ekfP.getSTM()
stm_inv_final_ekf = np.linalg.inv(stm_final_ekf)
xhat_final_ekf = xhat_ekf[-1,:]
xhat_0_ekf = stm_inv_final_ekf.dot(xhat_final_ekf)    # Estimation of the initial deviation
P_final_ekf = P_ekf[-1,:,:]
#P_0_ekf = stm_inv_final_ekf.dot(P_ekf[-1,:,:]).dot(stm_inv_final_ekf)

##### Errors
estimation_errors_ekf = Xhat_ekf - obs_states_vec
plt.figure()
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,0], '.', color='r')
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,1], '.', color='g')
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,2], '.', color='b')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Position Errors [m]')
plt.savefig('../report/include/errors_position_ekf.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,3], '.', color='r')
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,4], '.', color='g')
plt.plot(obs_time_vec/3600, estimation_errors_ekf[:,5], '.', color='b')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Velocity Errors [m/s]')
plt.savefig('../report/include/errors_velocity_ekf.png', bbox_inches='tight', dpi=300)

##### Pre-fit residuals
prefit_range_RMS_ekf = np.sqrt(np.sum(prefit_ekf[:,0]**2)/nmbrObs)
prefit_range_rate_RMS_ekf = np.sqrt(np.sum(prefit_ekf[:,1]**2)/nmbrObs)
plt.figure()
plt.plot(obs_time_vec/3600, prefit_ekf[:,0], '.', color='b', label='Range RMS = ' + str(round(prefit_range_RMS_ekf,3)) + ' m')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Range Pre-fit Residuals [m]')
plt.savefig('../report/include/prefit_range_ekf.png', bbox_inches='tight', dpi=300)


plt.figure()
plt.plot(obs_time_vec/3600, prefit_ekf[:,1], '.', color='g', label='Range-Rate RMS = ' + str(round(prefit_range_rate_RMS_ekf,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Range-rate Pre-fit Residuals [m/s]')
plt.savefig('../report/include/prefit_range_rate_ekf.png', bbox_inches='tight', dpi=300)


##### Post-fit residuals
postfit_range_RMS_ekf = np.sqrt(np.sum(postfit_ekf[:,0]**2)/nmbrObs)
postfit_range_rate_RMS_ekf = np.sqrt(np.sum(postfit_ekf[:,1]**2)/nmbrObs)
plt.figure()
plt.plot(obs_time_vec/3600, postfit_ekf[:,0], '.', color='b', label='Range RMS = ' + str(round(postfit_range_RMS_ekf,3)) + ' m')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Range Post-fit Residuals [m]')
plt.savefig('../report/include/postfit_range_ekf.png', bbox_inches='tight', dpi=300)

plt.figure()
plt.plot(obs_time_vec/3600, postfit_ekf[:,1], '.', color='g', label='Range-Rate RMS = ' + str(round(postfit_range_rate_RMS_ekf,3)) + ' m/s')
plt.legend()
plt.xlim([0, obs_time_vec[-1]/3600])
plt.xlabel('Observation Time [h]')
plt.ylabel('Range-rate Post-fit Residuals [m/s]')
plt.savefig('../report/include/postfit_range_rate_ekf.png', bbox_inches='tight', dpi=300)