######################################################
# ASEN 6080: Statistical Orbit Determination II
#
# Manuel F. Diaz Ramos
#
# Final Project
######################################################

from attitudeModels import rigidBodyAttittudeDynamicsModelMRPs, mrpKinematicsModel
from attitudeObsModels import *
from observationSimulator import observationSimulator
from integrators import Integrator
import attitudeKinematics
from ekfProc import  ekfProc
import numpy as np
import matplotlib.pyplot as plt


###### Attitude model

# Principal axis moments of inertia
I11 = 10.0      # [kg m^2] b1 Principal axis inertia
I22 = 5.0       # [kg m^2] b2 Principal axis inertia
I33 = 2.0       # [kg m^2] b3 Principal axis inertia

attitudeModelGenerator = rigidBodyAttittudeDynamicsModelMRPs.getDynamicModel(I11, I22, I33)

##### Initial conditions
mrp_BN_0 = np.array([0.3, -0.4, 0.5])               # Initial MRP Body/Inertial
w_BN_0_B = np.array([0.01, 0.04, -0.02])               # Initial angular velocity Body/Inertial written in Body frame                                      # [rad] Initial Argument of Latitude

##### Sensor models

# Simulation parameters
t0 = 0.0
tf = 1000.0
dt = 0.1


### MRP Observations
mrp_obs_rate = 0.2     # [Hz]
# Errors
mean_noise_mrp_obs = np.array([0.0,0.0,0.0])
R_noise_mrp_obs = 0.001**2*np.diag([1.0, 1.0, 1.0]) # MRPs

### Star Tracker
star_catalog_ST = star_catalog
for entry in star_catalog_ST:
    entry[2] = np.deg2rad(360 - entry[2]) # transforms from Sidereal hour angle to right ascension
    entry[3] = np.deg2rad(entry[3])

# Two star trackers in the body directions b1 and -b1
star_tracker_direction_body = [[1.0, 0.0, 0.0],
                               [-1.0, 0.0, 0.0]]
# Star tracker FOV: 120 deg
star_tracker_fov = [np.deg2rad(140),np.deg2rad(140)]

star_tracker_rate = 0.2     # [Hz]
mean_noise_ST = np.array([0.0,0.0,0.0])
R_noise_ST = 0.003**2*np.diag([1.0, 1.0, 1.0])

### Rate-gyros
gyro_rate = 10.0            # [Hz]

# Errors (ADIS 16137)
N_ARW = 0.55        # [deg/sqrt(hour)]
K_RRW = 7.0         # [deg/sqrt(hour^3)]

sigma_ARW = np.deg2rad(N_ARW/np.sqrt(3600)) # ARW: Angular Random Walk (rate white noise)
sigma_RRW = np.deg2rad(K_RRW/np.sqrt(3600**3)) # RRW: Rate Random Walk
initial_gyro_bias = np.array([0.001,0.002,-0.001])
Q = np.diag([sigma_ARW, sigma_ARW, sigma_ARW, sigma_RRW, sigma_RRW, sigma_RRW])**2

####### Observation simulator: MRP Observations
MRPobsModelGenerator = mrpObs.getObserverModel(mrp_obs_rate)
MRPobsModelGenerator.setNoise(True, mean_noise_mrp_obs, R_noise_mrp_obs)

gyroObsModelGenerator = rateGyroObs.getObserverModel(gyro_rate)
gyroObsModelGenerator.setNoise(True, None, None, [sigma_ARW*np.array([1.0,1.0,1.0]), sigma_RRW*np.array([1.0,1.0,1.0]), initial_gyro_bias])

obsGen = observationSimulator.getObservationSimulator(attitudeModelGenerator, [MRPobsModelGenerator,gyroObsModelGenerator], Integrator.RK4, attitudeKinematics.switchMRPrepresentation)

obsGen.simulate(np.concatenate((mrp_BN_0, w_BN_0_B)), (), t0, tf, dt)
(observers, observations) = obsGen.getObservations() # observers: contains the number of observer. observations: contains the observations
obs_time = obsGen.getTimeVector()
obs_states = obsGen.getObservedStates() # Vector with the observed states
#dynGen = gyroGen.getDynamicSimulator() # all the states.

obs_time_vec_mrp_obs = obs_time[0]
observations_mrp_obs = observations[0]
obs_true_states_mrp_obs = obs_states[0]
observers_mrp_obs = observers[0]

nmbr_obs_mrp_obs = len(observations_mrp_obs)

obs_time_vec_GYRO = obs_time[1]
observations_GYRO = observations[1]
obs_true_states_GYRO = obs_states[1]

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,0], label='$\sigma_1$ true')
plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,0], '.', label='$\sigma_1$ observed')
plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,0]-obs_true_states_mrp_obs[:,0], label='$\sigma_1$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\sigma_1$", size=14)
plt.subplot(312)
plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,1], label='$\sigma_2$ true')
plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,1], '.', label='$\sigma_2$ observed')
plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,1]-obs_true_states_mrp_obs[:,1], label='$\sigma_2$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\sigma_2$", size=14)
plt.subplot(313)
plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,2], label='$\sigma_3$ true')
plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,2], '.', label='$\sigma_3$ observed')
plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,2]-obs_true_states_mrp_obs[:,2], label='$\sigma_3$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\sigma_3$", size=14)
plt.xlabel("Time [sec]")
plt.savefig('../report/include/sigma.png', bbox_inches='tight', dpi=300)
plt.close()

# plt.figure()
# plt.hold(True)
# plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,2], label='$\sigma_3$ true')
# plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,2], label='$\sigma_3$ observed')
# plt.plot(obs_time_vec_mrp_obs, observations_mrp_obs[:,2]-obs_true_states_mrp_obs[:,2], label='$\sigma_3$ Error')
# plt.xlabel("Time [sec]")
# plt.ylabel("$\sigma_3$", size=18)
# plt.legend()
# plt.savefig('../report/include/sigma_3.png', bbox_inches='tight', dpi=300)
# plt.close()

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_GYRO, obs_true_states_GYRO[:,3], label='$\omega_1$ true')
plt.plot(obs_time_vec_GYRO, observations_GYRO[:,0], label='$\omega_1$ observed')
plt.plot(obs_time_vec_GYRO, observations_GYRO[:,0]-obs_true_states_GYRO[:,3], label='$\omega_1$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\omega_1$ $[rad/sec]$", size=14)
plt.subplot(312)
plt.plot(obs_time_vec_GYRO, obs_true_states_GYRO[:,4], label='$\omega_2$ true')
plt.plot(obs_time_vec_GYRO, observations_GYRO[:,1], label='$\omega_2$ observed')
plt.plot(obs_time_vec_GYRO, observations_GYRO[:,1]-obs_true_states_GYRO[:,4], label='$\omega_2$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\omega_2$ $[rad/sec]$", size=14)
plt.subplot(313)
plt.plot(obs_time_vec_GYRO, obs_true_states_GYRO[:,5], label='$\omega_3$ true')
plt.plot(obs_time_vec_GYRO, observations_GYRO[:,2], label='$\omega_3$ observed')
plt.plot(obs_time_vec_GYRO, observations_GYRO[:,2]-obs_true_states_GYRO[:,5], label='$\omega_3$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\omega_3$ $[rad/sec]$", size=14)
plt.xlabel("Time [sec]")
plt.savefig('../report/include/w.png', bbox_inches='tight', dpi=300)
plt.close()


# plt.figure()
# plt.hold(True)
# plt.plot(obs_time_vec_GYRO, obs_true_states_GYRO[:,5], label='$\omega_3$ true')
# plt.plot(obs_time_vec_GYRO, observations_GYRO[:,2], label='$\omega_3$ observed')
# plt.plot(obs_time_vec_GYRO, observations_GYRO[:,2]-obs_true_states_GYRO[:,5], label='$\omega_3$ Error')
# plt.xlabel("Time [sec]")
# plt.ylabel("$\omega_3$", size=18)
# plt.legend()
# plt.savefig('../report/include/w_3.png', bbox_inches='tight', dpi=300)
# plt.close()


####### Observation simulator: Star tracker Observations
starTrackerModelGenerator = starTrackerObs.getObserverModel(star_tracker_rate, star_catalog_ST, star_tracker_direction_body, star_tracker_fov)
starTrackerModelGenerator.setNoise(True, mean_noise_ST, R_noise_ST)

gyroObsModelGenerator = rateGyroObs.getObserverModel(gyro_rate)
gyroObsModelGenerator.setNoise(True, None, None, [sigma_ARW*np.array([1.0,1.0,1.0]), sigma_RRW*np.array([1.0,1.0,1.0]), initial_gyro_bias])

obsSTGen = observationSimulator.getObservationSimulator(attitudeModelGenerator, [starTrackerModelGenerator,gyroObsModelGenerator], Integrator.RK4, attitudeKinematics.switchMRPrepresentation)

obsSTGen.simulate(np.concatenate((mrp_BN_0, w_BN_0_B)), (), t0, tf, dt)
(observers_STgyro, observations_STgyro) = obsSTGen.getObservations() # observers: contains the number of observer. observations: contains the observations
obs_time_STgyro = obsSTGen.getTimeVector()
obs_states_STgyro = obsSTGen.getObservedStates() # Vector with the observed states
#dynGen = gyroGen.getDynamicSimulator() # all the states.

obs_time_vec_ST = obs_time_STgyro[0]
observations_ST = observations_STgyro[0]
obs_true_states_ST = obs_states_STgyro[0]
observers_ST = observers_STgyro[0]

nmbr_obs_ST = len(observations_ST)

obs_time_vec_GYROwST = obs_time_STgyro[1]
observations_GYROwST = observations_STgyro[1]
obs_true_states_GYROwST = obs_states_STgyro[1]
observers_GYROwST = observers_STgyro[0]

r_B_true_ST = np.zeros((nmbr_obs_ST, 3))
for i in range(0, nmbr_obs_ST):
    star_observed = star_catalog_ST[observers_ST[i]]
    star_RA = star_observed[2]
    star_dec = star_observed[3]
    cos_right_asc = np.cos(star_RA)
    sin_right_asc = np.sin(star_RA)
    cos_dec = np.cos(star_dec)
    sin_dec = np.sin(star_dec)

    r_N = np.array([cos_dec*cos_right_asc, cos_dec*sin_right_asc, sin_dec])

    mrp = obs_true_states_ST[i,0:3]

    BN = attitudeKinematics.mrp2dcm(mrp)

    r_B_true_ST[i,:] = BN.dot(r_N)

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_ST, r_B_true_ST[:,0], label='$x_{star-B}$ true')
plt.plot(obs_time_vec_ST, observations_ST[:,0], '.', label='$x_{star-B}$ observed')
plt.plot(obs_time_vec_ST, observations_ST[:,0]-r_B_true_ST[:,0], label='$x_{star-B}$ Error')
plt.ylabel("$x_{star-B}$", size=18)
plt.legend()
plt.subplot(312)
plt.plot(obs_time_vec_ST, r_B_true_ST[:,1], label='$y_{star-B}$ true')
plt.plot(obs_time_vec_ST, observations_ST[:,1], '.', label='$y_{star-B}$ observed')
plt.plot(obs_time_vec_ST, observations_ST[:,1]-r_B_true_ST[:,1], label='$y_{star-B}$ Error')
plt.ylabel("$y_{star-B}$", size=18)
plt.legend()
plt.subplot(313)
plt.plot(obs_time_vec_ST, r_B_true_ST[:,2], label='$z_{star-B}$ true')
plt.plot(obs_time_vec_ST, observations_ST[:,2], '.', label='$z_{star-B}$ observed')
plt.plot(obs_time_vec_ST, observations_ST[:,2]-r_B_true_ST[:,2], label='$z_{star-B}$ Error')
plt.ylabel("$z_{star-B}$", size=18)
plt.legend()
plt.xlabel("Time [sec]")
plt.savefig('../report/include/r_B_ST.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_ST, obs_true_states_ST[:,0], label='$\sigma_1$ true')
plt.subplot(312)
plt.plot(obs_time_vec_ST, obs_true_states_ST[:,1], label='$\sigma_2$ true')
plt.subplot(313)
plt.plot(obs_time_vec_ST, obs_true_states_ST[:,2], label='$\sigma_3$ true')
plt.xlabel("Time [sec]")
plt.ylabel("$\sigma$", size=18)
plt.savefig('../report/include/sigma_ST.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_GYROwST, obs_true_states_GYROwST[:,3], label='$\omega_1$ true')
plt.plot(obs_time_vec_GYROwST, observations_GYROwST[:,0], label='$\omega_1$ observed')
plt.plot(obs_time_vec_GYROwST, observations_GYROwST[:,0]-obs_true_states_GYROwST[:,3], label='$\omega_1$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\omega_1$ $[rad/sec]$", size=14)
plt.subplot(312)
plt.plot(obs_time_vec_GYROwST, obs_true_states_GYROwST[:,4], label='$\omega_2$ true')
plt.plot(obs_time_vec_GYROwST, observations_GYROwST[:,1], label='$\omega_2$ observed')
plt.plot(obs_time_vec_GYROwST, observations_GYROwST[:,1]-obs_true_states_GYROwST[:,4], label='$\omega_2$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\omega_2$ $[rad/sec]$", size=14)
plt.subplot(313)
plt.plot(obs_time_vec_GYROwST, obs_true_states_GYROwST[:,5], label='$\omega_3$ true')
plt.plot(obs_time_vec_GYROwST, observations_GYROwST[:,2], label='$\omega_3$ observed')
plt.plot(obs_time_vec_GYROwST, observations_GYROwST[:,2]-obs_true_states_GYROwST[:,5], label='$\omega_3$ Error')
plt.legend(prop={'size':8})
plt.ylabel("$\omega_3$ $[rad/sec]$", size=14)
plt.xlabel("Time [sec]")
plt.savefig('../report/include/w_ST.png', bbox_inches='tight', dpi=300)
plt.close()




########################## EKF Estimation ###############################

######################## Using MRP Observations #########################
### Filter initial estimate
Pbar_0 = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])**2
Xbar_0 = np.concatenate([np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])])


getAngularVelocityObservations = lambda t: obsGen.getNextObservation(t, 1) # Gyro Observations

# Estimation dynamic model
attitudeModelEstimation = mrpKinematicsModel.getDynamicModel(getAngularVelocityObservations, propagationFunction='F_plus_STM')
attitudeModelEstimation.setNoiseCompensation(True)

ekfP = ekfProc.getFilter(attitudeModelEstimation, MRPobsModelGenerator, Integrator.RK4, attitudeKinematics.switchMRPrepresentation)
ekfP.configureFilter(Xbar_0, Pbar_0, t0)
ekfP.josephFormulation(True)
#ckfP.setNumberIterations(3)

ekfP.startEKFafter(0)
ekfP.processAllObservations(observations_mrp_obs, obs_time_vec_mrp_obs, observers_mrp_obs, R_noise_mrp_obs, dt, Q = Q)
Xhat_ekf = ekfP.getEstimateVector()
xhat_ekf = ekfP.getDeviationEstimateVector()
Xref_ekf = ekfP.getReferenceStateVector()

P_ekf = ekfP.getCovarianceMatrixVector()
prefit_ekfP = ekfP.getPrefitResidualsVector()
postfit_ekfP = ekfP.getPostfitResidualsVector()
stm_ekfP = ekfP.getSTMMatrixfrom0Vector()

ekfP.plotPrefitResiduals(R_noise_mrp_obs, ['MRP 1', 'MRP 2', 'MRP3'], ['', '', ''],['b', 'g', 'r'],'../report/include/prefit_ekf.png')
ekfP.plotPostfitResiduals(R_noise_mrp_obs, ['MRP 1', 'MRP 2', 'MRP3'], ['', '', ''],['b', 'g', 'r'],'../report/include/postfit_ekf.png')


##### Errors
mrp_estimation_errors_ekf = Xhat_ekf[:,0:3] - obs_true_states_mrp_obs[:,0:3]

mrp1_errors_rms_ekf = np.sqrt(np.sum(mrp_estimation_errors_ekf[:,0]**2)/nmbr_obs_mrp_obs)
mrp2_errors_rms_ekf = np.sqrt(np.sum(mrp_estimation_errors_ekf[:,1]**2)/nmbr_obs_mrp_obs)
mrp3_errors_rms_ekf = np.sqrt(np.sum(mrp_estimation_errors_ekf[:,2]**2)/nmbr_obs_mrp_obs)

plt.figure()
plt.hold(True)
plt.subplot(311)
plt.plot(obs_time_vec_mrp_obs, Xhat_ekf[:,0], '.', color='r', label = 'EKF $\hat\sigma_1$')
plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,0],'.', color='g', label = 'True $\sigma_1$')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.legend()
plt.ylabel('$\hat\sigma_1$',size=14)
plt.subplot(312)
plt.plot(obs_time_vec_mrp_obs, Xhat_ekf[:,1], '.', color='r', label = 'EKF $\hat\sigma_2$')
plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,1],'.', color='g', label = 'True $\sigma_2$')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.legend()
plt.ylabel('$\hat\sigma_2$',size=14)
plt.subplot(313)
plt.plot(obs_time_vec_mrp_obs, Xhat_ekf[:,2], '.', color='r', label = 'EKF $\hat\sigma_3$')
plt.plot(obs_time_vec_mrp_obs, obs_true_states_mrp_obs[:,2],'.', color='g', label = 'True $\sigma_3$')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.legend()
plt.ylabel('$\hat\sigma_3$',size=14)
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/estimated_mrp.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.subplot(311)
plt.plot(obs_time_vec_mrp_obs, Xref_ekf[:,0], '.', color='r', label = 'EKF $\sigma_{ref-1}$')
plt.ylabel('$\sigma_{ref-1}$')
plt.legend()
plt.subplot(312)
plt.plot(obs_time_vec_mrp_obs, Xref_ekf[:,1], '.', color='r', label = 'EKF $\sigma_{ref-2}$')
plt.ylabel('$\sigma_{ref-2}$')
plt.legend()
plt.subplot(313)
plt.plot(obs_time_vec_mrp_obs, Xref_ekf[:,2], '.', color='r', label = 'EKF $\sigma_{ref-3}$')
plt.ylabel('$\sigma_{ref-2}$')
plt.legend()
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/reference_mrp.png', bbox_inches='tight', dpi=300)
plt.close()


plt.figure()
plt.hold(True)
plt.subplot(311)
plt.plot(obs_time_vec_mrp_obs, Xhat_ekf[:,3], '.', color='r', label = 'EKF $\hat \\beta_1$')
plt.axhline(y=initial_gyro_bias[0], color='g', label = 'True $\\beta_1$')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.legend()
plt.ylabel('$\hat \\beta_1$')
plt.subplot(312)
plt.plot(obs_time_vec_mrp_obs, Xhat_ekf[:,4], '.', color='r', label = 'EKF $\hat \\beta_2$')
plt.axhline(y=initial_gyro_bias[1], color='g', label = 'True $\\beta_2$')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.legend()
plt.ylabel('$\hat \\beta_2$')
plt.subplot(313)
plt.plot(obs_time_vec_mrp_obs, Xhat_ekf[:,5], '.', color='r', label = 'EKF $\hat \\beta_3$')
plt.axhline(y=initial_gyro_bias[2], color='g', label = 'True $\\beta_3$')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.legend()
plt.ylabel('$\hat \\beta_3$')
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/estimated_bias.png', bbox_inches='tight', dpi=300)
plt.close()


plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_mrp_obs, mrp_estimation_errors_ekf[:,0], '.', color='r', label = 'EKF RMS = ' + str(round(mrp1_errors_rms_ekf,3)))
plt.plot(obs_time_vec_mrp_obs, 3*np.abs(np.sqrt(P_ekf[:,0,0])), '--k')
plt.plot(obs_time_vec_mrp_obs, -3*np.abs(np.sqrt(P_ekf[:,0,0])), '--k')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.ylim([-0.005,0.005])
#plt.axvline(3, color='k',linestyle='--')
plt.ylabel('$\Delta \sigma_1$',size=14)
plt.legend()
plt.subplot(312)
plt.plot(obs_time_vec_mrp_obs, mrp_estimation_errors_ekf[:,1], '.', color='r', label = 'EKF RMS = ' + str(round(mrp2_errors_rms_ekf,3)))
plt.plot(obs_time_vec_mrp_obs, 3*np.abs(np.sqrt(P_ekf[:,1,1])), '--k')
plt.plot(obs_time_vec_mrp_obs, -3*np.abs(np.sqrt(P_ekf[:,1,1])), '--k')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.ylim([-0.005,0.005])
#plt.axvline(3, color='k',linestyle='--')
plt.ylabel('$\Delta \sigma_2$',size=14)
plt.legend()
plt.subplot(313)
plt.plot(obs_time_vec_mrp_obs, mrp_estimation_errors_ekf[:,2], '.', color='r', label = 'EKF RMS = ' + str(round(mrp3_errors_rms_ekf,3)))
plt.plot(obs_time_vec_mrp_obs, 3*np.abs(np.sqrt(P_ekf[:,2,2])), '--k')
plt.plot(obs_time_vec_mrp_obs, -3*np.abs(np.sqrt(P_ekf[:,2,2])), '--k')
plt.xlim([0, obs_time_vec_mrp_obs[-1]])
plt.ylim([-0.005,0.005])
#plt.ylim(-50,50)
#plt.axvline(3, color='k',linestyle='--')
plt.ylabel('$\Delta \sigma_3$',size=14)
plt.legend()
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/errors_mrps.png', bbox_inches='tight', dpi=300)
plt.close()


######################## Using Star Tracker Observations #########################
### Filter initial estimate
Pbar_0 = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])**2
Xbar_0 = np.concatenate([np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0])])


getAngularVelocityObservations = lambda t: obsSTGen.getNextObservation(t, 1) # Gyro Observations

# Estimation dynamic model
attitudeModelEstimation_ST = mrpKinematicsModel.getDynamicModel(getAngularVelocityObservations, propagationFunction='F_plus_STM')
attitudeModelEstimation_ST.setNoiseCompensation(True)

ekfP_ST = ekfProc.getFilter(attitudeModelEstimation_ST, starTrackerModelGenerator, Integrator.RK4, attitudeKinematics.switchMRPrepresentation)
ekfP_ST.configureFilter(Xbar_0, Pbar_0, t0)
ekfP_ST.josephFormulation(True)

ekfP_ST.startEKFafter(0)
ekfP_ST.processAllObservations(observations_ST, obs_time_vec_ST, observers_ST, R_noise_ST, dt, Q = Q)
Xhat_ekf_ST = ekfP_ST.getEstimateVector()
xhat_ekf_ST = ekfP_ST.getDeviationEstimateVector()
Xref_ekf_ST = ekfP_ST.getReferenceStateVector()

P_ekf_ST = ekfP_ST.getCovarianceMatrixVector()
prefit_ekfP_ST = ekfP_ST.getPrefitResidualsVector()
postfit_ekfP_ST = ekfP_ST.getPostfitResidualsVector()
stm_ekfP_ST = ekfP_ST.getSTMMatrixfrom0Vector()

ekfP_ST.plotPrefitResiduals(R_noise_ST, ['Star vector body 1', 'Star vector body 2', 'Star vector body 3'], ['', '', ''],['b', 'g', 'r'],'../report/include/prefit_ekf_ST.png')
ekfP_ST.plotPostfitResiduals(R_noise_ST, ['Star vector body 1', 'Star vector body 2', 'Star vector body 3'], ['', '', ''],['b', 'g', 'r'],'../report/include/postfit_ekf_ST.png')


##### Errors
mrp_estimation_errors_ekf_ST = Xhat_ekf_ST[:,0:3] - obs_true_states_ST[:,0:3]

mrp1_errors_rms_ekf_ST = np.sqrt(np.sum(mrp_estimation_errors_ekf_ST[:,0]**2)/nmbr_obs_ST)
mrp2_errors_rms_ekf_ST = np.sqrt(np.sum(mrp_estimation_errors_ekf_ST[:,1]**2)/nmbr_obs_ST)
mrp3_errors_rms_ekf_ST = np.sqrt(np.sum(mrp_estimation_errors_ekf_ST[:,2]**2)/nmbr_obs_ST)

plt.figure()
plt.hold(True)
plt.subplot(311)
plt.plot(obs_time_vec_ST, Xhat_ekf_ST[:,0], '.', color='r', label = 'EKF $\hat\sigma_1$')
plt.plot(obs_time_vec_ST, obs_true_states_ST[:,0],'.', color='g', label = 'True $\sigma_1$')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylabel('$\hat\sigma_1$')
plt.legend()
plt.subplot(312)
plt.plot(obs_time_vec_ST, Xhat_ekf_ST[:,1], '.', color='r', label = 'EKF $\hat\sigma_2$')
plt.plot(obs_time_vec_ST, obs_true_states_ST[:,1],'.', color='g', label = 'True $\sigma_2$')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylabel('$\hat\sigma_2$')
plt.legend()
plt.subplot(313)
plt.plot(obs_time_vec_ST, Xhat_ekf_ST[:,2], '.', color='r', label = 'EKF $\hat\sigma_3$')
plt.plot(obs_time_vec_ST, obs_true_states_ST[:,2],'.', color='g', label = 'True $\sigma_3$')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylabel('$\hat\sigma_3$')
plt.legend()
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/estimated_mrp_ST.png', bbox_inches='tight', dpi=300)
plt.close()


plt.figure()
plt.hold(True)
plt.subplot(311)
plt.plot(obs_time_vec_ST, Xhat_ekf_ST[:,3], '.', color='r', label = 'EKF $\hat \\beta_1$')
plt.axhline(y=initial_gyro_bias[0], color='g', label = 'True $\\beta_1$')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylabel('$\hat \\beta_1$')
plt.legend()
plt.subplot(312)
plt.plot(obs_time_vec_ST, Xhat_ekf_ST[:,4], '.', color='r', label = 'EKF $\hat \\beta_2$')
plt.axhline(y=initial_gyro_bias[1], color='g', label = 'True $\\beta_2$')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylabel('$\hat \\beta_2$')
plt.legend()
plt.subplot(313)
plt.plot(obs_time_vec_ST, Xhat_ekf_ST[:,5], '.', color='r', label = 'EKF $\hat \\beta_3$')
plt.axhline(y=initial_gyro_bias[2], color='g', label = 'True $\\beta_3$')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylabel('$\hat \\beta_3$')
plt.legend()
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/estimated_bias_ST.png', bbox_inches='tight', dpi=300)
plt.close()


plt.figure()
plt.subplot(311)
plt.plot(obs_time_vec_ST, mrp_estimation_errors_ekf_ST[:,0], '.', color='r', label = 'EKF RMS = ' + str(round(mrp1_errors_rms_ekf_ST,3)))
plt.plot(obs_time_vec_ST, 3*np.abs(np.sqrt(P_ekf_ST[:,0,0])), '--k')
plt.plot(obs_time_vec_ST, -3*np.abs(np.sqrt(P_ekf_ST[:,0,0])), '--k')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylim([-0.005,0.005])
#plt.axvline(3, color='k',linestyle='--')
plt.ylabel('$\Delta \sigma_1$')
plt.legend()
plt.subplot(312)
plt.plot(obs_time_vec_ST, mrp_estimation_errors_ekf_ST[:,1], '.', color='r', label = 'EKF RMS = ' + str(round(mrp2_errors_rms_ekf_ST,3)))
plt.plot(obs_time_vec_ST, 3*np.abs(np.sqrt(P_ekf_ST[:,1,1])), '--k')
plt.plot(obs_time_vec_ST, -3*np.abs(np.sqrt(P_ekf_ST[:,1,1])), '--k')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylim([-0.005,0.005])
#plt.axvline(3, color='k',linestyle='--')
plt.ylabel('$\Delta \sigma_2$')
plt.legend()
plt.subplot(313)
plt.plot(obs_time_vec_ST, mrp_estimation_errors_ekf_ST[:,2], '.', color='r', label = 'EKF RMS = ' + str(round(mrp3_errors_rms_ekf_ST,3)))
plt.plot(obs_time_vec_ST, 3*np.abs(np.sqrt(P_ekf_ST[:,2,2])), '--k')
plt.plot(obs_time_vec_ST, -3*np.abs(np.sqrt(P_ekf_ST[:,2,2])), '--k')
plt.xlim([0, obs_time_vec_ST[-1]])
plt.ylim([-0.005,0.005])
#plt.ylim(-50,50)
#plt.axvline(3, color='k',linestyle='--')
plt.ylabel('$\Delta \sigma_3$')
plt.legend()
plt.xlabel('Observation Time $[s]$')
plt.savefig('../report/include/errors_mrps_ST.png', bbox_inches='tight', dpi=300)
plt.close()