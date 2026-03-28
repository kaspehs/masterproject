import numpy as np
import matplotlib.pyplot as plt
from utils import vforce_CF
#import imp
#imp.reload(utils)

"""
    Simulating...
"""

# Case input
M = 16.79           # mass kg
C = 1.0e-4            # structural damping
K = 1218            # stiffness N/m
rho = 1000        # fluid density (kg/m3)
U = 0.65           # flow speed (m/s)
D = 0.1          # diameter of the cylinder (m)
nsteps = 200     # number of timesteps per cycle
n_memory = 500   # number of timesteps for calculation of instantaneous velocity

# empirical force coefficients in TD model
Cv = 1.2    # vortex shedding force coefficient in time domain model (-)
Cd = 1.2    # drag coefficient in time domain model (-)
Ca = 1.0    # added mass coefficient in still water (-)

# Synchronization model parameters
fhat0 = 0.144              # center of the synchronization in terms of fhat (-)
fhat_min = 0.08     # lower normalized frequency limit
fhat_max = 0.206 # higher normalized frequency limit

# 
dt = 0.001                # time step
T = 50                  # 
N = int(np.ceil(T/dt))

# preallocate space
time = np.zeros(N)   # time
y = np.zeros(N)      # displacement in y dir (CF)
dy = np.zeros(N)     # velocity
ddy = np.zeros(N)    # acceleration
Fy = np.zeros(N)    # Hydrodynamic force in y dir (CF)
Fcv = np.zeros(N)    # Hydrodynamic force in y dir (CF)
Fdy = np.zeros(N)    # Hydrodynamic force in y dir (CF)
Fca = np.zeros(N)    # Hydrodynamic force in y dir (CF)

phi_vy = np.zeros(N)                # phase of vortex shedding force
phi_vy[0] = 2*np.pi*np.random.rand(1)   # random initial value
sig_dy_loc = np.zeros(N)            # to calculate instantaneous phase of cylinder velocity
mean_dy_loc = np.zeros(N)
sig_ddy_loc = np.zeros(N)
mean_ddy_loc = np.zeros(N)

# Initial conditions:
A = 1.0*D
fhat = 0.17
omega_osc = 2*np.pi*fhat*U/D
Tosc = 2*np.pi/omega_osc
y[0] = A*np.sin(omega_osc*time[0])
dy[0] = omega_osc*A*np.cos(omega_osc*time[0])
ddy[0] = -omega_osc**2*A*np.sin(omega_osc*time[0])

# Simulate dynamics and calculate forces by TD model
for i in range(N-1):
    time[i] = i*dt

    Fy[i+1], phi_vy[i+1], sig_dy_loc[i+1], sig_ddy_loc[i+1], \
        Fca[i+1], Fcv[i+1], Fdy[i+1]= \
        vforce_CF(Cv,Cd,Ca,fhat0,fhat_min,fhat_max,dt,n_memory, rho, U, D, dy[i], \
        ddy[i], phi_vy[i], sig_dy_loc[i], sig_ddy_loc[i])
    
    y[i+1] = y[i] + dt*dy[i]
    dy[i+1] = dy[i] + dt*ddy[i] # dt/M*(-C*dy[i]-K*y[i]+Fy[i])
    ddy[i+1] = 1/M*(-C*dy[i+1]-K*y[i+1]+Fy[i+1])
#Fy[1]=[]

# # take the 100 last T
# Fy = Fy[i-int(np.floor(100*Tosc/dt)):i]  # obtained hydrodynamic force in CF (y) direction
# dy = dy[i-int(np.floor(100*Tosc/dt)):i]
# ddy = ddy[i-int(np.floor(100*Tosc/dt)):i]
# time = time[i-int(np.floor(100*Tosc/dt)):i]
# Fy = Fy[:15000]  # obtained hydrodynamic force in CF (y) direction
# y = y[:15000]
# dy = dy[:15000]
# ddy = ddy[:15000]
# time = time[:15000]


fig = plt.figure(figsize=(7,4))
plt.plot(time, Fy, label='Force (N)')
plt.plot(time, y*100, label=r'Displacement $\times 10^2$ (m)')
# plt.xlim([12, 14])
plt.title('Cross-flow force and displacement')
plt.ylabel('Simulation')
plt.xlabel('time (sec)')
plt.legend()
plt.show()

fig = plt.figure(figsize=(7,4))
plt.plot(time, y, label=r'Displacement (m)')
# plt.xlim([12, 14])
plt.title('Cross-flow force and displacement')
plt.ylabel('Simulation')
plt.xlabel('time (sec)')
plt.legend()
plt.show()

fig = plt.figure(figsize=(7,4))
plt.plot(time, Fy, label='Force (N)')
plt.plot(time, Fca, label='Fca (N)')
plt.plot(time, Fcv, label='Fcv (N)')
plt.plot(time, Fdy, label='Fd (N)')
plt.xlim([12, 14])
plt.title('Cross-flow force and displacement')
plt.ylabel('Simulation')
plt.xlabel('time (sec)')
plt.legend()
plt.show()

fig = plt.figure(figsize=(7,4))
plt.plot(time, Fy, label='Force (N)')
plt.plot(time, Fcv+Fdy, label='Fcv+Fdy (N)')
plt.plot(time, Fca, label='Fca (N)')
plt.xlim([12, 14])
plt.title('Cross-flow force and displacement')
plt.ylabel('Simulation')
plt.xlabel('time (sec)')
plt.legend()
plt.show()

fig = plt.figure(figsize=(7,4))
plt.plot(time, dy*100, label='velox10 (m/s)')
plt.plot(time, Fcv+Fdy, label='Fcv+Fdy (N)')
plt.plot(time, Fca, label='Fca (N)')
plt.xlim([12, 14])
plt.title('Cross-flow force and displacement')
plt.ylabel('Simulation')
plt.xlabel('time (sec)')
plt.legend()
plt.show()


fig = plt.figure(figsize=(7,4))
plt.plot(time, dy*100, label='velox10 (m/s)')
plt.plot(time, Fy, label='Fy (N)')
plt.xlim([12, 14])
plt.title('Cross-flow force and displacement')
plt.ylabel('Simulation')
plt.xlabel('time (sec)')
plt.legend()
plt.show()

# Derive coefficients:

# Excitation coefficient 
Cy = Fy/(0.5*rho*D*U**2)                     # normalize force (force x velocity)
CLv = 2*np.mean(Cy*np.cos(omega_osc*time)) # check multiplication!  # time averaged excitation coefficient

# Added mass coefficient 
Cya = Fy/(0.25*np.pi*D**2*rho*omega_osc**2*A)    # normalize force
CLa = 2*np.mean(Cya*np.sin(omega_osc*time)) # check multiplication # time-averaged added mass coefficient (force x acceleration)

# visualize synchronization curve
theta_data = np.arange(-np.pi, np.pi, 0.01)  # Phase difference btw cylinder velocity and vortex shedding force Fcv
fhat_data = np.zeros(theta_data.size)

for i in range (theta_data.size): #c check size vs. length!
    theta = theta_data[i]
    if theta <= 0:
        fhat_data[i] = fhat0+(fhat0-fhat_min)*np.sin(theta)
    else:
        fhat_data[i] = fhat0+(fhat_max-fhat0)*np.sin(theta)

fig = plt.figure(figsize=(7,4))
plt.plot(theta_data, fhat_data, '-k')
plt.ylabel('Normalized frequency fhat')
plt.xlabel(r'CF phase $\theta$ btw cylinder velocity and vortex shedding force Fcv')
plt.show()