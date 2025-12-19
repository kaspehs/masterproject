import numpy as np

def vforce_CF(Cv, Cd, Ca, fhat0, fhat_min, fhat_max, dt, N, rho, U, D, dy,
            ddy, phi_vy, sig_dy_r, sig_ddy_r):
    """
        Simplified script for simulating pure CF motion (y direction), no motion
        in IL (x) direction
    """ 
    try:
        V = np.array([U, -dy, np.zeros(U.size)])                  # velocity vector
        absV = np.sqrt(V[:,0]**2+V[:,1]**2+V[:,2]**2) # check multiplication!   # magnitude of relative velocity
        t1 = V[:,0]/absV
        t2 = V[:,1]/absV
        t3 = V[:,2]/absV
        t = np.array([t1, t2, t3])    # tangent vector
        n = np.array([-t[:,1], t[:,0], np.zeros(U.size)]) # normal vector
    except:
        V = np.array([U, -dy, 0])
        absV = np.sqrt(V[0]**2+V[1]**2+V[2]**2) # check multiplication!   # magnitude of relative velocity
        t1 = V[0]/absV
        t2 = V[1]/absV
        t3 = V[2]/absV
        t = np.array([t1, t2, t3])    # tangent vector
        n = np.array([-t[1], t[0], 0]) # normal vector


    # Establish motions in local coords
    try:
        dy_r = dy*n[:,1]
        ddy_r = ddy*n[:,1]
    except:
        dy_r = dy*n[1]
        ddy_r = ddy*n[1]

    # Update sigmas and means
    sig_dy_r = np.sqrt((N-1)/N*sig_dy_r**2 + 1/N*dy_r**2)

    sig_ddy_r = np.sqrt((N-1)/N*sig_ddy_r**2 + 1/N*ddy_r**2)

    # oscillation phase
    cos_phi_dy = dy_r/(sig_dy_r+np.spacing(1.))
    sin_phi_dy = -ddy_r/(sig_ddy_r+np.spacing(1.))

    phi_dy = np.angle(complex(cos_phi_dy,sin_phi_dy))

    ## syncronize
    # lift/CF phase
    theta = phi_dy-phi_vy
    theta = np.angle(complex(np.cos(theta), np.sin(theta))) # get angle between +-pi

    if theta <= 0:
        fhat = fhat0 + (fhat0-fhat_min)*np.sin(theta)
    else:
        fhat = fhat0 + (fhat_max-fhat0)*np.sin(theta)

    omega_vy = 2*np.pi*fhat*np.sqrt(U**2+dy**2)/D   # total speed sqrt(U^2+dy.^2)

    # update vortex phase
    phi_vy = phi_vy + omega_vy*dt

    ## calculate force:
    Fdy = -0.5*rho*D*Cd*np.sqrt(U**2 + dy**2)*dy                # Morison drag force Fdrag projected to y dir
    Fcv = 0.5*rho*D*Cv*np.sqrt(U**2 + dy**2)*U*np.cos(phi_vy)   # Vortex shedding force Fcv projected to y dir
    Fca = -0.25*rho*Ca*np.pi*D**2*ddy                           # Added mass force Fca projected to y dir

    # Total force in y (CF) direction 
    Fy = Fcv+Fdy+Fca

    return Fy, phi_vy, sig_dy_r, sig_ddy_r, Fca, Fcv, Fdy