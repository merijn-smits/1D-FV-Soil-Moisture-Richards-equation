'''
Functions to run the 1D finite water content richards by Ogden et al. 2015

    Ogden, F.L., Lai, W., Steinke, R.C., Zhu, J., Talbot, C.A., Wilson, J.L. (2015).
    A new general 1-D vadose zone flow solution method.
    Water Resources Research, 51, 4282-4300. doi:10.1002/2015WR017126

'''
import numpy as np
import pandas as pd


######## HELPER FUNCTIONS #######
def effective_water_content (theta, theta_r, theta_e) :
    '''
    calculate the effective water content using the residual water content and the effective porosity
    '''
    return (theta - theta_r)/(theta_e - theta_r)


def K_unsat (theta, theta_r, theta_e, labda, m, Ks):
    '''
    calculate the unsaturated hydraulic consuctivity unsing Mualem-vanGenuchten
     could later simplify tghe integration by neglecting the -1 and making it analytically integrable
    '''
    Se = effective_water_content(theta, theta_r, theta_e)

    return np.where(Se>= 1, Ks, Ks * Se ** labda * (1-(1 - Se**(1/m))**m)**2)

def h_theta (theta, theta_r, theta_e, alpha, m, n):
    '''
    calculate h from Se using van Genuchten 1980
    '''
    Se = effective_water_content(theta, theta_r, theta_e)
    return (1/alpha) * (Se**(-1/m)-1)**(1/n)



######## BIN DISCRETIZATION ######

def create_bins (N, theta_r, theta_e, labda, alpha, n, Ks):
    '''
    create bins with unique hydraulic properties
    '''
    m = 1-1/n
    delta_th = (theta_e - theta_r)  
    theta_bins = np.array([(i/N) for i in range(1,N+1)])*delta_th

    # calculate hydraulic properties of each bin
    K_bins = K_unsat(theta_bins, theta_r, theta_e, labda, m, Ks)
    h_bins = h_theta(theta_bins, theta_r, theta_e, alpha, m, n)

    return{
        'delta_theta': delta_th,
        'theta_bins' : theta_bins,
        'K_bins'     : K_bins,
        'h_bins'     : h_bins 
    }

test = create_bins(N = 10, theta_r = 0.02, theta_e = 0.42, labda = 0.4566599312, alpha = 0.021941504,n = 1.772038185,Ks = 25.7389262944)

######## INFILTRATION FUNCTIONS #####
'''
Calculate the infiltration depth using the Equation form Ogden and solve it numerically using Runge-Kutta 4
'''

def suction_head ():
    '''
    Calculate suction head 
    (use just h_bins for now or the possible the Sun et al. formula?)
    '''
    return ()


def infiltration_per_bin (z_j, h_j, K_j, delta_theta, Hp):
    '''
    calculate the infiltration for 1 bin with function

    dz_j/dt = K_j * (z_j + h_j + Hp) / (dtheta * z_j)
    or in alternative form:
    dz_j/dt = K_j/delta_theta * (1 + (h_j + Hp)/z_j)
    which when multiplied by delta theta is the Green-Ampt
    '''
    if z_j <= 0:
        return 0
    return K_j/delta_theta * (1 + (h_j + Hp)/z_j)

def RK4 (z_fronts, h_bins, K_bins, dtheta, Hp, dt, active = None):
    '''
    Use 4-th order Runge-Kutta to advance the ODE one timestep
    '''
    # take the amount of active bins
    N = len(z_fronts)
    if active is None:
            active = z_fronts > 0.0

    # calculate for each front infiltration velocity    
    def rhs(z):
        dz = np.zeros(N)
        for j in range(N):
            if active[j]:
                dz[j] = infiltration_per_bin(z[j], h_bins[j], K_bins[j], dtheta, Hp)
        return dz

    # Calculate the Runge-Kutta steps
    k1 = rhs(z_fronts)
    k2 = rhs(z_fronts + 0.5 * dt * k1)
    k3 = rhs(z_fronts + 0.5 * dt * k2)
    k4 = rhs(z_fronts + dt * k3)

    # Calculate the new depths of the infiltration front in each bin
    z_new = z_fronts + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.maximum(z_new, 0.0)   # fronts cannot be negative

###### STARTING INFILTRATION ####

def seed_depth(K_j, h_j, dtheta, dt):
    """
    Analytical seed depth for bin j at the very first time step (z_j = 0).
    Derived by integrating dz/dt = K_j * h_j / (dtheta * z_j) for small z:
        z_j * dz_j = K_j * h_j / dtheta * dt  =>  z_j = sqrt(2*K_j*h_j*dt/dtheta)
    """
    return max(np.sqrt(2.0 * K_j * h_j * dt / dtheta), 1e-9)

def activate_ponded_bins(z_fronts, h_bins, K_bins, dtheta, dt):
    """
    Seed all inactive bins when ponding first occurs (Hp > 0, or i > f_p).
    Only affects bins with z_j == 0; already active bins are unchanged.
    """
    z_new = z_fronts.copy()
    for j in range(len(z_fronts)):
        if z_fronts[j] == 0.0:
            z_new[j] = seed_depth(K_bins[j], h_bins[j], dtheta, dt)
    return z_new


def potential_infiltration_rate(z_fronts, h_bins, K_bins, dtheta, Hp=0.0):
    """
    Total potential infiltration rate f_p [m/hr]: the rate at which the soil
    can accept water given the current front positions and ponded depth.

    f_p = sum_j [ K_j * (z_j + h_j + Hp) / (dtheta * z_j) ]   [Ogden 2015, eq. 18 and integrate/sum according to 13]

    Only bins with z_j > 0 (active bins) contribute.
    """
    f_p = 0.0
    for j in range(len(z_fronts)):
        if z_fronts[j] > 0.0:
            f_p += K_bins[j] * (z_fronts[j] + h_bins[j] + Hp) / (dtheta * z_fronts[j])
    return f_p


def handle_surface_flux(z_fronts, h_bins, K_bins, dtheta, dt,
                         rainfall_rate, Hp, max_depth=10.0):
    """
    Determine ponding state and advance wetting fronts for one time step.

    Logic (Ogden 2015, Section 2.2):
    1. Compute potential rate f_p from current front depths.
    2a. PONDED  (rainfall >= f_p OR Hp > 0):
          - Activate all inactive bins (seed depths)
          - Advance fronts at full potential rate
          - Excess rainfall accumulates as ponded depth: Hp += (i - f_p)*dt
          - Existing Hp also drains at the actual infiltration rate
    2b. SUB-PONDED (rainfall < f_p AND Hp == 0):
          - Scale front advance so total flux = rainfall rate
          - Approximation: all bins scaled uniformly (Ogden 2015, p. 4286)
          - Hp remains 0

    Parameters
    ----------
    z_fronts     : array (N,), wetting front depths [m]
    h_bins       : array (N,), capillary heads [m]
    K_bins       : array (N,), hydraulic conductivities [m/hr]
    dtheta       : bin width [-]
    dt           : time step [hr]
    rainfall_rate: surface water flux [m/hr]; np.inf for constant ponding
    Hp           : current ponded depth [m]
    max_depth    : simulation domain depth [m]

    Returns
    -------
    z_new    : array (N,), updated wetting front depths
    Hp_new   : updated ponded depth [m]
    f_actual : actual infiltration rate this step [m/hr]
    """
    # --- Existing ponded depth drains first (treat like extra head) ---
    if Hp > 0.0:
        z_fronts = activate_ponded_bins(z_fronts, h_bins, K_bins, dtheta, dt)

    f_p = potential_infiltration_rate(z_fronts, h_bins, K_bins, dtheta, Hp)

    # Constant ponded condition signalled by np.inf
    if np.isinf(rainfall_rate):
        z_new  = RK4(z_fronts, h_bins, K_bins, dtheta, Hp,dt)
        Hp_new = Hp          # maintained externally (constant ponding assumption)
        f_actual = f_p
        z_new = np.minimum(z_new, max_depth)
        return z_new, Hp_new, f_actual

    if rainfall_rate >= f_p:
        # ---- PONDING CASE ----
        # All bins active; advance at full potential rate
        z_fronts = activate_ponded_bins(z_fronts, h_bins, K_bins, dtheta, dt)
        # Recompute f_p after seeding (newly seeded bins now contribute)
        f_p = potential_infiltration_rate(z_fronts, h_bins, K_bins, dtheta, Hp)

        z_new = RK4(z_fronts, h_bins, K_bins, dtheta, dt, Hp)

        # Net change in ponded depth: rain in minus infiltration out
        # If Hp > 0, it also drains at f_p (part of f_actual is from Hp)
        dHp = (rainfall_rate - f_p) * dt
        Hp_new = max(Hp + dHp, 0.0)
        f_actual = f_p

    else:
        # ---- SUB-PONDING CASE ----
        # Scale the effective time step so that total flux = rainfall_rate.
        # This is an approximation: in reality only a subset of bins should
        # be active, but uniform scaling preserves the relative front structure.
        if f_p > 1e-15:
            dt_eff = dt * (rainfall_rate / f_p)
        else:
            dt_eff = 0.0

        active = z_fronts > 0.0
        z_new = RK4(
            z_fronts, h_bins, K_bins, dtheta, dt_eff, Hp=0.0, active=active
        )
        Hp_new   = 0.0
        f_actual = rainfall_rate

    z_new = np.minimum(z_new, max_depth)
    return z_new, Hp_new, f_actual

###### CAPILLARY RELAXATION ####

def capillary_relax (z_fronts):
    '''
    Use capillary relaxation to prevent the water in the coarser porse to take over 
    the water in the smaller pores


    '''
    mask = z_fronts > 0
    z_fronts[mask] =  np.sort(z_fronts[mask])
    z_fronts = np.flip(z_fronts[mask])

            
    return z_fronts


#test
Hp = 0.001
dt = 1/24/60 #in minutes
N = 10
z_fronts = np.zeros(N)+0.01
t_max = 360/24/60 #in minutes

t_steps = int(t_max/dt)
time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])
z_history = np.zeros((t_steps, N))

for i ,t  in enumerate(time_vec):
    z_fronts = RK4(z_fronts, test['h_bins'], test['K_bins'], test['delta_theta'], Hp, dt)
    z_fronts = capillary_relax(z_fronts)
    z_history[i,:] = z_fronts
