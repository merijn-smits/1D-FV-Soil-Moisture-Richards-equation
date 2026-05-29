'''
Functions to run the 1D finite water content richards by Ogden et al. 2015 and Talbot and Ogden 2008

    Ogden, F.L., Lai, W., Steinke, R.C., Zhu, J., Talbot, C.A., Wilson, J.L. (2015).
    A new general 1-D vadose zone flow solution method.
    Water Resources Research, 51, 4282-4300. doi:10.1002/2015WR017126

    Talbot, C. A., and F. L.Ogden (2008), 
    A method for computing infiltration and redistribution in a discretized moisture content domain, 
    Water Resour. Res., 44, W08453, doi:10.1029/2008WR006815.

'''
import numpy as np


######## HELPER FUNCTIONS #######
def effective_water_content (theta, theta_r, theta_e) :
    '''
    calculate the effective water content using the residual water content and the effective porosity
    apply a clip to prevent dividing by 0
    '''
    return np.clip((theta - theta_r)/(theta_e - theta_r),1e-5,1)


def K_unsat (theta, theta_r, theta_e, labda, m, Ks):
    '''
    calculate the unsaturated hydraulic consuctivity unsing Mualem-vanGenuchten
    '''
    Se = effective_water_content(theta, theta_r, theta_e)

    return np.where(Se>= 1, Ks, Ks * Se ** labda * (1-(1 - Se**(1/m))**m)**2)



def theta_h (h, theta_r, theta_e, alpha, m, n):
    '''
    calculating theta from h using van Genuchten 1980 and substitutign eq.2 in eq.3
    '''
    return theta_r+(theta_e - theta_r)*(1/(1+(alpha*h)**n))**m


def h_theta (theta, theta_r, theta_e, alpha, m, n):
    '''
    calculate h from Se using van Genuchten 1980 (inverting the formula)
    '''
    Se = effective_water_content(theta, theta_r, theta_e)
    return (1/alpha) * (Se**(-1/m)-1)**(1/n)



######## BIN DISCRETIZATION ######

def create_bins (N, theta_r, theta_e, labda, alpha, n, Ks, h_max, h_min, dt, tol = 0.0005):
    '''
    create finite volume soil moisture bins with unique hydraulic properties
    '''
    m = 1-1/n
    delta_th = (theta_e - theta_r) / N # <- divide by N to get the delta range for a bin, not the total soil moisture range
    theta_bins = theta_r + (np.arange(1,N + 1)) * delta_th # changed bin descritization, evaluate them at the right side of each bin (Ogden 2015, p4286 4.)

    # calculate hydraulic properties of each bin 
    K_bins = K_unsat(theta_bins, theta_r, theta_e, labda, m, Ks)
    h_bins = h_theta(theta_bins, theta_r, theta_e, alpha, m, n)
    h_bins = np.minimum(h_bins, h_max)
    h_bins = np.maximum(h_bins, h_min)

    '''
    calculate the seeding depth to initialise infiltration (Ogden and Talbot, 2008)
    This is needed because infiltration (Ogden eq. 18) wil become singular for depth = 0 
    Thefore the Green-Ampt cumulative infiltration equation is used to iteratively determine the infiltration

    F = Ksat * t + psi_f * Delta_theta * ln(1 + F / (psi_f * Delta_theta))
    ''' 

    Geff = 1/alpha * (0.046*m + 2.07*m**2 + 19.5*m**3)/(1 + 4.7*m + 16*m**2) # from Morel Seytoux 1996

    cumK = np.cumsum(K_bins) #use this from c-code


    dry_bins = np.zeros(len(K_bins))
    for j, k in enumerate(K_bins) :
        diff = 1 
        F = 10
        while abs(diff) > tol :
            F_new = k*dt + Geff*(theta_e-theta_r)* np.log(1 + F/ (Geff*(theta_e-theta_r)))
            diff = F - F_new
            F = F_new
        dry_bins[-(j+1)] = F #as a capillary relaxation after the 'initial' infiltration step, no clue if this is valid?
    


    return{
        'delta_theta': delta_th,
        'theta_bins' : theta_bins,
        'K_bins'     : K_bins,
        'h_bins'     : h_bins,
        'dry_depth'  : dry_bins,
        'cum_K'      : cumK
    }


######## INFILTRATION FUNCTIONS #####
'''
Calculate the infiltration depth using the Equation form Ogden and solve it numerically using Runge-Kutta 4
'''

def RK4 (z, Geff, MoL, Hp, dt, active = None):
    '''
    Use 4-th order Runge-Kutta to advance the ODE one timestep for 1 bin
    ODE: dz_j/dt = ΅((K(θd)-K(θi)) / (θd-θi) * (1 + (h_j + Hp)/z_j)
    '''

    # calculate for each front infiltration velocity Ogden eq 18   
    def rhs(z):
        dz = MoL * (1 + (Geff + Hp)/z)
        return dz

    # Calculate the Runge-Kutta steps
    k1 = rhs(z)
    k2 = rhs(z + 0.5 * dt * k1)
    k3 = rhs(z + 0.5 * dt * k2)
    k4 = rhs(z + dt * k3)

    # Calculate the front advance
    dz = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.maximum(dz, 0.0)   # fronts only go down



####### BIN ACTIVATION #####
'''
New functions to activate and deactivate bins, depending on rainfall. 
Idea: add ponded depth and rainfall together and remove the water needed to satisfy the waterdemand for 
Theta bins from left to right. If bins do not contain water at the moment, the dry bin depth is used (Talbot and Ogden, 2008)
Only if all theta bins are filled, water is saved as ponded

Still need to think about how to calculate the head from ponded water. 
For now just use the ponded depth fromk the previous step and add uninfiltrated water at the end to ponded.
'''
def infiltration_capacity(z_fronts, Geff, K_bins, dtheta, N, Hp=0.0):
    """
    Total potential infiltration rate f_p [cm/day]: the rate at which the soil
    can accept water given the current front positions and ponded depth.

    f_p / dtheta = sum_j [ K_j * (z_j + h_j + Hp) / (dtheta * z_j) ]   [Ogden 2015, eq. 18 and integrate/sum according to 13]
    """
    f_p = 0.0
    for j in range(len(z_fronts)):
        if z_fronts[j] > 0.0:
            f_p += K_bins[j] * (z_fronts[j] + Geff + Hp) /  z_fronts[j]
    return f_p

def effective_cap_drive (alpha, m, theta_d, theta_r, theta_e, n):
    '''
    Calculate the value for Geff (Ogden 2015) / psi_f (Rawls&Brakensiek) / HcM (Morel Seytoux 1996).
    This is the Matric suction/ capillary drive from the Green Ampt equation in Ogden (2015) defined as 
    the maximum of |psi(theta_d)| and HcM 
    with theta_d as the theta of the highest theta bin containing water or HcM from Morel Seytoux (1996):
        HcM = 1/alpha * (0.046m + 2.07m^2 + 19.5m^3)/(1 + 4.7m + 16m2)
    '''
    HcM = 1/alpha * (0.046*m + 2.07*m**2 + 19.5*m**3)/(1 + 4.7*m + 16*m**2)
    psi_d = h_theta(theta_d, theta_r,theta_e, alpha, m, n)
    print(f'Hcm = {HcM}, psi_d = {psi_d}')
    return max(HcM, psi_d)



def handle_infiltration (rainfall_rate, bins, z_fronts, 
                        alpha, m, n, theta_r, theta_e,
                        dt, Hp, max_depth):
    '''
    for one time step calculate the infiltration per bin and substract the infiltration from each bin from the total
    if the infiltration from a bin is more than what is left, function breaks and the infiltration for that bin is the remainder
    '''
    
    rain_sum = rainfall_rate* dt 
    water_available = Hp + rain_sum # this thus creates that there is no ponded head effect from rainfall during this timestep

    # calculate the term (K(θd)-Κ(θi))/(θd-θi) (first term of equation 18 from Ogden)
    # θi is the bin were the water extends form surface to max_depth
    sat_idx = np.where(z_fronts >= max_depth)[0]
    if len(sat_idx) == 0:
        theta_i = theta_init
    else:
        theta_i = bins['theta_bins'][sat_idx[-1]]
    #theta_i = max(bins['theta_bins'][np.sum(z_fronts == max_depth)-2], theta_r)# -2 because the initial is assumed at the left edge of the bin (c)

    # θd is the right most bin containing water
    active_idx = np.where(z_fronts > 0)[0]
    if len(active_idx):
        theta_d = bins['theta_bins'][active_idx[-1]]
    else:
        theta_d = theta_i
    #theta_d = bins['theta_bins'][np.sum(z_fronts != 0 )-1]
    #calculate the Method of Lines finite difference form of the partial derivative (Ogden, eq. 17)
    #this is the same for all bins
    MoL = ((bins['cum_K'][bins['theta_bins']==theta_d] - bins['cum_K'][bins['theta_bins']==theta_i])
            /
           (theta_d - theta_i)).item()

    #calculate Geff, which is the max of |ψ(θd)| and HcM as calculated by Morel Seytoux
    Geff = effective_cap_drive(alpha, m, theta_d, theta_r, theta_e, n)
    print(
            f"theta_i={theta_i}",
            f"theta_d={theta_d}",
            f"MoL={MoL}",
            f"Geff={Geff}"
        )

    z_new = np.copy(z_fronts)
    K_bins = bins['K_bins']
    delta_theta = float(bins['delta_theta'])

    '''
    calculate the increase in front depth (dz) according to Ogden 2015 par. 3.7 and Eq 18

    '''
    while water_available > 1e-5:   #will not be exactly 0 because of numerics
        for j in range(len(z_fronts)):
            if z_fronts[j] >= max_depth:
                dz = K_bins[j] *dt/delta_theta #this is and front advancement [cm]
                print(f'bin {j} saturated')
            elif z_fronts[j] > 0:
                dz = RK4(z_fronts[j], Geff, MoL, Hp, dt)
                print(f'bin {j} already active')
            else:
                dz = RK4(bins['dry_depth'][j], Geff, MoL, Hp, dt)
                print(f'bin {j} activated')
            
            demand = dz * delta_theta #this computes the water depth that is infiltrated in this bin and in this timestep
            if demand <= water_available:              
                z_new[j] += dz
                water_available -= demand
            else:
                z_new[j] += water_available / delta_theta
                water_available = 0
                break            
            #print(f'left over water = {water_available} cm, infiltrated water = {demand} cm')
            #print(f'j = {j}, drydepth = {bins['dry_depth'][j]},dz = {dz}, demand = {demand}, K-bins = {bins['K_bins'][j]}')
    return z_new, water_available 
        





###### INFILTRATION INITIALISATION ####

'''
these functions define what type of infiltration is used (ponded or non-ponded) and 
activate the infiltration type depending on the rainfall or the ponding
this will allow variable input for Hp or rainfall instead of a predefined e.g. Hp = 0.0001

These will be replaced by the section above


'''

def RK4_all_fronts (z_fronts, Geff, K_bins, dtheta, Hp, dt, active = None):
    '''
    Use 4-th order Runge-Kutta to advance the ODE one timestep (old)
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
                dz[j] = infiltration_per_bin(z[j], Geff, K_bins[j], dtheta, Hp)
        return dz

    # Calculate the Runge-Kutta steps
    k1 = rhs(z_fronts)
    k2 = rhs(z_fronts + 0.5 * dt * k1)
    k3 = rhs(z_fronts + 0.5 * dt * k2)
    k4 = rhs(z_fronts + dt * k3)

    # Calculate the new depths of the infiltration front in each bin
    z_new = z_fronts + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.maximum(z_new, 0.0)   # fronts cannot be negative



def seed_depth(K_j, h_j, dtheta, dt):
    """
    Analytical seed depth for bin j at the very first time step (z_j = 0).
    Derived by integrating dz/dt = K_j * h_j / (dtheta * z_j) for small z:
        z_j * dz_j = K_j * h_j / dtheta * dt  =>  z_j = sqrt(2*K_j*h_j*dt/dtheta)
    """
    depth2 = (2.0 * K_j * h_j * dt / dtheta)
    return np.sqrt(depth2) if depth2 > 0 else 0.0

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


def infiltration_capacity(z_fronts, Geff, K_bins, dtheta, N, Hp=0.0):
    """
    Total potential infiltration rate f_p [cm/day]: the rate at which the soil
    can accept water given the current front positions and ponded depth.

    f_p / dtheta = sum_j [ K_j * (z_j + h_j + Hp) / (dtheta * z_j) ]   [Ogden 2015, eq. 18 and integrate/sum according to 13]
    """
    f_p = 0.0
    for j in range(len(z_fronts)):
        if z_fronts[j] > 0.0:
            f_p += K_bins[j] * (z_fronts[j] + Geff + Hp) /  z_fronts[j]
    return f_p
    

def invert_K(i, theta_r, theta_e, Ks, m, labda, tol=1e-3, N = 100):
    """
    Find theta_surf such that K(theta_surf) = i via bisection.
    Valid for 0 < i <= Ks. Returns theta_r if i <= 0, theta_e if i >= Ks.

    Comes from the idea that for unponded infiltration, conductivity = infiltration
    i.e K(theta) = i
    theta is a specific soil moisture which describes which pores are filled at a specific infiltration rate.

    the active theta bins are determined from this.
    """
    if i <= 0.0:
        return theta_r
    if i >= Ks:
        return theta_e

    lo, hi = theta_r + 1e-6, theta_e - 1e-6
    for _ in range(N):                         
        mid = 0.5 * (lo + hi)
        if K_unsat(mid, theta_r, theta_e, labda, m, Ks) < i:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return 0.5 * (lo + hi)

def handle_surface_flux(rainfall_rate, Hp, theta_r, theta_e, m, Ks, labda, dt, bins,
                         z_fronts, alpha, n, max_depth=10.0):
    """
    Determine ponding state and advance wetting fronts for one time step.

    1. Compute potential rate f_p from current front depths.
    2a. Ponded  (rainfall >= f_p OR Hp > 0):
          - Activate all inactive bins (seed depths)
          - Advance fronts at full potential rate
          - Excess rainfall accumulates as ponded depth: Hp += (i - f_p)*dt
          - Existing Hp also drains at the actual infiltration rate
    2b. Un-ponded (rainfall < f_p AND Hp == 0):
          - Scale front advance so total flux = rainfall rate
          - The activate bins are calculated with the asumption that infiltration is precipitation and infiltration is hydraulic conductivity
            therefore P = K(theta). the chracteristic theta can be calculated with K^-1(P) = theta
          - Hp remains 0

    Parameters
    ----------
    z_fronts     : array (N,), wetting front depths [m]
    h_bins       : array (N,), capillary heads [m]
    K_bins       : array (N,), hydraulic conductivities [m/hr]
    delta_theta       : bin width [-]
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
    # # --- Existing ponded depth drains first (treat like extra head) ---
    # if Hp > 0.0:
    #     z_fronts = activate_ponded_bins(z_fronts, Geff, bins['K_bins'], bins['delta_theta'], dt)

    #calculate matric suction and the potential infiltration for the timestep.
    #first calculate theta_d: the right most active bin 
    theta_d = bins['theta_bins'][np.sum(z_fronts != 0 )-1]
    Geff = suction_head(alpha,m,theta_d,theta_r,theta_e,n)
    f_p = potential_infiltration_rate(z_fronts, Geff , bins['K_bins'], bins['delta_theta'], Hp)

    # Constant ponded condition signalled by np.inf
    if np.isinf(rainfall_rate):
        z_new  = RK4(z_fronts, Geff, bins['K_bins'], bins['delta_theta'], Hp,dt)
        Hp_new = Hp          # maintained externally (constant ponding assumption)
        f_actual = f_p
        z_new = np.minimum(z_new, max_depth)
        return z_new, Hp_new, f_actual
  
    # Ponded infiltration
    if rainfall_rate >= f_p or Hp > 0:
        print(f"Ponded: Hp={Hp:.4f}, f_p={f_p:.4f}, rainfall={rainfall_rate}")
    
        # Activate bins if transitioning to ponded
        if np.any(z_fronts == 0.0):
            z_fronts = activate_ponded_bins(z_fronts, bins['h_bins'], bins['K_bins'], bins['delta_theta'], dt)
            print (z_fronts)
        
        # compute f_p after initial seeding
        theta_d = bins['theta_bins'][np.sum(z_fronts != 0)-1]
        Geff = suction_head(alpha,m,theta_d,theta_r,theta_e,n)
        f_p = potential_infiltration_rate(z_fronts, Geff, bins['K_bins'], bins['delta_theta'], Hp)
        print (f"f_p = {f_p}")
        # advance the fronts for one time step 
        z_new = RK4(z_fronts, Geff, bins['K_bins'], bins['delta_theta'], Hp, dt)

        #calculate the amount of infiltrated water
        f_actual = bins['delta_theta'] * np.sum(z_new - z_fronts) / dt
        print(f'f_actual = {f_actual} cm')


        # Net change in ponded depth: rain in minus infiltration out
        # If Hp > 0, it also drains at f_p (part of f_actual is from Hp)
        dHp = (rainfall_rate - f_p) * dt
        Hp_new =  max(Hp + dHp, 0.0)
        return z_new, Hp_new, f_actual

    # Non-ponded infiltration    
    else:
        print(f"Non-ponded: Hp={Hp:.4f}, f_p={f_p:.4f}, rainfall={rainfall_rate}")
        #calulate the largest bin that is active
        theta_surf = invert_K(rainfall_rate, theta_r, theta_e, Ks, m, labda)

        #get the index from the closest theta bin
        idx = (np.abs(bins['theta_bins'] - theta_surf)).argmin()

        #set the infiltration depth for inactive front to zero 
        z_fronts[idx + 1:] = 0

        # Seed any newly active bins that haven't started yet
        for j in range(idx + 1):
            if z_fronts[j] == 0.0:
                z_fronts[j] = seed_depth(
                    bins['K_bins'][j], bins['h_bins'][j], bins['delta_theta'], dt
                )

        # Recompute Geff with updated active set
        theta_d = bins['theta_bins'][np.sum(z_fronts != 0)-1]
        Geff = suction_head(alpha, m, theta_d, theta_r, theta_e, n)


        active = z_fronts > 0.0
        z_new = RK4(
            z_fronts, Geff, bins['K_bins'], bins['delta_theta'], Hp, dt, active=active
        )
        Hp_new   = 0.0
        f_actual = rainfall_rate
        return z_new, Hp_new, f_actual

    z_new = np.minimum(z_new, max_depth)
    return z_new, Hp_new, f_actual



###### CAPILLARY RELAXATION ####

def capillary_relax (z_fronts):
    '''
    Use capillary relaxation to prevent the water in the coarser porse to take over 
    the water in the smaller pores - sort only the active bins in descending order


    '''
    mask = z_fronts > 0
    if np.any(mask):
       
        z_fronts[mask] = np.sort(z_fronts[mask])[::-1]

    return z_fronts



