'''
Functions to run the 1D finite water content richards by Ogden et al. 2015 and Talbot and Ogden 2008

    Ogden, F.L., Lai, W., Steinke, R.C., Zhu, J., Talbot, C.A., Wilson, J.L. (2015).
    A new general 1-D vadose zone flow solution method.
    Water Resources Research, 51, 4282-4300. doi:10.1002/2015WR017126

    Talbot, C. A., and F. L.Ogden (2008), 
    A method for computing infiltration and redistribution in a discretized moisture content domain, 
    Water Resour. Res., 44, W08453, doi:10.1029/2008WR006815.

    copy from original funcs.py, but adapted to accomodate layered soils. 

'''
import numpy as np

import logging

logger = logging.getLogger(__name__)


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
    calculate h from theta using van Genuchten 1980 (inverting the 
    '''
    Se = effective_water_content(theta, theta_r, theta_e)
    return (1/alpha) * (Se**(-1/m)-1)**(1/n)



######## BIN DISCRETIZATION ######

def create_bins (N, theta_r, theta_e, labda, alpha, n, Ks, h_max, h_min, h_init, dt, tol = 0.0005):
    '''
    create finite volume soil moisture bins with unique hydraulic properties
    '''
    m = 1-1/n
    delta_th = (theta_e - theta_r) / N # <- divide by N to get the delta range for a bin, not the total soil moisture range
    theta_bins = theta_r + (np.arange(1,N + 1)) * delta_th # changed bin descritization, evaluate them at the right side of each bin (Ogden 2015, p4286 4.)

    # calculate hydraulic properties of each bin 
    K_bins = K_unsat(theta_bins, theta_r, theta_e, labda, m, Ks)
    h_bins = h_theta(theta_bins, theta_r, theta_e, alpha, m, n)
    h_bins[N-1] = 0.5*(h_bins[N-1] +h_bins[N-2]) #take the midpoint of the last bin to prevent singularity since h=0 otherwise 
    h_bins = np.minimum(h_bins, h_max)
    h_bins = np.maximum(h_bins, h_min)
    
    #calculate the initial θ values for this soil
    theta_init = max(theta_bins[1],theta_h(h_init,theta_r,theta_e, alpha, m, n))

    bins = {'alpha'      : alpha,
            'theta_r'    : theta_r,
            'theta_e'    : theta_e,
            'n'          : n,
            'm'          : 1-1/n,
            'delta_theta': delta_th,
            'theta_bins' : theta_bins,
            'K_bins'     : K_bins,
            'h_bins'     : h_bins,
            'theta_init' : theta_init
        }

    '''
    calculate the seeding depth to initialise infiltration (Ogden and Talbot, 2008)
    This is needed because infiltration (Ogden eq. 18) wil become singular for depth = 0 
    Thefore the Green-Ampt infiltration equation is used to iteratively determine the infiltration at the start

    F = Ksat * t + psi_f * Delta_theta * ln(1 + F / (psi_f * Delta_theta))

    this only gives the maximum inifltration depth
    Next, the infiltration depth is calulated per bin
    ''' 
    GA_depth = G_A_dry_depth(alpha,m,n,theta_e, theta_r, Ks, dt)
    bins = (T_O_dry_depth(bins, dt, 1e-4, GA_depth))
    return bins
 

def G_A_dry_depth (alpha, m, n, theta_e, theta_r, Ks, dt, tol = 0.005):
    '''
    calculate the maximum dry depth using the GA equatuion;

    F = Ks * dt + Geff * eff_porosity * ln(1 + z_old/Geff) - z_old * eff_porosity

    use newton raphson to find Z for the first timestep
    '''
    Geff = 1/alpha * (0.046*m + 2.07*m**2 + 19.5*m**3)/(1 + 4.7*m + 16*m**2) # from Morel Seytoux 1996


    diff = 1 
    z_old = Geff
    while abs(diff) > tol :
        f = Ks*dt + Geff*(theta_e-theta_r)* np.log(1 + z_old / Geff) - z_old * (theta_e-theta_r)
        f_prime = Geff * (theta_e-theta_r) / (Geff + z_old) - (theta_e-theta_r)
        if(f_prime == 0.0):
            # If f prime is zero, set z to the depth calculated during this step.-
            z_new = Ks * dt / (theta_e-theta_r) + Geff * np.log(1 + Ks * dt / ((theta_e-theta_r)*Geff))
        else:
            z_new = z_old - (f / f_prime) # Newton rapson step
        diff = z_old - z_new
        z_old = z_new
        #print(f'diff = {diff}, z_new = {z_new}')
    return z_new * 10 #from c code, then why use this function at all, it does not limit anything yet

def T_O_dry_depth (bins, dt, minimum_dry_depth, GA_depth, iter_lim = 1000, tol = 0.005):
    '''
    calculate the dry depth to prevent singularity 
    Using the Green-Ampt infiltration depth formula, the maximum infiltration depth is first calculated.

    Then, using the Talbot Ogden depth (Talbot 2008) the initial infiltration depth per bin is calculated. 
    using Newton Raphson
    '''
    k = bins['K_bins']
    h = bins['h_bins']
    delta_theta = bins['delta_theta']
    dry_depth = np.zeros(len(bins['K_bins']))
    for j in range(len(bins['K_bins'])):
        z_old = k[j]* dt
        diff = 10
        iteration = 0
        while abs(diff) > tol and iteration < iter_lim:
            f = (k[j]* dt + h[j] * delta_theta * np.log(1 + (z_old / h[j]))) - z_old*delta_theta
            f_prime = (delta_theta / (1 + (z_old/h[j]))) - delta_theta
            #print(f'f = {f}, f_prime = {f_prime}')
            if f_prime == 0.0 :
                # if the slope is zero, set z_new to the minimum depth to avoid singularity.
                z_new = minimum_dry_depth
                diff = 0.0
            else:
                z_new = z_old - (f / f_prime)
                diff = z_old - z_new
                z_old = z_new
            iteration += 1
            #print(f'iter = {iteration}, diff = {diff}, z_new = {z_new}, bin = {j}') 
        if z_new > GA_depth:
            z_new = GA_depth
        elif z_new < minimum_dry_depth:
            z_new = minimum_dry_depth
        dry_depth[j] = z_new
        bins['dry_depth'] = dry_depth #FIXED cap relax is already taken care of later.
    return bins

def build_layers_by_soil(bofek, soils):
    """
    Parse bofek.csv into a layers_by_soil dict for use in infiltration_layered().

    Each profile (row) has up to 9 layers defined by paired columns:
        isoilN  — soil type ID (references hydraulic property database)
        izN     — bottom depth of layer N [cm]

    Inactive layer slots are marked by isoil=0 and iz=99999 and are dropped.

    Duplicate bodemcode entries: first occurrence is kept 

    """
    

    # Strip whitespace from bodemcode
    bofek['bodemcode'] = bofek['bodemcode'].str.strip()

    # Identify paired layer columns
    n_layers = 9
    isoil_cols = [f'isoil{i}' for i in range(1, n_layers + 1)]
    iz_cols    = [f'iz{i}'    for i in range(1, n_layers + 1)]

    # Keep first occurrence for duplicate bodemcodes
    bofek = bofek.drop_duplicates(subset='bodemcode')

    layers_by_soil = {}

    for _, row in bofek.iterrows():
        name   = row['bodemcode']
        layers = []

        for isoil_col, iz_col in zip(isoil_cols, iz_cols):
            isoil = int(row[isoil_col])
            depth = float(row[iz_col])
            

            # remove the no_value 
            if isoil == 0 or depth == 99999:
                break

            top = layers[-1]['depth'] if layers else 0.0

            layers.append({
                **soils[isoil],
                'depth':   depth,
                'thickness': depth - top
                })

        layers_by_soil[name] = layers

    return layers_by_soil


######## INFILTRATION FUNCTIONS #####
'''
Calculate the infiltration depth using the Equation form Ogden and solve it numerically using Runge-Kutta 4
'''

def RK4 (z, Geff, MoL, head, dt, active = None):
    '''
    Use 4-th order Runge-Kutta to advance the ODE one timestep for 1 bin
    ODE: dz_j/dt = ΅((K(θd)-K(θi)) / (θd-θi) * (1 + (h_j + Hp)/z_j)
    '''

    # calculate for each front infiltration velocity Ogden eq 18   
    def rhs(z):
        dz = MoL * (1 + (head + Geff )/z) # add +Hp to Geff for ponded depth
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
Functions to activate and deactivate bins, depending on rainfall. 
Idea: add ponded depth and rainfall together and remove the water needed to satisfy the waterdemand for 
Theta bins from left to right. If bins do not contain water at the moment, the dry bin depth is used (Talbot and Ogden, 2008)
Only if all theta bins are filled, water is saved as ponded

Still need to think about how to calculate the head from ponded water. 
For now just use the ponded depth from the previous step and add uninfiltrated water at the end to ponded.
'''
def infiltration_capacity(z_fronts, Geff, K_bins, dtheta, N, Hp=0.0):
    """
    no longer used, remove

    Total potential infiltration rate f_p [cm/day]: the rate at which the soil
    can accept water given the current front positions and ponded depth.

    f_p / dtheta = sum_j [ K_j * (z_j + h_j + Hp) / (dtheta * z_j) ]   [Ogden 2015, eq. 18 and integrate/sum according to 13]
    """
    f_p = 0.0
    for j in range(len(z_fronts)):
        if z_fronts[j] > 0.0:
            f_p += K_bins[j] * (z_fronts[j] + Geff + Hp) /  z_fronts[j]
    return f_p

def effective_cap_drive (soil, theta_d):
    '''
    Calculate the value for Geff (Ogden 2015) / psi_f (Rawls&Brakensiek) / HcM (Morel Seytoux 1996).
    This is the Matric suction/ capillary drive from the Green Ampt equation in Ogden (2015) defined as 
    the maximum of |psi(theta_d)| and HcM 
    with theta_d as the theta of the highest theta bin containing water or HcM from Morel Seytoux (1996):
        HcM = 1/alpha * (0.046m + 2.07m^2 + 19.5m^3)/(1 + 4.7m + 16m2)
    '''
    alpha = soil['alpha']
    m = soil['m']
    theta_r = soil['theta_r']
    theta_e = soil['theta_e']
    n = soil['n']

    HcM = 1/alpha * (0.046*m + 2.07*m**2 + 19.5*m**3)/(1 + 4.7*m + 16*m**2)
    psi_d = h_theta(theta_d, theta_r,theta_e, alpha, m, n)
    #logging.info(f'HcM = {HcM}, psi_d = {psi_d}')
    return max(HcM, psi_d)



def handle_infiltration (rainfall_rate, soil, z_fronts, dt, Hp_top, Hp_bot, N):
    '''
    for one time step calculate the infiltration per bin  for one soil layer and substract the infiltration from each bin from the total
    if the infiltration from a bin is more than what is left, function breaks and the infiltration for that bin is the remainder
    '''
    
    rain_sum = rainfall_rate* dt 
    water_available = Hp_top + rain_sum # this thus creates that there is no ponded head effect from rainfall during this timestep

    #calculate the head differnce between the botom and top, which is the force driving infiltration
    head = Hp_top - Hp_bot
    
    # calculate the term (K(θd)-Κ(θi))/(θd-θi) (first term of equation 18 from Ogden)
    # θi is the left most bin were the water extends form surface to max_depth
    try:
        sat_idx = np.max(np.where(z_fronts >= soil['thickness'])[0])
    except ValueError:
        sat_idx = np.abs(soil['theta_bins']-soil['theta_init']).argmin()
    if sat_idx == 0:
        theta_i = soil['theta_r']
    # elif len(sat_idx) == N: # prevent problems when all bins are completetly full
    #     theta_i = bins['theta_bins'][np.max(sat_idx)]
    else:
        theta_i = soil['theta_bins'][sat_idx]


    #calculate the highest active bin, if no bins are active yet, this is the bin just after the fully saturated bins.
    try:
        active_idx = np.max(np.where(z_fronts > 0 ))    # get the array of bins that are active #FIXLATER: check if this works at initialisation.
    except ValueError:
        active_idx = active_idx = sat_idx+1

    theta_d = soil['theta_bins'][active_idx]

    #logging.info(f'active_idx = {active_idx}, sat_idx = {sat_idx}, θd = {theta_d}, θi = {theta_i}')


    #calculate the Method of Lines finite difference form of the partial derivative (Ogden, eq. 17)
    try:
        MoL = ((soil['K_bins'][active_idx] - soil['K_bins'][sat_idx]) 
            /(theta_d - theta_i)).item()
    except ZeroDivisionError:
        logging.warning(f'theta_i = {theta_i} = theta_d = {theta_d}, setting MoL to 0')
        MoL = 0

    #calculate Geff, which is the max of |ψ(θd)| and HcM as calculated by Morel Seytoux
    # FIXLATER should be dependent on the highest active bin form last iter
    Geff = effective_cap_drive(soil, theta_d)

    '''
    print(
            f"theta_i={theta_i}",
            f"theta_d={theta_d}",
            f"MoL={MoL}",
            f"Geff={Geff}"
        )
    '''

    z_new = np.copy(z_fronts)
    K_bins = soil['K_bins']
    delta_theta = float(soil['delta_theta'])

    
    # NEW: Separate calculation of infiltration of fully saturated bins
    #  calculate the demand
    sat_inf = K_bins[sat_idx] * (1 + head/soil['thickness']) *dt #this is saturated infiltration for bins touching GW [cm] # add + Hp/max_depth to add extra head due to ponding
    # caluclate left over water for unsat bins
    water_available -= sat_inf
    max_bin = sat_idx # for printing if all bins are full
    #logger.info(f'max sat = {sat_idx}, sat_inf = {sat_inf}, k_sat = {K_bins[sat_idx]}') 
    

    for j in range(sat_idx + 1, len(z_fronts)):

        #calculate potential front advance if water supply is unlimited
        if z_fronts[j] > 0:
            dz = RK4(z_fronts[j], Geff, MoL, head, dt)
            #print(f'bin {j} already active')
        else:
            dz = soil['dry_depth'][j]
            #print(f'bin {j} activated')
        
        #second calculate the actual infiltration
        demand = dz * delta_theta #this computes the water depth that is potentially infiltrated in this bin and in this timestep
        if demand <= water_available:              
            z_new[j] += dz
            water_available -= demand
            #print(f'dz = {dz}, demand = {demand}')
            max_bin = j
        else:
            #use last bit of available water for infiltration
            #print(f'demand = {demand}, water available = {water_available}')
            z_new[j] += water_available / delta_theta
            dz = demand / delta_theta - water_available / delta_theta #leftover demand for unsatisfied bins
            # if water_available > 0:
                #logger.info(f'no more water avalable for bin {j}, left over dz = {round(dz,2)}')
            water_available = 0
            #print(f'no more water avalable for bin {j}, left over dz = {dz}')
           
            #get water from the bins to the right to satisfy the needs to the left
            #first find the currently last active bin, if none are active, this is the bin id of theta_init
            try:
                last_non_zero = max(np.where(z_new>0)[0])
            except ValueError:
                last_non_zero = -1 #no bins are active infiltration fronts

            while((dz >0) & (last_non_zero > j)):
                if(z_new[last_non_zero]> dz):
                    #if water in the right most bin can satisfy the whole demand of the bin j
                    z_new[j] += dz 
                    z_new[last_non_zero] -= dz
                    dz = 0
                    #logger.info(f'full demand of bin {j} satisfied by bin {last_non_zero}')
                     # to ensure the left over water in the current max bin is also emptied in other bins
                elif(z_new[last_non_zero]>=0):
                    #if water needs to be taken from more than 1 bin, take the remainder of this bin and go one bin to the left
                    z_new[j] += z_new[last_non_zero]
                    dz -= z_new[last_non_zero]
                    z_new[last_non_zero]=0
                    last_non_zero -= 1 #and go one bin to the left
                    #logger.info(f'all water from {last_non_zero} to {j}, demand not satisfied')
                #elif(z_new[last_non_zero] == 0):
                #    last_non_zero = -1 #no water 
                    #logger.info(f'no water left in bin {last_non_zero}, proceeding'
                #print(f'last_non_zero = {last_non_zero}, j = {j}')
            if z_new[j]>0:
                max_bin = j

            
            #print(f'last bin used = {j}')


           
        #print(f'left over water = {water_available} cm, infiltrated water = {demand} cm')
        #print(f'j = {j}, drydepth = {bins['dry_depth'][j]},dz = {dz}, demand = {demand}, K-bins = {bins['K_bins'][j]}')
    front_inf = np.sum((z_new - z_fronts)  * soil['delta_theta'])
    infiltration = (sat_inf + front_inf) #infiltration in mm/day at this time step
    #logger.info(f'sat_inf = {sat_inf}, front_inf = {front_inf}, cum_inf = {infiltration}')
    
    #Track the amount of water leaving the layer and set z_new to the maximum depth of the layer
    z_total_unsat = z_new.copy()
    z_new = np.minimum(z_new, soil['thickness'])
    unsat_exfil = np.sum((z_total_unsat-z_new) * delta_theta)
    exfil_tot = unsat_exfil + sat_inf
    
    #add to the ponded water at the bottom
    Hp_bot += exfil_tot
    #exfil_psi = soil['h_bins'][np.max(np.where(exfil_array>0))]

    try:
        first_bin = np.min(np.where(z_new< soil['thickness']))
    except ValueError:
        first_bin = 0
    front_speed = z_new[first_bin] - z_fronts[first_bin]
    #logger.info(f'Max bin act = {max_bin}, max bin calc= {max_bin_theta}, Hp = {round(Hp,2)}, Geff = {round(Geff,2)}, θi = {theta_i}, θd = {theta_d}, MoL = {round(MoL,2)}')
    return z_new, water_available, infiltration, max_bin,front_speed, Hp_bot

#def layered_soil_handler():



###### CAPILLARY RELAXATION ####

def capillary_relax (z_fronts, max_depth):
    '''
    Use capillary relaxation to prevent the water in the coarser porse to take over 
    the water in the smaller pores - sort only the active bins in descending order


    '''
    mask = np.where((z_fronts > 0) & (z_fronts < max_depth))[0]
    #print(f'mask = {mask}')
    z_fronts[mask] = np.flip(np.sort(z_fronts[mask]))

    return z_fronts


