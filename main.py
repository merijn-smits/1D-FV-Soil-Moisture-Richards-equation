import funcs
import numpy as np

#Soil params, here Staringreeks B01 (fijn zand) all in [cm/day]
theta_r = 0.02
theta_e = 0.427
Ks = 31.23
n = 1.735
m = 1-1/n
labda = 0.981
alpha = 0.0217

#Soil params, here Staringreeks B10 (lichte klei)
theta_r = 0.01
theta_e = 0.448
Ks = 3.83
n = 1.135
m = 1-1/n
labda = 4.581
alpha = 0.0128

#test
h_max = 17000  # maximum matric suction [cm] (16000 is wilting point)
t_steps = 300
#dt = 1/24/60 #in [minutes]
N = 100
t_max = 360  *1/24/60 #in [minutes]
max_depth = 100 #maximum modeling depth in [cm]
rainfall_rate = 10
dt = t_max/t_steps
time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])
z_history = np.zeros((t_steps, 100))
test_bins = funcs.create_bins(N , theta_r, theta_e, labda, alpha,n ,Ks, h_max)
#init

Hp = 0.1  #initial ponding depth [cm]
theta_init = 0.15
idx = (np.abs(test_bins['theta_bins'] - theta_init)).argmin()
z_fronts = np.zeros(N)     # remove +0.01 when infiltration initialisation is added
z_fronts[:idx] = max_depth #set the bins below theta_init active


for i ,t  in enumerate(time_vec):
    z_fronts  = funcs.RK4(z_fronts, test_bins['h_bins'], test_bins['K_bins'], test_bins['delta_theta'], Hp, dt)
    # z_fronts, Hp, f_actual = funcs.handle_surface_flux(
    #                             rainfall_rate, Hp, theta_r, theta_e, m, Ks, labda, dt, 
    #                             test_bins, z_fronts= z_fronts)
    z_fronts = funcs.capillary_relax(z_fronts)
    z_history[i,:] = z_fronts

z_fronts, Hp, f_actual = funcs.handle_surface_flux(
                            rainfall_rate, Hp, theta_r, theta_e, m, Ks, labda, dt, 
                            test_bins, z_fronts, alpha, n)

z_fronts = funcs.capillary_relax(z_fronts)
        
z_fronts  = funcs.RK4(z_fronts, test_bins['h_bins'], test_bins['K_bins'], test_bins['delta_theta'], Hp, dt)



f_p = funcs.potential_infiltration_rate(z_fronts, test_bins['h_bins'], test_bins['K_bins'], test_bins['delta_theta'], Hp)

#calulate the largest bin that is active
theta_surf = funcs.invert_K(rainfall_rate, theta_r, theta_e, Ks, m, labda)

#get the index from the closest theta bin
theta_bins = test['theta_bins']
idx = (np.abs(theta_bins - theta_surf)).argmin()

sliced = {
    key: value[:idx]
    for key, value in test.items()
    if isinstance(value, np.ndarray)
}


active = z_fronts > 0.0
z_new = RK4(
    z_fronts, h_bins, K_bins, dtheta, dt, Hp=0.0, active=active
)
Hp_new   = 0.0
f_actual = rainfall_rate