import funcs
import numpy as np


#test
Hp = 0.01
dt = 1/24/60 #in minutes
N = 50
z_fronts = np.zeros(N)+0.001     # remove +0.01 when infiltration initialisation is added
t_max = 360/24/60 #in minutes
rainfall_rate = 1

#Soil params, here Staringreeks B01
theta_r = 0.01
theta_e = 0.42
Ks = 25
m = 0.42
n = 1.772038185
labda = 0.981
alpha = 0.021941504

#init
t_steps = int(t_max/dt)
time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])
z_history = np.zeros((t_steps, N))
test_bins = funcs.create_bins(N , theta_r, theta_e, labda, alpha,n ,Ks )


for i ,t  in enumerate(time_vec):
    z_fronts  = funcs.RK4(z_fronts, test_bins['h_bins'], test_bins['K_bins'], test_bins['delta_theta'], Hp, dt)
    # z_fronts, Hp, f_actual = funcs.handle_surface_flux(
    #                             rainfall_rate, Hp, theta_r, theta_e, m, Ks, labda, dt, 
    #                             test_bins, z_fronts= z_fronts)
    z_fronts = funcs.capillary_relax(z_fronts)
    #z_history[i,:] = z_fronts


                


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