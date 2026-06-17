from datetime import datetime
import funcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import logging

#Soil params, here Staringreeks B01 (fijn zand) all in [cm/day]
theta_r = 0.02
theta_e = 0.427
Ks = 31.23
n = 1.735
m = 1-1/n
labda = 0.981
alpha = 0.0217
'''
#Soil params, here Staringreeks B18 Kleiig veen
theta_r = 0.0
theta_e = 0.765
Ks = 13.14
n = 1.151
m = 1-1/n
labda = 0
alpha = 0.0205

#Soil params, here Staringreeks B10 (lichte klei)
theta_r = 0.01
theta_e = 0.448
Ks = 3.83
n = 1.135
m = 1-1/n
labda = 4.581
alpha = 0.0128

'''

#settings
h_max = 17000  # maximum matric suction [cm] (16000 is wilting point)
h_min = 0.001   # minimum matricsuction [cm] to prevent numerical issues
t_steps = 900
#dt = 1/24/60 #in [minutes]
N = 100
t_max = 240  *1/24/60 #in [minutes]
max_depth = 20 #maximum modeling depth in [cm]
dt = t_max/t_steps
print(dt * 60*24*60) # print timestep in seconds
time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])
bins = funcs.create_bins(N , theta_r, theta_e, labda, alpha,n ,Ks, h_max, h_min, dt)


#set rainfall
rainfall_rate = 100 #* 24/10 #[mm/hr] to [cm/day]
rain_end = 240 * 1/24/60 #[minutes] to [days]
rain_vec = np.zeros(t_steps)
rain_vec[:np.where(time_vec == (rain_end/t_max))[0][0]] = rainfall_rate
cum_rain = t_max * rainfall_rate #FIXLATER change to accomodate rain_vec

#set up message logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig (filename = 'messages.log',
                        filemode = 'w',
                        encoding = 'utf-8',
                        format = '{levelname}:{name}:{message}',
                        style = '{',
                        level = logging.INFO)
logging.info(f'time = {datetime.now()}')

Hp = 0 #initial ponding depth [cm]
theta_init = 0.2
idx = (np.abs(bins['theta_bins'] - theta_init)).argmin()
z_fronts = np.zeros(N)     # remove +0.01 when infiltration initialisation is added
z_history = np.zeros((t_steps, N))
cum_inf = np.zeros(t_steps)
z_fronts[:idx] = max_depth #set the bins below theta_init active
z_init = np.sum(z_fronts * bins['delta_theta'])
max_bin_list = []
frontspeed_list =[]
Hp_list = []
'''
# one by one manual loop
z_fronts, Hp, infiltration = funcs.handle_infiltration (rainfall_rate, bins, z_fronts, 
                                        alpha, m, n, theta_r, theta_e,
                                        dt, Hp, max_depth, theta_init, N)

# deactive_fronts = np.where((z_old == z_fronts)  & (z_old != max_depth))[0]
# slugs = funcs.init_detach_slugs(z_fronts)
# slugs = advance_slugs()
z_fronts = funcs.capillary_relax(z_fronts, max_depth)
#z_old = z_fronts
'''
#automatic loop
for i ,t  in enumerate(time_vec):
    logging.info(f't = {i}, rain = {rain_vec[i]}')
    z_fronts, Hp, infiltration, max_bin, frontspeed = funcs.handle_infiltration (rain_vec[i], bins, z_fronts, 
                                            alpha, m, n, theta_r, theta_e,
                                            dt, Hp, max_depth, theta_init,N)                                   
    z_fronts = funcs.capillary_relax(z_fronts, max_depth)
    logging.info(np.round(z_fronts[95:],3))
    cum_inf[i] = infiltration
    Hp_list.append(Hp)
    max_bin_list.append(max_bin)
    frontspeed_list.append(frontspeed)
    z_history[i,:] = z_fronts

results_df  = pd.DataFrame(z_history).T





#### Evaluation of results ####
eval_df = pd.DataFrame({
    "max_bin": max_bin_list,
    "front_speed": frontspeed_list,
    "Hp": Hp_list,
    "cum_inf": cum_inf})

logging.info(f'totale infiltratie = {cum_inf[-1]}')

#calculate the mass balance error
abs_error = cum_rain - cum_inf[-1] - Hp
perc_error = abs_error/cum_rain*100

#plot some evaluation plots
fig, ax = plt.subplots()
l1, = ax.plot(cum_inf, color="tab:blue", label="cum_inf")
l2, = ax.plot(frontspeed_list, color="tab:green", label="front_speed")
ax.set_xlabel("Timestep")
ax.set_ylabel("cum_inf", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_ylim(0, 0.02)

ax2 = ax.twinx()
l3, = ax2.plot(Hp_list, color="tab:orange", label="Ponded depth")
ax2.set_ylabel("Ponded depth", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
#ax2.set_ylim(0,0.1)


ax3 = ax.twinx()
l4, = ax3.plot(max_bin_list, color="tab:purple", label="max_bin_list")
ax3.set_ylabel("max_bin_list", color="tab:purple")
ax3.tick_params(axis="y", labelcolor="tab:purple")
ax3.set_ylim(0, 100)
# move the third axis to avoid overlap
ax3.spines["right"].set_position(("outward", 60))

# combined legend
lines = [l1, l2, l3, l4]
labels = [ln.get_label() for ln in lines]
ax.legend(lines, labels)

ax.set_title("Cumulative Infiltration, Front speed, Ponded water, Max bin")
plt.show()


#plot with theta
x_theta = bins["theta_bins"]

fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x_theta, z_history[0], marker="o", lw=2)

ax.set_xlabel("Soil moisture θ")
ax.set_ylabel("Front depth [cm]")
ax.set_title("Soil front depth over time")
ax.invert_yaxis()
ax.set_xlim(x_theta.min(), x_theta.max())
ax.set_ylim(z_history.max() * 1.05, z_history.min() * 0.95)

def update(frame):
    line.set_xdata(x_theta)
    line.set_ydata(z_history[frame])
    ax.set_title(f"Time step {frame + 1} / {z_history.shape[0]}")
    return (line,)

anim = FuncAnimation(fig, update, frames=z_history.shape[0], interval=50, blit=False)

plt.show()

anim

'''
#Plot with bin index
x_bins = np.arange(z_history.shape[1])
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x_bins, z_history[0], marker="o", lw=2)

ax.set_xlabel("Bin index")
ax.set_ylabel("Front depth [cm]")
ax.set_title("Soil front depth over time")
ax.invert_yaxis()
ax.set_xlim(0, z_history.shape[1] - 1)
ax.set_ylim(z_history.max() * 1.05, z_history.min() * 0.95)

def update(frame):
    line.set_ydata(z_history[frame])
    ax.set_title(f"Time step {frame + 1} / {z_history.shape[0]}")
    return (line,)

anim = FuncAnimation(fig, update, frames=z_history.shape[0], interval=100, blit=False)

plt.show()

anim

# create K(θ) plot
x = bins["theta_bins"]
y = (bins["K_bins"])

fig, ax = plt.subplots()
ax.plot(x, y, marker="o", linestyle="-")
ax.set_xlabel("Soil moisture θ")
ax.set_ylabel("Hydraulic conductivity K")
ax.set_title("unsaturated hydraulic conductivity")
plt.show()






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
'''