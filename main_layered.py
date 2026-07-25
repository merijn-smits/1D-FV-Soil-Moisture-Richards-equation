from datetime import datetime
import funcs_layered as funcs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import logging

'''
the the main.loopred function, but adapted to accomodate layered soils from the bofek
'''

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

##SETTINGS##
h_max = 17000  # maximum matric suction [cm] (16000 is wilting point)
h_min = 0.001   # minimum matricsuction [cm] to prevent numerical issues
max_depth = 120 #maximum modeling depth in [cm]
h_init = 1000

t_steps = 900
N = 100
t_max = 480  *1/24/60 #in [minutes]
dt = t_max/t_steps
time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])

##RAINFALL SETTINGS###
rainfall_rate = 50 * 24/10 #[mm/hr] to [cm/day]
rain_end = 480 * 1/24/60 #[minutes] to [days]
rain_vec = np.zeros(t_steps)
rain_vec[:np.where(time_vec == (rain_end/t_max))[0][0]] = rainfall_rate
cum_rain = t_max * rainfall_rate #FIXLATER change to accomodate rain_vec

##LOAD STARINGREEKS##
staring = pd.read_csv('staring.csv')
theta_r = np.array(staring['wcr'])
theta_e = np.array(staring['wcs'])
Ks = np.array(staring['ksfit'])
n = np.array(staring['npar'])
m = 1-1/n
labda = np.array(staring['lambda'])
alpha = np.array(staring['alpha'])

##INITIALISE BINS##
'''
Create a dict of dicts with all soil properties per staringreeks. 
This dict is used in the infiltration handelr to look up the soil properties each timestep.
'''
soils = {}

for i, name in enumerate(staring['unit']):
    print(f'soil is {i+1}')
    #Create the bins
    bins = funcs.create_bins(N , theta_r[i], theta_e[i], labda[i], alpha[i],n[i] ,Ks[i], h_max, h_min, h_init, dt)
    bins = {
        'soil': name, 
        **bins
        }
    soils[name] = bins

##INTIALISE PROFILES##
bofek = pd.read_csv('bofek.csv')
profiles = funcs.build_layers_by_soil(bofek, soils)


##INTIALISE RESULTS##

#soil_code = np.array(staring['unit'])
mean_inf = np.zeros(len(profiles))
infiltration_list = []

for j, profile in enumerate(profiles):
    if profile != 'EZ50A':
        continue
    #Set initial values
    Hp_top = 0 #initial ponding depth [cm]
    Hp_bot = 0
    z_fronts = np.zeros(N)     
    z_history = np.zeros((t_steps, N))

    #calculate the initial amount of water in all bins
    inf_mm_day = np.zeros(t_steps)
    max_bin_list = np.zeros(t_steps)
    frontspeed_list = np.zeros(t_steps)
    Hp_top_list = np.zeros(t_steps)
    Hp_bot_list = np.zeros(t_steps)
    mass_balance = np.zeros(t_steps)

    

    #Timeloop
    for i ,t  in enumerate(time_vec):
        logging.info(f't = {i}, rain = {rain_vec[i]}')
        z_fronts, Hp_top, infiltration, max_bin, frontspeed, Hp_bot = funcs.handle_infiltration (rain_vec[i], profiles[profile][0], z_fronts, dt, Hp_top, Hp_bot, N)                                   
        z_fronts = funcs.capillary_relax(z_fronts, max_depth)

        #checks 
        #logging.info(np.round(z_fronts[95:],3))
        inf_mm_day[i] = infiltration/dt*10
        Hp_top_list[i] = Hp_top
        max_bin_list[i] = max_bin
        frontspeed_list[i] = frontspeed
        z_history[i,:] = z_fronts
        Hp_bot_list[i] = Hp_bot

        #calculate mass balance per time step
        delta_Hp_top = Hp_top_list[i]-Hp_top_list[i-1]
        delta_Hp_bot = Hp_bot_list[i]-Hp_bot_list[i-1]
        delta_fronts = (np.sum(z_history[i]) - np.sum(z_history[i-1]))*profiles[profile][0]['delta_theta']
        mass_balance[i] = rain_vec[i]*dt - delta_Hp_top - delta_Hp_bot - delta_fronts
        #logging.info(f'Inf_mm_day = {inf_mm_day[i]}')

    #calculate the total infiltration [cm]
    mean_inf[j] = np.mean(inf_mm_day)
    infiltration_list.append(inf_mm_day)
    #results_df  = pd.DataFrame(z_history).T

#Merge results with Bofek data and write to file
results = pd.DataFrame({
    'soil_code' : soil_code,
    'infiltration_FVR' : mean_inf    
    })

infiltration = round(pd.DataFrame(infiltration_list).T,1)
infiltration.rename(columns= lambda x: x+1,inplace = True)
infiltration.insert(0, 'time_days',time_vec*t_max)
infiltration.to_csv('./results/FVR_infiltration.csv')


bofek = pd.read_csv('bofek.csv')[['bodemcode','isoil1']]
bofek_new = pd.merge(bofek, results, left_on = 'isoil1', right_on = 'soil_code').drop(columns= ['soil_code', 'isoil1'])
bofek_new.to_csv(f'./results/Inf_{round(rainfall_rate)}mm_h_{h_init}_z_{round(max_depth)}.csv', sep = ',')


'''
#### Evaluation of results ####
eval_df = pd.DataFrame({
    "max_bin": max_bin_list,
    "front_speed": frontspeed_list,
    "Hp": Hp_list,
    "cum_inf": cum_inf})



#calculate the mass balance error
abs_error = cum_rain - cum_inf[-1] - Hp
perc_error = abs_error/cum_rain*100

#Create eval plots
fig, axes = plotting.plot_evaluation(cum_inf, frontspeed_list, Hp_list, max_bin_list)
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


#### left over ####

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