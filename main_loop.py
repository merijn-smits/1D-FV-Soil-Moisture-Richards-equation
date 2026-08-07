from datetime import datetime
import funcs
import plotting 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import logging

'''
the original main function, but lopped over all soils of the staring reeks
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
t_start = datetime.now()
logging.info(f't_start = {t_start}')

##SETTINGS##
h_max = 17000  # maximum matric suction [cm] (16000 is wilting point)
h_min = 0.001   # minimum matricsuction [cm] to prevent numerical issues
max_depth = 100 #maximum modeling depth in [cm]
h_init = 1000
N = 100
t_max = 60*24  *1/24/60 #from [minutes] to days
dt = 2  *1/24/60 #from [minutes] to days
t_steps = round(t_max/dt)
time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])

##RAINFALL SETTINGS###
rainfall_rate = 20 * 24/10 #[mm/hr] to [cm/day]
rain_end = 60 * 1/24/60 #[minutes] to [days]
rain_vec = np.zeros(t_steps)
rain_vec[:np.where(time_vec == (rain_end/t_max))[0][0]+1] = rainfall_rate  #+1 for right exclusive index
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

##INITIALISE RESULTS##
soil_code = np.array(staring['unit'])
mean_inf = np.zeros(len(soil_code))
infiltration_list = []
Hp_frame = []

for j in range(len(staring)):
    if j == 0:
        logging.info(f'time = {datetime.now()}')
        
    print(f'soil is {j+1}')
    #Create the bins
    bins = funcs.create_bins(N , theta_r[j], theta_e[j], labda[j], alpha[j],n[j] ,Ks[j], h_max, h_min, dt)

    #Set initial values
    Hp = 0 #initial ponding depth [cm]
    theta_init = max(bins['theta_bins'][1],funcs.theta_h(h_init,theta_r[j],theta_e[j], alpha[j], m[j], n[j])) #force the first bin to be full of water
    idx = (np.abs(bins['theta_bins'] - theta_init)).argmin()
    z_fronts = np.zeros(N)     
    z_history = np.zeros((t_steps, N))
    z_fronts[:idx] = max_depth #set the bins below theta_init to fully saturated
    z_init = np.sum(z_fronts * bins['delta_theta'])
    inf_mm_day = np.zeros(t_steps)
    max_bin_list = np.zeros(t_steps)
    frontspeed_list = np.zeros(t_steps)
    Hp_list = np.zeros(t_steps)
    

    #Timeloop
    for i ,t  in enumerate(time_vec):
        logging.info(f't = {i}, rain = {rain_vec[i]}')
        z_fronts, Hp, infiltration, max_bin, frontspeed = funcs.handle_infiltration (rain_vec[i], bins, z_fronts, 
                                                alpha[j], m[j], n[j], theta_r[j], theta_e[j],
                                                dt, Hp, max_depth, theta_init,N)                                   
        z_fronts = funcs.capillary_relax(z_fronts, max_depth)
        logging.info(np.round(z_fronts[95:],3))
        inf_mm_day[i] = infiltration/dt*10
        Hp_list[i] = Hp
        max_bin_list[i] = max_bin
        frontspeed_list[i] = frontspeed
        z_history[i,:] = z_fronts
        logging.info(f'Inf_mm_day = {inf_mm_day[i]}')

    #calculate the total infiltration [cm]
    mean_inf[j] = np.mean(inf_mm_day)
    infiltration_list.append(inf_mm_day)
    #results_df  = pd.DataFrame(z_history).T

# unlayered_inf = np.array(infiltration_list[4])

# #Merge results with Bofek data and write to file
# results = pd.DataFrame({
#     'soil_code' : soil_code,
#     'infiltration_FVR' : mean_inf    
#     })

# #create plots for the influence of initial soilmoisture
# # results_field_cap = results
# # results_field_cap['infiltration_FVR'] = results_field_cap['infiltration_FVR']/24 #convert to mm/hr
# results_wilting = results
# results_wilting['infiltration_FVR'] = results_wilting['infiltration_FVR']/24 #convert to mm/hr
# merged = pd.merge(results_field_cap,results_wilting, on = 'soil_code')
# merged.columns = ['soil_code','field_capacity', 'wilting_point']
# fig, ax = plotting.plot_field_vs_wilting(merged)

infiltration = round(pd.DataFrame(infiltration_list).T,1)
infiltration.rename(columns= lambda x: x+1,inplace = True)
infiltration.insert(0, 'time_days',time_vec*t_max)
infiltration.to_csv('./results/FVR_1000_bui8_dt2m_tmax4hr.csv')
t_end = datetime.now()
t_diff = t_end -t_start
logging.info(f't_end = {t_end}, t_diff = {t_start}')


'''

### Coupling to BOFEK to create infiltration map
bofek = pd.read_csv('bofek.csv')[['bodemcode','isoil1']]
bofek_new = pd.merge(bofek, results, left_on = 'isoil1', right_on = 'soil_code').drop(columns= ['soil_code', 'isoil1'])
bofek_new.to_csv(f'./results/Inf_{round(rainfall_rate)}mm_h_{h_init}_z_{round(max_depth)}.csv', sep = ',')


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