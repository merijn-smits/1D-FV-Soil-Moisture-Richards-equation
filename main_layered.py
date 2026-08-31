from datetime import datetime
import funcs_layered as funcs
import numpy as np
import pandas as pd
import logging
import plotting


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
                        style = '{')
logging.info(f'time = {datetime.now()}')

##SETTINGS##
h_max = 17000  # maximum matric suction [cm] (16000 is wilting point)
h_min = 0.001   # minimum matricsuction [cm] to prevent numerical issues
#max_depth = 120 #maximum modeling depth in [cm]
h_init = 1000
dt = 0.25 /24/60 #from minutes to days
t_max = 240  *1/24/60 #in [minutes]
t_steps = round(t_max/dt)
N = 100

time_vec = np.array([(i/t_steps) for i in range(1,t_steps+1)])

##RAINFALL SETTINGS###
rainfall_rate = 20 * 24/10 #[mm/hr] to [cm/day]
rain_end = 240 * 1/24/60 #[minutes] to [days]
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
mean_inf = np.zeros(len(profiles))

#select one soil
soil_code = "Zd30"

for j, profile in enumerate(profiles):
    if profile != soil_code:
        continue
    #Set initial values
    n_layers = len(profiles[profile])
    Hp_array = np.zeros((t_steps,n_layers+1))
    z_fronts = np.zeros((N,n_layers))     
    z_history = np.zeros((t_steps, N, n_layers))

    #initalise all results evaluations
    inf_mm_day = np.zeros((t_steps,n_layers))
    max_bin_list = np.zeros((t_steps,n_layers))
    frontspeed_list = np.zeros((t_steps,n_layers))
    Hp_top_list = np.zeros((t_steps,n_layers))
    Hp_bot_list = np.zeros((t_steps,n_layers))
    mass_balance = np.zeros((t_steps,n_layers))
    mass_bal_tot = np.zeros(t_steps)
    cum_inf = np.zeros((t_steps,n_layers))
    delta_stored_vec = np.zeros((t_steps,n_layers))

    

    #Timeloop
    for i ,t  in enumerate(time_vec):
        logging.info(f't = {i}, rain = {rain_vec[i]}')

        #reset the head the the bottom of the last layer to zero for free drainage
        if i > 0:
            Hp_array[i, :] = Hp_array[i-1, :]   # carry forward as starting point
            water_out = 0
        Hp_array[i, -1] = 0.0                   # free drainage at bottom boundary only,
        
        #Loop through all the layers
        for layer in range(len(profiles[profile])):
            logging.info(f'layer = {layer}')
            if layer == 0:
                rain = rain_vec
            else:
                rain = np.zeros(len(rain_vec))
            #     continue
            z_fronts[:,layer], Hp_array[i,layer], infiltration, max_bin, frontspeed, Hp_array[i,layer+1] = funcs.handle_infiltration (rain[i], profiles[profile][layer], z_fronts[:,layer], dt, Hp_array[i,layer], Hp_array[i,layer+1], N)                                   
            z_fronts[:,layer] = funcs.capillary_relax(z_fronts[:,layer], profiles[profile][layer]['thickness'])

            #checks 
            #logging.info(np.round(z_fronts[95:],3))
            inf_mm_day[i,layer] = infiltration/dt*10
            max_bin_list[i,layer] = max_bin
            frontspeed_list[i,layer] = frontspeed
            z_history[i,:,layer] = z_fronts[:,layer]

            
            # Mass balance check per layer per timestep
            z_stored = np.sum(z_history[i,:,layer] * profiles[profile][layer]['delta_theta'])
            z_stored_prev = np.sum(z_history[max(0,i-1),:,layer] * profiles[profile][layer]['delta_theta']) if i>0 else 0
            delta_stored = z_stored - z_stored_prev
            delta_stored_vec[i,layer] = delta_stored

            #if first layer, water in = rain - Δ stored ponded at the top
            if layer == 0:
                water_in = rain_vec[i] * dt - (Hp_array[i, layer]   - (Hp_array[i-1, layer]   if i>0 else 0))
            else:
            #for the other layer water in = waterdraining [layer-1][i] - Δ store at top [layer]
                water_in  = water_out -(Hp_array[i, layer]   - (Hp_array[i-1, layer]   if i>0 else 0))
            
            water_out = Hp_array[i, layer+1] - (Hp_array[i-1, layer+1] if i>0 else 0)
            mb_error  = water_in - water_out - delta_stored
            if abs(mb_error) > 1e-4:
                logging.warning(
                    f't={i} layer={layer}: MB error={mb_error:.6f} '
                    f'water_in={water_in:.4f} water_out={water_out:.4f} delta_stored={delta_stored:.4f}'
                )
            cum_inf[i,layer] = water_in
            inf_rate = cum_inf[:,0]/dt *10
            #logging.info(f'Inf_mm_day = {inf_mm_day[i]}')
        #calculate total mass balance
        mass_bal_tot[i] = rain_vec[i]*dt - np.sum(Hp_array[i,:]-Hp_array[i-1,:])- np.sum(delta_stored_vec[i])

    #calculate the total infiltration [cm]
    mean_inf[j] = np.mean(inf_mm_day)
    #results_df  = pd.DataFrame(z_history).T

# layered_inf = inf_rate

# df = pd.DataFrame({
#     'time':time_vec/dt*t_max,
#     'layered_inf':layered_inf
# })

#Merge results with Bofek data and write to file
results = pd.DataFrame({
    'soil_code' : soil_code,
    'infiltration_FVR' : mean_inf    
    })

infiltration = round(pd.DataFrame(inf_mm_day[:,0]),1)
infiltration.rename(columns= lambda x: x+1,inplace = True)
infiltration.insert(0, 'time_days',time_vec*t_max)
infiltration.to_csv('./results/FVR_layered_100_bui8_dt15_tm4.csv')


bofek = pd.read_csv('bofek.csv')[['bodemcode','isoil1']]
bofek_new = pd.merge(bofek, results, left_on = 'isoil1', right_on = 'soil_code').drop(columns= ['soil_code', 'isoil1'])
bofek_new.to_csv(f'./results/Inf_{round(rainfall_rate)}mm_h_{h_init}.csv', sep = ',')


#set layer to evaluate
layer = 0

#Create eval plots
fig, axes = plotting.plot_evaluation(
    cum_inf[:,layer], 
    frontspeed_list[:,layer],    
    Hp_array[:,layer], 
    Hp_array[:,layer+1], 
    max_bin_list[:,layer])



'''
anim = plotting.animate_fronts(
    z_history    = z_history,
    profiles     = profiles,
    profile_name = 'Zd30',
    Hp_array     = Hp_array,
    interval     = 50        # ms between frames
)
# anim.save("animation_layered.mp4", writer="ffmpeg", fps=30, dpi=100)
'''