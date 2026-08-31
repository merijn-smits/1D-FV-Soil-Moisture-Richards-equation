from numpy import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
This script fits the infiltration curves of e.g. the Modified green ampt equation and 
the Finite volume richards equationto the horton curves. 
It returns a plot which visualises how the best fitting horton plot looks
compared to the original infiltration curves, and it returns a csv with the horton paramter values.
'''


df = pd.read_csv('./results/FVR_1000_240.0cmd_dt0.25m_tmax4hr.csv') #in mm/day
###HORTON PARAMETER FITTING###

def horton(t, f0, fc, k):
    return fc + (f0 - fc) * np.exp(-t * k)

#define weights, focus on the first four hours
WEIGHT_EARLY = 1
WEIGHT_LATE  = 1
four_hours   = 2 / 24        # in days

bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])  # all params must be positive
soil_cols = [c for c in df.columns if c != 'time_days']
n_soils = len(soil_cols)

f0 = np.zeros(n_soils)
fc = np.zeros(n_soils)
k = np.zeros(n_soils)
f0_cov = np.zeros(n_soils)
fc_cov = np.zeros(n_soils)
k_cov = np.zeros(n_soils)


weights = np.where(df['time_days'] <= four_hours, WEIGHT_EARLY, WEIGHT_LATE)

for idx, col in enumerate(soil_cols):
    #only select the part of the infiltration curve where infiltration is less than the precipitation
    y = np.array(df[col][df[col]<2400])    # FIXLATER: 2400 is the rainfall rate for this simulation, make variable
    t = np.array(df['time_days'][:len(y)])
    weights = np.where(df['time_days'] <= four_hours, WEIGHT_EARLY, WEIGHT_LATE)[-len(y):]
    p0 = [y[0], y[-1], four_hours]

    popt, pcov = curve_fit(
        horton,
        t,
        y,
        p0=p0,
        bounds=bounds,
        sigma=1.0 / weights,
        absolute_sigma=False,
        maxfev=10000
    )

    f0[idx], fc[idx], k[idx] = popt
    #f0_cov[idx], fc_cov[idx], k_cov[idx] = pcov #Later analyse varience and covarience

    f_horton = horton(t, *popt)
    t_hours  = t * 24
    mask_4h  = weights>1 # FIXLATER; chnage to allow weights also to be high at the tail
    RMSE = np.sqrt(sum((f_horton-y)**2)/len(y))
    # plot the Horton fits and save to png
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f"Finite volume Richards → Horton Fit horizont {col}", fontsize=13, fontweight='bold')

    ax.plot(t_hours, y, label='Finite volume Richards', color='steelblue', lw=1.5, alpha=0.8)
    ax.plot(t_hours, f_horton, label='Horton fit', color='orangered', lw=2, linestyle='--')
    ax.set_xlabel('Tijd (uur)')
    ax.set_ylabel('Infiltratiesnelheid (mm/dag)')
    ax.set_title(f'f0 = {round(f0[idx])} mm/dag, fc = {round(fc[idx])} mm/dag, k = {round(k[idx],3)} 1/dag, RMSE = {round(RMSE,1)}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./Graphs/horton/horton_fit_weighted_soil_{col}.png', dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()

#Divisions are to compare to Rioned, without everything is in mm/day orf day
results = pd.DataFrame({
    'soil' : soil_cols,
    'f_init' : f0/24, #mm/hr
    'f_equ' : fc/24, #mm/hr
    'k' : k/24/60 #1/min
})
#results.to_csv('./results/horton_params_FVR_Rioned.csv')
