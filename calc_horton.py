import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
This script fits the infiltration curves of e.g. the Modified green ampt equation and the Finite volume richards equation
to the horton curves. It returns a plot which visualises how the best fitting horton plot looks compared to the original infiltration curves, and it returns a csv with the horton paramter values.
'''


df = pd.read_csv('./results/FVR_infiltration.csv')
###HORTON PARAMETER FITTING###

def horton(t, f0, fc, k):
    return fc + (f0 - fc) * np.exp(-t / k)

#define weights, focus on the first four hours
WEIGHT_EARLY = 10.0
WEIGHT_LATE  = 1.0
four_hours   = 4 / 24        # in days

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
    y = np.array(df[col][df[col]<1200])    # FIXLATER: 1200 is the rainfall rate for this simulation, make variable
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


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Green-Ampt → Horton Fit soil {col}", fontsize=13, fontweight='bold')

    for ax, mask, title in zip(
        axes,
        [pd.Series([True] * len(y)), mask_4h],
        ["Full curve (2 days)", "First 4 hours — high-weight zone"]
    ):
        ax.plot(t_hours[mask], y[mask],
                label='Modified Green-Ampt', color='steelblue', lw=1.5, alpha=0.8)
        ax.plot(t_hours[mask], f_horton[mask],
                label='Horton fit', color='orangered', lw=2, linestyle='--')
        if title.startswith("Full"):
            ax.axvline(4, color='gray', linestyle=':', lw=1.2, label='4h weight boundary')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Infiltration rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'./Graphs/horton/horton_fit_weighted_soil_{col}.png', dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()

results = pd.DataFrame({
    'soil' : soil_cols,
    'f_init' : f0,
    'f_equ' : fc,
    'k' : k
})
results.to_csv('./results/horton_params_FVR.csv')
