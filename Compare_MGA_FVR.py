import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
This script loads both the Modofied green ampt and Finite volume richards infiltration rates\
and plots them in a graph 
'''
MGA = pd.read_csv('staring_MGA_120mm_h1000_4hr.csv').iloc[:,1:]
FVR = pd.read_csv('staring_FVR_120mm_h1000_4hr.csv').iloc[:,1:]

merged  = pd.merge(MGA,FVR)

top = merged.iloc[:18]
sub = merged.iloc[18:]

x_top = np.arange(len(top))
x_sub = np.arange(len(sub))

colors = {'FVR': '#4c78a8', 'MGA': '#f58518'}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharey=True)

width = 0.35

ax1.bar(x_top - width / 2, top['infiltration_FVR'], width, color=colors['FVR'], label='FVR')
ax1.bar(x_top + width / 2, top['infiltration_MGA'], width, color=colors['MGA'], label='MGA')
ax1.set_title('FVR vs MGA 4hr mean infiltration for topsoils')
ax1.set_xticks(x_top)
ax1.set_xticklabels(top['soil_code'], rotation=90)
ax1.set_ylabel('infiltration (mm/hr)')
ax1.legend()

ax2.bar(x_sub - width / 2, sub['infiltration_FVR'], width, color=colors['FVR'], label='FVR')
ax2.bar(x_sub + width / 2, sub['infiltration_MGA'], width, color=colors['MGA'], label='MGA')
ax2.set_title('FVR vs MGA 4hr mean infiltration for subsoils')
ax2.set_xticks(x_sub)
ax2.set_xticklabels(top['soil_code'], rotation=90)
ax2.set_xlabel('soil_code')
ax2.set_ylabel('infiltration (mm/hr)')
ax2.legend()

fig.tight_layout()
fig