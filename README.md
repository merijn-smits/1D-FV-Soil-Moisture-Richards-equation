# 1D-FV-Soil-Moisture-Richards-equation #
1D finite soil moisture Richards equation as described in https://doi.org/10.1002/2015WR017126
I created the package in Positron, which allows line by line execution. To generate animations, run files in the terminal, otherwise animations do not generate

The dir consists of three seperate models: 
1. main.py assumes the entire soil column  to be the top soil, Van Genuchten parameters must be added at the top by hand
2. main_loop.py also assumes a uniform soil profile but uses the Staringreeks directly as input, the topsoilis assumed to be representative for the enitire soil profile
3. main_layered.py where soil columns from the BOFEK are used

## unlayered soils: main.py ##
Above the logger setup, add your parameters once. Here the discretization of the soil moisture in bins is also set.
Everthing below must be reset for each run and runs the FVR for the whole length of time. in z_fronts the front positions of all bins are written, these are subsequently stored in z_history per timestep.


## unlayered soils: main_loop.py ##
Pretty much the same as main.py, but automatically takes all Van Genuchten soil parameters from the Staringreeks. The Staringreeks and BOFEK are added as a csv, but can also be re-generated using get_Staringreeks.r when a Staringreeks update is available. Up to Intialise results it is all settings. Currently is set to write the a file with infiltration rates at each timestep for each soil to file, and write a file with the average infiltration during the whole simulation period. 
Various plots and animations are available for evaluation. These are currently commented out.

## layered soils: main_layered.py ##
Follows the same logic as the previous files, but uses a flux-driven extension to apply layered soils. Soil profiles from the BOFEK are used for this. Up to INITIALIZE BINS is settings. The infiltration DataFrame gives the infiltration rate at each timestep.

## plotting.py ##
contains functions for creating plots and animations

## calc_horton.py ##
contains a function for fitting the FVR on the horton function and getting its parameters.