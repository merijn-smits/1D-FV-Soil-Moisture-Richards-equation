# TASKS #

## Bugs ##

- ~~Geff now once shoots to bin 99 and never drops back to highest value of last iter~~
    - ~~make something that pulls water to the left ~~

## Current ##
- ~~time implementation~~
- ~~capillary relaxation~~

    - ~~think of way to improve infiltration of non-ponded water~~:
        - ~~find a way to find the theta_surface using an inversion of K(theta_surface) = infiltration --> bissection~~
        - ~~or cechk out Newton raphson if it performs better~~
- ~~check inputs for theta_(r/i) and theta_(s/e)~~
- ~~review the input for Sf in the infiltration~~
    -~~Evaluate Geff as conductivity weighted so that low theta bins do not contribute to the conductivity:~~
        ~~Geff,j​=∫θj−1​θj​​K(θ)dθ∫θj−1​θj​​h(θ)K(θ)dθ​~~
- ~~create module to infiltration handler which pulls water from right bins to left if any bin has unsatisfied demand.~~
- ~~Evaluate the pulling and bin activation~~
- ~~perform mass balance checks~~ -> mb error<1E-3%
- ~~check infiltration per timestep~~ -> checked and graphs created, not fixable short term
- ~~check if ponded infiltration performs well and also that only ponded (no rain) works well~~ -> works reasonable, but jumps remain
- ~~think of way to calculate a 'constant' infiltration for a typical shower ( duration 1 hr + 3 hr after)~~
    - ~~Infiltration after ponding has has infiltrated is not necessary~~
- ~~use actual rainfall and initialize the possibility for non-ponded infiltration~~
- ~~Evaluate infiltration of fully saturated bins and not fully saturated bins~~ -> still a weird jump present when switchting from fronts to saturated infiltration

- ~~Integrate full staringreeks~~

- ~~compare with (modified) Green-Ampt ~~
- ~~add layered soils~~
- evaluate sat_inf, only the highest saturated bin? multiply by Δθ? Sum the infiltation in sat_bins?
- for layered soils
    - improve the layer transitions. With a layered soil of the same two soil layers, there is still a jump, transition should be smooth.
        - this could be θi and θd which is dependent on only this layer, while it should perhaps be from the whole infiltration front?
        - --> somehow move sat_idx and active_idx to global
    - waterbalance gets violated if sub layers have a much lower k than toplayers. then Hp_array[0] negative and infiltration somehow increases, while it should decrease. Something gets mixed up in the handle_infitration
    - probably best solution is to mak the connection head-driven, but this needs a root finder to find the head at the boundary



## Long-term ##
- ~~make theta_init dependent on the soil~~
- cleam up units --> make everthing mm and hours?
- compare with observations
- the dips in inifiltration for ponded only
- clean up 'bins' input
- evaluate the oscillation in infiltration for rainfall infiltration
- add falling slugs
- ~~add layered soils~~
- add groundwater module
- improve speed by changing for loops to arrays/masks


