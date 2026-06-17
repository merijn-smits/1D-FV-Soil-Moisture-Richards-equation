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
- ~~Possibly think of way to calculate a 'constant' infiltration for a typical shower ( duration 1 hr + 3 hr after)~~
    - ~~Infiltration after ponding has has infiltrated is not necessary~~
- ~~use actual rainfall and initialize the possibility for non-ponded infiltration
- Evaluate infiltration of fully saturated bins and not fully saturated bins

- clean up 'bins' input

- Integrate full staringreeks

- compare with (modified) Green-Ampt 
- compare with observations

- 

## Long-term ##
- the dips in inifiltration for ponded only
- evaluate the oscillation in infiltration for rainfall infiltration
- add falling slugs
- add layered soils
- add groundwater module
- add layered soils
- improve speed by changing for loops to arrays/masks

