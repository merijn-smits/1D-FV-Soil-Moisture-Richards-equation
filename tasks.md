# TASKS #

## Current ##
- ~~time implementation~~
- ~~capillary relaxation~~
- use actual rainfall and initialize the possibility for non-ponded infiltration
    - think of way to improve infiltration of non-ponded water:
        - ~~find a way to find the theta_surface using an inversion of K(theta_surface) = infiltration --> bissection~~
        - or cechk out Newton raphson if it performs better
- check inputs for theta_(r/i) and theta_(s/e)
- review the input for Sf in the infiltration
    - Evaluate Geff as conductivity weighted so that low theta bins do not contribute to the conductivity:
        Geff,j​=∫θj−1​θj​​K(θ)dθ∫θj−1​θj​​h(θ)K(θ)dθ​
- compare with (modified) Green-Ampt

## Long-term ##
- add falling slugs
- clean up 'bins' input
- add groundwater module
- add layered soils

