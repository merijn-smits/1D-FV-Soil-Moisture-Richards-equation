"""
fwc_functions.py
================
Finite Water-Content (FWC) vadose zone flow model.

Primary reference:
    Ogden, F.L., Lai, W., Steinke, R.C., Zhu, J., Talbot, C.A., Wilson, J.L. (2015).
    A new general 1-D vadose zone flow solution method.
    Water Resources Research, 51, 4282-4300. doi:10.1002/2015WR017126

Supporting references:
    Talbot, C.A. & Ogden, F.L. (2008). A method for computing infiltration and
    redistribution in a discretized moisture content domain. WRR, 44, W08453.

    Zhu, J., Ogden, F.L., Lai, W., Chen, X., Talbot, C.A. (2016). An explicit
    approach to capture diffusive effects in the finite water-content method.
    Advances in Water Resources, 88, 141-155.

    van Genuchten, M.Th. (1980). A closed-form equation for predicting the
    hydraulic conductivity of unsaturated soils. SSSA Journal, 44, 892-898.

    Mualem, Y. (1976). A new model for predicting the hydraulic conductivity
    of unsaturated porous media. Water Resources Research, 12, 513-522.

Also contains:
    - Standard Green-Ampt (Newton-Raphson, Euler, RK4)
    - Modified Green-Ampt (Sun et al. 2023 analytical form)
    - Comparison wrapper

Unit convention: SI throughout (metres [m], hours [hr]).
    Convert Rawls et al. (1983) values: cm -> m (*0.01), cm/hr -> m/hr (*0.01)

Compatible with Positron IDE (line-by-line execution).
No code executes at import time.
"""

import numpy as np
from typing import Optional


# =============================================================================
# SECTION 1: VAN GENUCHTEN / MUALEM HYDRAULIC FUNCTIONS
# =============================================================================
# All soil-water characteristic and conductivity functions.
# The Mualem constraint m = 1 - 1/n is assumed throughout (standard form).

def effective_saturation(theta, theta_r, theta_s):
    """
    Effective saturation Se [-].
    Se = (theta - theta_r) / (theta_s - theta_r)    [vG 1980, eq. 2]

    Clamped to [1e-10, 1] to prevent singularities at the dry and wet limits.
    This clamp is applied before any h or K calculation.
    """
    Se = (theta - theta_r) / (theta_s - theta_r)
    return np.clip(Se, 1e-10, 1.0)


def vg_capillary_head(Se, alpha, n, m=None):
    """
    Van Genuchten capillary pressure head h [m] as a function of Se.
    Inverted from Se = [1 + (alpha*h)^n]^(-m):
        h = (1/alpha) * (Se^(-1/m) - 1)^(1/n)    [vG 1980, eq. 3 inverted]

    Returns positive h (suction head convention: h > 0 means unsaturated).
    alpha : van Genuchten alpha [1/m]
    n     : van Genuchten shape parameter [-]
    m     : set to 1 - 1/n (Mualem constraint) if None
    """
    if m is None:
        m = 1.0 - 1.0 / n
    Se = np.clip(Se, 1e-10, 1.0 - 1e-10)
    return (1.0 / alpha) * (Se ** (-1.0 / m) - 1.0) ** (1.0 / n)


def vg_head_from_theta(theta, theta_r, theta_s, alpha, n, m=None):
    """Convenience wrapper: capillary head directly from water content theta."""
    Se = effective_saturation(theta, theta_r, theta_s)
    return vg_capillary_head(Se, alpha, n, m)


def kr_mualem_vg(Se, n, l=0.5, m=None):
    """
    Relative hydraulic conductivity Kr [-] by Mualem (1976) combined with
    van Genuchten (1980) retention (the standard Mualem-vG model):
        Kr = Se^l * [1 - (1 - Se^(1/m))^m]^2    [vG 1980, eq. 8]

    l = 0.5 is Mualem's pore-connectivity / tortuosity parameter.
    m = 1 - 1/n (Mualem constraint).
    """
    if m is None:
        m = 1.0 - 1.0 / n
    Se = np.clip(Se, 1e-10, 1.0)
    inner = (1.0 - Se ** (1.0 / m)) ** m
    Kr = Se ** l * (1.0 - inner) ** 2
    return np.clip(Kr, 0.0, 1.0)


def hydraulic_conductivity(theta, theta_r, theta_s, Ks, alpha, n, l=0.5, m=None):
    """
    Unsaturated hydraulic conductivity K(theta) = Ks * Kr(Se(theta)) [m/hr].
    """
    if m is None:
        m = 1.0 - 1.0 / n
    Se = effective_saturation(theta, theta_r, theta_s)
    Kr = kr_mualem_vg(Se, n, l, m)
    return Ks * Kr


def dh_dtheta(theta, theta_r, theta_s, alpha, n, m=None):
    """
    Derivative of capillary head with respect to water content: dh/dtheta [m].
    Computed via the chain rule: dh/dtheta = (dh/dSe) * (dSe/dtheta).

    dSe/dtheta = 1 / (theta_s - theta_r)

    dh/dSe is the derivative of h = (1/alpha)*(Se^(-1/m) - 1)^(1/n) w.r.t. Se:
        dh/dSe = -(1/(alpha*n*m)) * (Se^(-1/m) - 1)^(1/n - 1) * Se^(-1/m - 1)

    Returns negative value (h decreases as theta increases).
    Used in computing soil water diffusivity D = K * |dh/dtheta|.
    """
    if m is None:
        m = 1.0 - 1.0 / n
    Se = effective_saturation(theta, theta_r, theta_s)
    Se = np.clip(Se, 1e-8, 1.0 - 1e-8)

    dSe_dtheta = 1.0 / (theta_s - theta_r)

    term = Se ** (-1.0 / m) - 1.0
    term = np.maximum(term, 1e-10)
    dh_dSe = -(1.0 / (alpha * n * m)) * term ** (1.0 / n - 1.0) * Se ** (-1.0 / m - 1.0)

    return dh_dSe * dSe_dtheta


def soil_water_diffusivity(theta, theta_r, theta_s, Ks, alpha, n, l=0.5, m=None):
    """
    Soil water diffusivity D(theta) = K(theta) * |dh/dtheta| [m^2/hr].

    This is the term explicitly OMITTED in the base FWC advection-only method
    (Ogden 2015, eq. 3 vs eq. 4). It is re-introduced as a correction in the
    Zhu et al. (2016) diffusion extension used in Section 6 of this file.

    Physically: D controls the smoothing of sharp moisture fronts and becomes
    important in fine-textured soils where the wetting front is diffuse.
    """
    if m is None:
        m = 1.0 - 1.0 / n
    K = hydraulic_conductivity(theta, theta_r, theta_s, Ks, alpha, n, l, m)
    abs_dh_dtheta = np.abs(dh_dtheta(theta, theta_r, theta_s, alpha, n, m))
    return K * abs_dh_dtheta


# =============================================================================
# SECTION 2: BIN DISCRETIZATION
# =============================================================================
# Ogden et al. (2015) transform the Richards PDE by discretizing the water
# content domain into N bins, leaving depth z as a continuous free variable.
# Each bin j represents a class of pore sizes characterised by a specific
# water content interval [theta_{j-1}, theta_j].
#
# Paper convention (Ogden 2015, Section 2):
#   - Bins numbered j = 1..N (0-indexed as j = 0..N-1 in this code)
#   - Bin j has RIGHT boundary at theta_j = theta_r + j * dtheta
#   - Hydraulic properties (h_j, K_j, D_j) evaluated at the right boundary
#   - theta_e = effective (drainable) porosity; upper limit of the domain
#
# Physical interpretation of bin order (0-indexed):
#   j = 0   : finest pores  -> theta low -> h HIGH -> K low
#   j = N-1 : coarsest pores -> theta high -> h LOW -> K high (= Ks)
#
# Early in an event, capillary suction (h) dominates: fine-pore bins (j=0)
# advance fastest. Late in an event, gravity (K) dominates: coarse-pore bins
# (j=N-1) advance fastest. This transition can cause front disorder (Section 7).

def setup_bins(theta_r, theta_e, N, Ks, alpha, n, l=0.5, m=None):
    """
    Discretize the water content domain [theta_r, theta_e] into N equal bins.

    Parameters
    ----------
    theta_r : residual water content [-]
    theta_e : effective (drainable) porosity [-]; upper bound of theta domain
    N       : number of bins (Ogden 2015 uses N=200; Lai 2015 shows N=10 suffices
              when coupling to simpler GA schemes)
    Ks      : saturated hydraulic conductivity [m/hr]
    alpha   : van Genuchten alpha [1/m]
    n       : van Genuchten n [-]
    l       : Mualem pore-connectivity parameter [-], default 0.5
    m       : van Genuchten m [-]; computed as 1-1/n if None

    Returns
    -------
    dict with:
        theta_bins : right boundary of each bin, shape (N,)
        h_bins     : capillary head at right boundary [m], shape (N,)
        K_bins     : hydraulic conductivity [m/hr], shape (N,)
        D_bins     : soil water diffusivity [m^2/hr], shape (N,)
        dtheta     : bin width [-]
        N, theta_r, theta_e : echoed back for downstream use
    """
    if m is None:
        m = 1.0 - 1.0 / n

    dtheta = (theta_e - theta_r) / N

    # Right boundary of bin j (1-indexed in paper -> 0-indexed here as j+1)
    j_idx = np.arange(1, N + 1)                         # 1..N
    theta_bins = theta_r + j_idx * dtheta               # shape (N,)

    # Hydraulic properties at bin right boundaries
    # Note: theta_e acts as theta_s in the Se calculation here because
    # the bin domain spans [theta_r, theta_e] by construction
    Se_bins = effective_saturation(theta_bins, theta_r, theta_e)
    h_bins  = vg_capillary_head(Se_bins, alpha, n, m)
    K_bins  = Ks * kr_mualem_vg(Se_bins, n, l, m)
    D_bins  = soil_water_diffusivity(theta_bins, theta_r, theta_e, Ks, alpha, n, l, m)

    return {
        'theta_bins': theta_bins,
        'h_bins':     h_bins,
        'K_bins':     K_bins,
        'D_bins':     D_bins,
        'dtheta':     dtheta,
        'N':          N,
        'theta_r':    theta_r,
        'theta_e':    theta_e,
    }


# =============================================================================
# SECTION 3: INFILTRATION FRONT ODE AND RK4 ADVANCEMENT
# =============================================================================
# Core of the FWC method. Each bin j has a wetting front at depth z_j(t).
# The ODE governing z_j is Darcy's law applied within the water-content space:
#
#   dz_j/dt = K_j * (z_j + h_j + Hp) / (dtheta * z_j)     [Ogden 2015, eq. 7]
#
# Physically: this is the Green-Ampt equation for a single pore-size class.
# The full Green-Ampt model is the special case N=1.
# The ensemble of N ODEs together describe the continuous moisture profile.

def _dzdt_single_bin(z_j, h_j, K_j, dtheta, Hp=0.0):
    """
    Right-hand side of the infiltration ODE for a single bin j.

    dz_j/dt = K_j * (z_j + h_j + Hp) / (dtheta * z_j)

    Driving forces in numerator:
        z_j : gravitational head (wetting front depth acts as hydraulic head)
        h_j : capillary suction head at bin right boundary [m]
        Hp  : ponded surface water depth [m]
    Resistance in denominator:
        dtheta * z_j : moisture deficit integrated over the wetted column

    Returns 0 if z_j <= 0 (bin not yet active).
    """
    if z_j <= 0.0:
        return 0.0
    return K_j * (z_j + h_j + Hp) / (dtheta * z_j)


def rk4_step_infiltration(z_fronts, h_bins, K_bins, dtheta, dt,
                           Hp=0.0, active=None):
    """
    Advance all wetting fronts one time step dt using 4th-order Runge-Kutta.

    Parameters
    ----------
    z_fronts : array (N,), current wetting front depths [m]
    h_bins   : array (N,), capillary head per bin right boundary [m]
    K_bins   : array (N,), hydraulic conductivity per bin [m/hr]
    dtheta   : scalar, bin width [-]
    dt       : time step [hr]
    Hp       : ponded depth at surface [m], 0 for non-ponded
    active   : bool array (N,); which bins to advance. None -> all where z_j > 0

    Returns
    -------
    z_new : array (N,), updated wetting front depths [m]
    """
    N = len(z_fronts)
    if active is None:
        active = z_fronts > 0.0

    def rhs(z):
        dz = np.zeros(N)
        for j in range(N):
            if active[j]:
                dz[j] = _dzdt_single_bin(z[j], h_bins[j], K_bins[j], dtheta, Hp)
        return dz

    k1 = rhs(z_fronts)
    k2 = rhs(z_fronts + 0.5 * dt * k1)
    k3 = rhs(z_fronts + 0.5 * dt * k2)
    k4 = rhs(z_fronts + dt * k3)

    z_new = z_fronts + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.maximum(z_new, 0.0)   # fronts cannot be negative


# =============================================================================
# SECTION 4: SURFACE FLUX — PRE-PONDING AND PONDED CONDITIONS
# =============================================================================
# Under ponded conditions (Hp > 0), all bins are active and advance at their
# full potential rate. Under sub-ponding rainfall (i < f_p), the total flux
# is constrained to the rainfall rate.
#
# Bin activation (Ogden 2015, Section 2.1):
#   When ponding begins, inactive bins (z_j = 0) are seeded with a small
#   initial depth derived from the analytical first-step solution:
#       z_j,seed = sqrt(2 * K_j * h_j * dt / dtheta)
#   This avoids the 1/z_j singularity in the ODE at z_j = 0.

def _seed_depth(K_j, h_j, dtheta, dt):
    """
    Analytical seed depth for bin j at the very first time step (z_j = 0).
    Derived by integrating dz/dt = K_j * h_j / (dtheta * z_j) for small z:
        z_j * dz_j = K_j * h_j / dtheta * dt  =>  z_j = sqrt(2*K_j*h_j*dt/dtheta)
    """
    return max(np.sqrt(2.0 * K_j * h_j * dt / dtheta), 1e-9)


def activate_ponded_bins(z_fronts, h_bins, K_bins, dtheta, dt):
    """
    Seed all inactive bins when ponding first occurs (Hp > 0, or i > f_p).
    Only affects bins with z_j == 0; already active bins are unchanged.
    """
    z_new = z_fronts.copy()
    for j in range(len(z_fronts)):
        if z_fronts[j] == 0.0:
            z_new[j] = _seed_depth(K_bins[j], h_bins[j], dtheta, dt)
    return z_new


def potential_infiltration_rate(z_fronts, h_bins, K_bins, dtheta, Hp=0.0):
    """
    Total potential infiltration rate f_p [m/hr]: the rate at which the soil
    CAN accept water given the current front positions and ponded depth.

    f_p = sum_j [ K_j * (z_j + h_j + Hp) / (dtheta * z_j) ]   [Ogden 2015, eq. 7]

    Only bins with z_j > 0 (active bins) contribute.
    """
    f_p = 0.0
    for j in range(len(z_fronts)):
        if z_fronts[j] > 0.0:
            f_p += K_bins[j] * (z_fronts[j] + h_bins[j] + Hp) / (dtheta * z_fronts[j])
    return f_p


def handle_surface_flux(z_fronts, h_bins, K_bins, dtheta, dt,
                         rainfall_rate, Hp, max_depth=10.0):
    """
    Determine ponding state and advance wetting fronts for one time step.

    Logic (Ogden 2015, Section 2.2):
    1. Compute potential rate f_p from current front depths.
    2a. PONDED  (rainfall >= f_p OR Hp > 0):
          - Activate all inactive bins (seed depths)
          - Advance fronts at full potential rate
          - Excess rainfall accumulates as ponded depth: Hp += (i - f_p)*dt
          - Existing Hp also drains at the actual infiltration rate
    2b. SUB-PONDED (rainfall < f_p AND Hp == 0):
          - Scale front advance so total flux = rainfall rate
          - Approximation: all bins scaled uniformly (Ogden 2015, p. 4286)
          - Hp remains 0

    Parameters
    ----------
    z_fronts     : array (N,), wetting front depths [m]
    h_bins       : array (N,), capillary heads [m]
    K_bins       : array (N,), hydraulic conductivities [m/hr]
    dtheta       : bin width [-]
    dt           : time step [hr]
    rainfall_rate: surface water flux [m/hr]; np.inf for constant ponding
    Hp           : current ponded depth [m]
    max_depth    : simulation domain depth [m]

    Returns
    -------
    z_new    : array (N,), updated wetting front depths
    Hp_new   : updated ponded depth [m]
    f_actual : actual infiltration rate this step [m/hr]
    """
    # --- Existing ponded depth drains first (treat like extra head) ---
    if Hp > 0.0:
        z_fronts = activate_ponded_bins(z_fronts, h_bins, K_bins, dtheta, dt)

    f_p = potential_infiltration_rate(z_fronts, h_bins, K_bins, dtheta, Hp)

    # Constant ponded condition signalled by np.inf
    if np.isinf(rainfall_rate):
        z_new  = rk4_step_infiltration(z_fronts, h_bins, K_bins, dtheta, dt, Hp)
        Hp_new = Hp          # maintained externally (constant ponding assumption)
        f_actual = f_p
        z_new = np.minimum(z_new, max_depth)
        return z_new, Hp_new, f_actual

    if rainfall_rate >= f_p:
        # ---- PONDING CASE ----
        # All bins active; advance at full potential rate
        z_fronts = activate_ponded_bins(z_fronts, h_bins, K_bins, dtheta, dt)
        # Recompute f_p after seeding (newly seeded bins now contribute)
        f_p = potential_infiltration_rate(z_fronts, h_bins, K_bins, dtheta, Hp)

        z_new = rk4_step_infiltration(z_fronts, h_bins, K_bins, dtheta, dt, Hp)

        # Net change in ponded depth: rain in minus infiltration out
        # If Hp > 0, it also drains at f_p (part of f_actual is from Hp)
        dHp = (rainfall_rate - f_p) * dt
        Hp_new = max(Hp + dHp, 0.0)
        f_actual = f_p

    else:
        # ---- SUB-PONDING CASE ----
        # Scale the effective time step so that total flux = rainfall_rate.
        # This is an approximation: in reality only a subset of bins should
        # be active, but uniform scaling preserves the relative front structure.
        if f_p > 1e-15:
            dt_eff = dt * (rainfall_rate / f_p)
        else:
            dt_eff = 0.0

        active = z_fronts > 0.0
        z_new = rk4_step_infiltration(
            z_fronts, h_bins, K_bins, dtheta, dt_eff, Hp=0.0, active=active
        )
        Hp_new   = 0.0
        f_actual = rainfall_rate

    z_new = np.minimum(z_new, max_depth)
    return z_new, Hp_new, f_actual


# =============================================================================
# SECTION 5: REDISTRIBUTION — FALLING SLUGS
# =============================================================================
# When rainfall ceases (or between events), infiltration fronts decouple from
# the surface and become "falling slugs": finite water parcels that continue
# moving downward under gravity alone (Talbot & Ogden 2008, Section 3).
#
# Each slug is associated with bin j and described by:
#   z_top : depth of the upper (drying) boundary [m]
#   z_bot : depth of the lower (wetting) boundary [m]
#
# Both boundaries advance at the gravity-only velocity:
#   dz/dt = K_j / dtheta    [Talbot & Ogden 2008, eq. 9]
#
# The slug maintains its length (z_bot - z_top) while falling, conserving
# water volume. When the top of a slug meets the next slug below it, they merge.

def init_slugs_from_fronts(z_fronts, K_bins, dtheta):
    """
    Convert active infiltration fronts to falling slugs when rainfall stops.
    Each active bin j yields one slug with:
        z_top = 0 (the slug extends from the surface)
        z_bot = z_j (current wetting front depth)

    Returns list of dicts [{'bin': j, 'z_top': float, 'z_bot': float}, ...]
    Only bins with z_j > 0 are included.
    """
    slugs = []
    for j, z_j in enumerate(z_fronts):
        if z_j > 0.0:
            slugs.append({
                'bin':   j,
                'z_top': 0.0,
                'z_bot': z_j,
            })
    return slugs


def advance_slugs(slugs, K_bins, dtheta, dt):
    """
    Advance all slugs under gravity for one time step.
    Both z_top and z_bot move at v_j = K_j / dtheta (gravity-only velocity).
    Slug length is preserved; mass is conserved by construction.

    Modifies slugs in place. Returns updated list.
    """
    for slug in slugs:
        j  = slug['bin']
        v  = K_bins[j] / dtheta        # gravity velocity [m/hr]
        slug['z_top'] += v * dt
        slug['z_bot'] += v * dt
    return slugs


def merge_slugs(slugs):
    """
    Merge any pair of slugs in the same bin that overlap in depth.
    Overlapping slugs arise when a faster-falling upper slug catches a
    slower lower slug (e.g. from a previous rainfall event).

    Merging conserves water volume: the merged slug spans from the shallowest
    z_top to the deepest z_bot of the overlapping pair.

    Returns cleaned slug list.
    """
    if not slugs:
        return slugs

    # Group by bin
    from collections import defaultdict
    by_bin = defaultdict(list)
    for s in slugs:
        by_bin[s['bin']].append(s)

    merged = []
    for j, bin_slugs in by_bin.items():
        # Sort by z_top (shallowest first)
        bin_slugs.sort(key=lambda s: s['z_top'])
        current = bin_slugs[0].copy()
        for nxt in bin_slugs[1:]:
            if nxt['z_top'] <= current['z_bot']:
                # Overlap: extend current slug to cover both
                current['z_bot'] = max(current['z_bot'], nxt['z_bot'])
            else:
                merged.append(current)
                current = nxt.copy()
        merged.append(current)

    return merged


def slugs_to_fronts(slugs, N):
    """
    Convert slug list back to a z_fronts array (deepest z_bot per bin).
    Used when rainfall resumes and fronts need to be re-activated.
    """
    z_fronts = np.zeros(N)
    for slug in slugs:
        j = slug['bin']
        z_fronts[j] = max(z_fronts[j], slug['z_bot'])
    return z_fronts


# =============================================================================
# SECTION 6: DIFFUSION CORRECTION (ZHU ET AL. 2016)
# =============================================================================
# The base FWC method (Ogden 2015) solves the ADVECTION-LIKE term of the Soil
# Moisture Velocity Equation (SMVE) and explicitly omits the DIFFUSION-LIKE
# term D(theta) * d^2theta/dz^2 (Ogden 2015, eq. 3 vs 4).
#
# Zhu et al. (2016) add back the diffusive term as an explicit correction
# applied AFTER each advection step. The diffusive flux between adjacent bins
# j and j+1 (at depths z_j and z_{j+1}) is:
#
#   q_diff_{j,j+1} = D_avg * dtheta / |z_j - z_{j+1}|    [m/hr per bin width]
#
# This flux moves water from the deeper (more advanced) front toward the
# shallower one, smoothing the piecewise moisture profile.
#
# Capillary relaxation (Section 7) must be applied BEFORE diffusion to ensure
# fronts are physically ordered (z_j >= z_{j+1} for j < j+1 under normal conditions).

def diffusion_correction(z_fronts, K_bins, D_bins, dtheta, dt):
    """
    Apply Zhu et al. (2016) diffusive correction to wetting front depths
    after the RK4 advection step.

    For each adjacent active pair (j, j+1):
      - Compute average diffusivity D_avg = 0.5*(D_j + D_{j+1})
      - Diffusive flux rate: q = D_avg * dtheta / dz^2  [1/hr]
        where dz = z_j - z_{j+1} (depth difference between fronts)
      - The correction dz_diff = q * dtheta * dt redistributes front depths:
        deeper front (z_j) slows slightly; shallower front (z_{j+1}) advances slightly
      - Applied symmetrically: z_j decreases, z_{j+1} increases by dz_diff/2

    Note: only applied where dz > a minimum threshold to avoid numerical issues.
    The correction is limited to at most half the depth difference to prevent
    fronts from crossing (which would require additional relaxation).

    Parameters
    ----------
    z_fronts : array (N,) wetting front depths [m] (post-advection)
    K_bins   : array (N,) hydraulic conductivities [m/hr]
    D_bins   : array (N,) soil water diffusivities [m^2/hr]
    dtheta   : bin width [-]
    dt       : time step [hr]

    Returns z_fronts with diffusion correction applied.
    """
    N = len(z_fronts)
    z_corr = z_fronts.copy()
    active = z_fronts > 0.0

    for j in range(N - 1):
        if not (active[j] and active[j + 1]):
            continue

        dz = z_corr[j] - z_corr[j + 1]   # should be >= 0 after relaxation
        if dz < 1e-6:
            continue

        D_avg = 0.5 * (D_bins[j] + D_bins[j + 1])

        # Diffusive correction magnitude [m]: limits front j from advancing too far
        # ahead of front j+1; Zhu eq. (6): dz_diff = D * dtheta / dz^2 * dtheta * dt
        dz_diff = D_avg * (dtheta ** 2) / (dz ** 2) * dt

        # Limit to at most half the current separation (prevents crossing)
        dz_diff = min(dz_diff, 0.5 * dz)

        # Deeper front (j) slows; shallower front (j+1) speeds up
        z_corr[j]     -= dz_diff
        z_corr[j + 1] += dz_diff

    return np.maximum(z_corr, 0.0)


# =============================================================================
# SECTION 7: CAPILLARY RELAXATION
# =============================================================================
# Ogden (2015), Section 3.4: under certain conditions (particularly when the
# water table rises rapidly, or late in an event when gravity-driven coarse-pore
# bins catch up to capillarity-driven fine-pore bins), the front profile can
# become "disordered": a bin j with HIGHER water content (coarser pores, lower
# capillary suction) has a DEEPER front than bin j-1 (finer pores).
#
# This is physically impossible: coarser pores cannot hold water at a depth
# where finer pores (with higher capillary suction) are not yet wet.
#
# Physical ordering: z_fronts[0] >= z_fronts[1] >= ... >= z_fronts[N-1]
# (finest pores, highest h -> deepest front; coarsest pores -> shallowest)
#
# Capillary relaxation restores this ordering by averaging the depths of
# disordered pairs while conserving total water volume:
#   z_avg = (z_j + z_{j+1}) / 2
# This is applied iteratively until the profile is fully ordered.

def capillary_relaxation(z_fronts):
    """
    Detect and correct disordered wetting fronts by capillary relaxation.
    (Ogden 2015, Section 3.4)

    Disorder condition: z_fronts[j] < z_fronts[j-1] is NORMAL (finer pores
    are deeper). Disorder occurs when z_fronts[j+1] > z_fronts[j], i.e. a
    coarser-pore bin has overtaken a finer-pore bin.

    Resolution: average the two disordered depths (conserves water volume),
    then re-check. Iterate until fully ordered (bubble-sort style).

    Only active bins (z > 0) are checked and corrected.
    """
    N = len(z_fronts)
    z_new = z_fronts.copy()
    active = z_new > 0.0

    # Bubble-sort style iteration: O(N^2) worst case, acceptable for N <= 200
    changed = True
    while changed:
        changed = False
        for j in range(N - 1):
            if not (active[j] and active[j + 1]):
                continue
            # Disorder: higher-theta bin (j+1) is deeper than lower-theta bin (j)
            if z_new[j + 1] > z_new[j]:
                z_avg = 0.5 * (z_new[j] + z_new[j + 1])
                z_new[j]     = z_avg
                z_new[j + 1] = z_avg
                changed = True

    return z_new


# =============================================================================
# SECTION 8: GROUNDWATER TABLE MODULE (OPTIONAL)
# =============================================================================
# Ogden (2015), Section 3.3: a rising water table generates upward-moving
# "groundwater fronts" in each bin. These are the mirror of infiltration fronts:
# they move upward from the water table as saturated water is pushed into the
# overlying vadose zone.
#
# Groundwater front velocity (upward, so depth decreases):
#   dz_gw_j/dt = -K_j * (z_gw_j - z_wt + h_j) / (dtheta * (z_gw_j - z_wt))
#                                                     [Ogden 2015, eq. 21]
#
# When an infiltration front z_inf_j meets a groundwater front z_gw_j from
# above (z_inf_j >= z_gw_j), the column is fully saturated in that bin.
#
# Equation 22 (Ogden 2015) modifies the capillary term during applied surface
# flux to account for groundwater: h_j is replaced by h_j + H_gw where H_gw
# is the groundwater head contribution. This is handled implicitly here by
# passing an effective Hp that includes the groundwater head.

def init_groundwater_fronts(water_table_depth, N):
    """
    Initialise groundwater front depths for all N bins at the water table.
    At t=0, all bins are saturated below the water table:
        z_gw_j = water_table_depth  for all j

    water_table_depth : depth to water table [m], positive downward
    Returns z_gw : array (N,)
    """
    return np.full(N, float(water_table_depth))


def _dzdt_gw_single(z_gw_j, h_j, K_j, dtheta, z_wt):
    """
    Upward velocity of groundwater front for bin j.
    [Ogden 2015, eq. 21]

    dz_gw_j/dt = -K_j * (z_gw_j - z_wt + h_j) / (dtheta * (z_gw_j - z_wt))

    Returns 0 if the front is already at or below the water table.
    Negative return value means the front moves upward (z decreases).
    """
    dz = z_gw_j - z_wt
    if dz <= 1e-9:
        return 0.0
    return -K_j * (dz + h_j) / (dtheta * dz)


def advance_groundwater_fronts(z_gw, h_bins, K_bins, dtheta, dt,
                                z_wt_old, z_wt_new):
    """
    Advance groundwater fronts one time step using RK4.
    Water table depth is linearly interpolated between z_wt_old and z_wt_new.

    When the water table rises (z_wt decreases), groundwater fronts propagate
    upward. If z_wt falls, groundwater fronts settle back toward z_wt.

    Parameters
    ----------
    z_gw     : array (N,) groundwater front depths [m]
    z_wt_old : water table depth at start of step [m]
    z_wt_new : water table depth at end of step [m]

    Returns z_gw_new : array (N,), updated groundwater front depths
    """
    N = len(z_gw)
    z_wt_mid = 0.5 * (z_wt_old + z_wt_new)

    def rhs_gw(z, z_wt):
        dz = np.zeros(N)
        for j in range(N):
            if z[j] > z_wt:
                dz[j] = _dzdt_gw_single(z[j], h_bins[j], K_bins[j], dtheta, z_wt)
        return dz

    k1 = rhs_gw(z_gw, z_wt_old)
    k2 = rhs_gw(z_gw + 0.5 * dt * k1, z_wt_mid)
    k3 = rhs_gw(z_gw + 0.5 * dt * k2, z_wt_mid)
    k4 = rhs_gw(z_gw + dt * k3, z_wt_new)

    z_gw_new = z_gw + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Groundwater front cannot be above the water table
    z_gw_new = np.maximum(z_gw_new, z_wt_new)
    return z_gw_new


def check_front_merging(z_inf, z_gw):
    """
    Check which bins have their infiltration front reaching the groundwater front.
    Returns bool array (N,): True where z_inf_j >= z_gw_j (fully saturated bin).
    When merged, the entire column in that bin is saturated; no further advancement.
    """
    return z_inf >= z_gw


# =============================================================================
# SECTION 9: PROFILE RECONSTRUCTION
# =============================================================================
# The N wetting front depths collectively define the moisture profile theta(z).
# At any depth z, theta(z) = theta_r + dtheta * (number of bins whose front
# has passed depth z), i.e. each bin contributes its dtheta increment once its
# front z_j >= z.
# This produces a staircase approximation of the continuous Richards profile.
# As N -> infinity, the staircase converges to the smooth moisture front.

def reconstruct_theta_profile(z_fronts, theta_r, dtheta, max_depth, n_z=500):
    """
    Reconstruct the piecewise moisture profile theta(z) from bin front depths.

    At depth z: theta(z) = theta_r + sum_{j: z_fronts[j] >= z} dtheta

    Parameters
    ----------
    z_fronts  : array (N,), wetting front depths at current time [m]
    theta_r   : residual water content [-]
    dtheta    : bin width [-]
    max_depth : simulation domain depth [m]
    n_z       : number of depth points for output grid

    Returns (z_grid [m], theta_grid [-])
    """
    z_grid = np.linspace(0.0, max_depth, n_z)
    theta_grid = np.array([
        theta_r + dtheta * np.sum(z_fronts >= z)
        for z in z_grid
    ])
    return z_grid, theta_grid


# =============================================================================
# SECTION 10: MAIN FWC TIME-STEPPING DRIVER
# =============================================================================

def run_fwc(soil_params, forcing, sim_params):
    """
    Full FWC simulation driver following Ogden et al. (2015).

    Per time step the sequence is:
        1. Surface flux handler  (Section 4) — ponding / sub-ponding logic
        2. Diffusion correction  (Section 6) — Zhu et al. (2016), optional
        3. Capillary relaxation  (Section 7) — restore physical front ordering
        4. Groundwater module    (Section 8) — optional
        5. Redistribution        (Section 5) — falling slugs if no rainfall

    Parameters
    ----------
    soil_params : dict
        theta_r    : residual water content [-]
        theta_e    : effective (drainable) porosity [-]
        theta_init : initial water content [-], default = theta_r
        Ks         : saturated hydraulic conductivity [m/hr]
        alpha      : van Genuchten alpha [1/m]
        n          : van Genuchten n [-]
        l          : Mualem l [-], default 0.5

    forcing : dict
        time       : array [hr], time vector (length T)
        rainfall   : array [m/hr], length T; use np.inf for constant ponding;
                     use 0.0 for redistribution-only periods
        water_table: array [m] depth to water table, length T, or None

    sim_params : dict
        N               : number of bins, default 200
        dt              : time step [hr]
        use_diffusion   : bool, apply Zhu et al. (2016) correction, default True
        use_groundwater : bool, enable groundwater module, default False
        max_depth       : simulation domain depth [m], default 10.0

    Returns
    -------
    dict with keys:
        time          : time vector [hr]
        f_rate        : infiltration rate [m/hr], shape (T,)
        F_cumulative  : cumulative infiltration [m], shape (T,)
        Hp            : ponded depth [m], shape (T,)
        z_fronts_all  : wetting front depths (N x T)
        theta_profile : (z_grid, theta_grid) at final time step
        bins          : bin setup dict from setup_bins()
    """
    # ---- Unpack soil parameters ----
    theta_r    = soil_params['theta_r']
    theta_e    = soil_params['theta_e']
    theta_init = soil_params.get('theta_init', theta_r)
    Ks         = soil_params['Ks']
    alpha      = soil_params['alpha']
    n_vg       = soil_params['n']
    l_vg       = soil_params.get('l', 0.5)
    m_vg       = 1.0 - 1.0 / n_vg

    # ---- Unpack simulation parameters ----
    N               = sim_params.get('N', 200)
    dt              = sim_params['dt']
    use_diffusion   = sim_params.get('use_diffusion', True)
    use_groundwater = sim_params.get('use_groundwater', False)
    max_depth       = sim_params.get('max_depth', 10.0)

    # ---- Unpack forcing ----
    time_vec  = forcing['time']
    rain_vec  = forcing['rainfall']
    wt_vec    = forcing.get('water_table', None)
    n_steps   = len(time_vec)

    # ---- Set up bins ----
    bins       = setup_bins(theta_r, theta_e, N, Ks, alpha, n_vg, l_vg, m_vg)
    h_bins     = bins['h_bins']
    K_bins     = bins['K_bins']
    D_bins     = bins['D_bins']
    dtheta     = bins['dtheta']
    theta_bins = bins['theta_bins']

    # ---- Initialise wetting fronts ----
    # Bins whose right boundary <= theta_init are already moist: seed them.
    # Bins drier than theta_init are inactive (z = 0).
    z_fronts = np.zeros(N)
    pre_wet  = theta_bins <= theta_init
    for j in np.where(pre_wet)[0]:
        z_fronts[j] = _seed_depth(K_bins[j], h_bins[j], dtheta, dt)

    Hp    = 0.0         # ponded depth [m]
    slugs = []          # list of falling slugs (populated during dry periods)

    # ---- Groundwater initialisation (optional) ----
    if use_groundwater and wt_vec is not None:
        z_gw = init_groundwater_fronts(wt_vec[0], N)
    else:
        z_gw = None

    # ---- Output arrays ----
    f_rate_out   = np.zeros(n_steps)
    F_cum_out    = np.zeros(n_steps)
    Hp_out       = np.zeros(n_steps)
    z_fronts_all = np.zeros((N, n_steps))
    F_cum        = 0.0

    # ---- Time loop ----
    for i in range(n_steps):
        rain_i = rain_vec[i]

        if rain_i > 0.0 or np.isinf(rain_i):
            # ---- RAINFALL OR PONDING PERIOD ----
            # If we have slugs from a previous dry period, re-absorb them as fronts.
            if slugs:
                z_from_slugs = slugs_to_fronts(slugs, N)
                # Merge slug fronts with any pre-existing fronts (take the deeper)
                z_fronts = np.maximum(z_fronts, z_from_slugs)
                slugs = []

            # Step 1: advance infiltration fronts
            z_fronts, Hp, f_actual = handle_surface_flux(
                z_fronts, h_bins, K_bins, dtheta, dt, rain_i, Hp, max_depth
            )

        else:
            # ---- DRY PERIOD: redistribution only ----
            # Convert active fronts to slugs on first dry step
            if not slugs and np.any(z_fronts > 0.0):
                slugs = init_slugs_from_fronts(z_fronts, K_bins, dtheta)
                z_fronts = np.zeros(N)   # fronts now tracked as slugs

            # Advance slugs under gravity
            slugs = advance_slugs(slugs, K_bins, dtheta, dt)
            slugs = merge_slugs(slugs)
            # No infiltration during dry period
            f_actual = 0.0
            # Drain any remaining Hp through soil (use f_p for empty fronts)
            if Hp > 0.0:
                Hp = max(Hp - potential_infiltration_rate(
                    z_fronts, h_bins, K_bins, dtheta, Hp) * dt, 0.0)

        # Step 2: diffusion correction (Zhu et al. 2016)
        if use_diffusion and np.any(z_fronts > 0.0):
            z_fronts = diffusion_correction(z_fronts, K_bins, D_bins, dtheta, dt)

        # Step 3: capillary relaxation (restore physical front ordering)
        if np.any(z_fronts > 0.0):
            z_fronts = capillary_relaxation(z_fronts)

        # Step 4: groundwater module (optional)
        if use_groundwater and z_gw is not None and wt_vec is not None:
            z_wt_old = wt_vec[max(0, i - 1)]
            z_wt_new = wt_vec[i]
            z_gw = advance_groundwater_fronts(
                z_gw, h_bins, K_bins, dtheta, dt, z_wt_old, z_wt_new
            )
            # Merge: where infiltration front has reached groundwater front,
            # set infiltration front to groundwater depth (column fully saturated)
            merged_mask = check_front_merging(z_fronts, z_gw)
            z_fronts[merged_mask] = z_gw[merged_mask]

        # Step 5: enforce domain limit
        z_fronts = np.minimum(z_fronts, max_depth)

        # ---- Accumulate outputs ----
        F_cum          += f_actual * dt
        f_rate_out[i]   = f_actual
        F_cum_out[i]    = F_cum
        Hp_out[i]       = Hp
        z_fronts_all[:, i] = z_fronts

    # ---- Reconstruct final moisture profile ----
    theta_profile = reconstruct_theta_profile(
        z_fronts_all[:, -1], theta_r, dtheta, max_depth
    )

    return {
        'time':          time_vec,
        'f_rate':        f_rate_out,
        'F_cumulative':  F_cum_out,
        'Hp':            Hp_out,
        'z_fronts_all':  z_fronts_all,
        'theta_profile': theta_profile,
        'bins':          bins,
    }


# =============================================================================
# SECTION 11: GREEN-AMPT METHODS FOR COMPARISON
# =============================================================================
# Standard GA (Newton-Raphson) and modified analytical GA (Sun et al. 2023)
# retained from the earlier conversation for direct comparison with FWC.
# Units must match FWC: [m] and [m/hr].

def green_ampt_NR(time, Ks, psi, delta_theta, tol=1e-10, max_iter=200):
    """
    Standard Green-Ampt cumulative infiltration F(t) by Newton-Raphson.

    Solves the implicit equation:
        F = Ks*t + M*ln(1 + F/M),  where M = psi * delta_theta
    at each time step independently (no error accumulation).

    Parameters: Ks [m/hr], psi [m], delta_theta [-]
    Returns dict: time, F_cumulative [m], f_rate [m/hr]
    """
    M      = psi * delta_theta
    n_t    = len(time)
    F_cum  = np.zeros(n_t)
    f_rate = np.zeros(n_t)

    F_prev = Ks * time[0] + 1e-9
    for i in range(n_t):
        F_est = F_prev
        for _ in range(max_iter):
            g   = F_est - Ks * time[i] - M * np.log(1.0 + F_est / M)
            gp  = 1.0 - M / (M + F_est)
            dF  = g / gp
            F_est = max(F_est - dF, 1e-10)
            if abs(dF) < tol:
                break
        F_cum[i]  = F_est
        f_rate[i] = Ks * (1.0 + M / F_est)
        F_prev    = F_est

    return {'time': time, 'F_cumulative': F_cum, 'f_rate': f_rate}


def modified_ga_sun2023(time, Ks, Sf, delta_theta, H_p=0.0):
    """
    Analytical modified Green-Ampt from Sun et al. (2023), equations 15-17.

    Closed-form sqrt(t) solution valid under the suction-dominated assumption
    (Sf >> Z_f). Derived by linearising the force balance and integrating
    the resulting separable ODE (see paper derivation in conversation).

        Z_f = 4*sqrt(2) * sqrt(Ks_eff*(Sf+H)*t / ((4+pi)*delta_theta))

    where Ks_eff = 0.5*Ks (Bouwer 1966 effective conductivity at wetting front).

    Parameters: Ks [m/hr], Sf [m], delta_theta [-], H_p [m] ponded depth
    Returns dict: time, Zf [m], f_rate [m/hr], F_cumulative [m]
    """
    Ks_eff = 0.5 * Ks
    t      = np.maximum(time, 1e-10)
    Zf     = 4.0 * np.sqrt(2.0) * np.sqrt(
        Ks_eff * (Sf + H_p) * t / ((4.0 + np.pi) * delta_theta)
    )
    f_rate = Ks_eff * (Zf + Sf + H_p) / np.maximum(Zf, 1e-10)
    CI     = ((4.0 + np.pi) / 8.0) * Zf * delta_theta

    return {'time': time, 'Zf': Zf, 'f_rate': f_rate, 'F_cumulative': CI}


# =============================================================================
# SECTION 12: COMPARISON WRAPPER
# =============================================================================

def compare_ga_fwc(soil_params, forcing, sim_params):
    """
    Run FWC, standard GA (NR), and modified GA (Sun 2023) on the same inputs.

    soil_params must include both:
        FWC parameters : theta_r, theta_e, theta_init, Ks, alpha, n, l
        GA parameters  : psi [m], delta_theta [-] (or derived from theta_e, theta_init)
        Sun 2023 param : Sf [m] (modified suction head; defaults to psi if absent)

    Returns
    -------
    dict with keys 'fwc', 'ga_nr', 'ga_sun', each containing their result dict,
    plus 'time' for convenience.
    """
    time_vec = forcing['time']

    # ---- FWC ----
    res_fwc = run_fwc(soil_params, forcing, sim_params)

    # ---- Standard GA (Newton-Raphson) ----
    psi = soil_params.get('psi')
    if psi is None:
        raise KeyError("soil_params must include 'psi' [m] for Green-Ampt comparison")

    delta_theta = soil_params.get(
        'delta_theta',
        soil_params['theta_e'] - soil_params.get('theta_init', soil_params['theta_r'])
    )
    Ks = soil_params['Ks']

    # GA is defined for ponded conditions (t=0 ponded); use time vector directly
    res_ga_nr = green_ampt_NR(time_vec, Ks, psi, delta_theta)

    # ---- Modified GA (Sun et al. 2023) ----
    Sf = soil_params.get('Sf', psi)   # use psi as proxy for Sf if not given
    res_ga_sun = modified_ga_sun2023(time_vec, Ks, Sf, delta_theta)

    return {
        'fwc':    res_fwc,
        'ga_nr':  res_ga_nr,
        'ga_sun': res_ga_sun,
        'time':   time_vec,
    }