import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation



'''compare = pd.DataFrame({'layered':layered_inf, 'unlayered':unlayered_inf})
compare['time_hr'] = time_vec* t_max*24
compare['layered'] = compare['layered']/24
compare['unlayered'] = compare['unlayered']/24

#creatre plot
soil = 'Zd21'
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(compare['time_hr'], compare['unlayered'], label='uniforme bodem', color='C0', linewidth=2)
ax.plot(compare['time_hr'], compare['layered'],   label='gelaagde bodem',   color='C1', linewidth=2)
ax.set_title(f'Effect van een gelaagde bodem voor bodem {soil}')
ax.set_xlabel('Tijd (uur)')
ax.set_ylabel('Infiltratie (mm/uur)')
ax.legend()
'''



def plot_field_vs_wilting(
    merged,
    soil_col='soil_code',
    field_col='field_capacity',
    wilt_col='wilting_point',
    figsize=(12, 6)
):
    x = merged[soil_col].astype(str)
    width = 0.35
    idx = np.arange(len(x))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Infiltratie voor bodems op veldcapaciteit en verwelkingspunt')
    ax.bar(idx - width / 2, merged[field_col], width, label='Veldcapaciteit')
    ax.bar(idx + width / 2, merged[wilt_col], width, label='Verwelkingspunt')

    ax.set_xticks(idx)
    ax.set_xticklabels(x, rotation=90)
    ax.set_xlabel('Staringreeks')
    ax.set_ylabel('Infiltratie (mm/uur)')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _make_twin_axis(ax, position=None, side="right"):
    twin = ax.twinx()
    if position is not None:
        twin.spines[side].set_position(("outward", position))
    if side == "left":
        twin.spines["right"].set_visible(False)
        twin.set_frame_on(True)
        twin.patch.set_visible(False)
        twin.yaxis.set_label_position("left")
        twin.yaxis.tick_left()
    return twin


def plot_evaluation(cum_inf, frontspeed, Hp_top_list, Hp_bot_list, max_bin_list,
                    title="Cumulative Infiltration, Front speed, Ponded water top, ponded water bot, Max bin",
                    figsize=(8, 5), savepath=None):
    """
    Plot evaluation metrics with two left y-axes and two right y-axes.
    Returns: fig, (ax_left1, ax_left2, ax_right1, ax_right2)
    """
    fig, ax_left1 = plt.subplots(figsize=figsize)

    # main left axis for front_speed
    l1, = ax_left1.plot(frontspeed, color="tab:green", label="front_speed", zorder=1)
    ax_left1.set_xlabel("Timestep")
    ax_left1.set_ylabel("front_speed", color="tab:green")
    ax_left1.tick_params(axis="y", labelcolor="tab:green")

    # second left axis for cum_inf, plotted on top
    ax_left2 = _make_twin_axis(ax_left1, position=60, side="left")
    l2, = ax_left2.plot(cum_inf, color="tab:blue", label="cum_inf", zorder=2)
    ax_left2.set_ylabel("cum_inf", color="tab:blue")
    ax_left2.tick_params(axis="y", labelcolor="tab:blue")

    # right axis 1 for Hp (both top and bot on same axis)
    ax_right1 = ax_left1.twinx()
    l3, = ax_right1.plot(Hp_top_list, color="tab:orange", label="Ponded depth top")
    l5, = ax_right1.plot(Hp_bot_list, color="pink", label="Ponded depth bot")
    ax_right1.set_ylabel("Ponded depth", color="tab:orange")
    ax_right1.tick_params(axis="y", labelcolor="tab:orange")

    # right axis 2 for max_bin, moved outward
    ax_right2 = _make_twin_axis(ax_left1, position=60, side="right")
    l4, = ax_right2.plot(max_bin_list, color="tab:purple", label="max_bin")
    ax_right2.set_ylabel("max_bin", color="tab:purple")
    ax_right2.tick_params(axis="y", labelcolor="tab:purple")

    lines = [l1, l2, l3, l4, l5]
    labels = [ln.get_label() for ln in lines]
    ax_left1.legend(lines, labels, loc='center right')

    if title:
        fig.suptitle(title)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")

    return fig, (ax_left1, ax_left2, ax_right1, ax_right2)



def animate_fronts(z_history, profiles, profile_name, Hp_array, interval=50):
    """
    Animate wetting front depths over time for a layered soil profile.

    Each layer is plotted in absolute depth on a shared axis, with the
    theta_bins of each layer mapped to the range [0, 1] on the x-axis.
    Layer interfaces are marked with horizontal grey lines.

    Parameters
    ----------
    z_history    : np.ndarray, shape (t_steps, N, n_layers)
                   Wetting front depths per bin per layer per time step [cm].
    profiles     : dict
                   The profiles dict from main.py. Must contain profile_name
                   as a key, whose value is a list of soil layer dicts. Each
                   layer dict must contain:
                       'theta_bins' : array (N,)
                       'theta_r'    : float
                       'theta_e'    : float
                       'thickness'  : float  [cm]
    profile_name : str
                   Key into profiles to select the soil profile to animate.
    interval     : int, optional
                   Time between animation frames in milliseconds. Default 50.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
           The animation object. Keep a reference to prevent garbage collection.
    """
    layers     = profiles[profile_name]
    n_layers   = len(layers)
    t_steps    = z_history.shape[0]
    #N          = z_history.shape[1]

    # --- Compute absolute depth offsets for each layer ---
    # Layer 0 starts at depth 0; each subsequent layer starts where the
    # previous one ends.
    layer_tops = np.zeros(n_layers)
    for k in range(1, n_layers):
        layer_tops[k] = layer_tops[k - 1] + layers[k - 1]['thickness']

    total_depth = layer_tops[-1] + layers[-1]['thickness']

    theta_raw = [layers[k]['theta_bins'] for k in range(n_layers)]    

    # --- Colour palette: one colour per layer ---
    cmap   = plt.cm.tab10
    colors = [cmap(k) for k in range(n_layers)]

    # --- Set up figure ---
    fig, ax = plt.subplots(figsize=(7, 8))

    ax.set_xlabel("Soil moisture θ [-]")    
    ax.set_ylabel("Depth [cm]")
    ax.set_xlim(0, 1)
    ax.set_ylim(30, -total_depth * 0.02)   # invert: 0 at top Change displayed depth
    ax.set_title(f"Profile: {'Zand/B01'}  |  time (minutes) 1 / {round(t_steps/4)}")

    # --- Draw layer interface lines ---
    for k in range(1, n_layers):
        ax.axhline(
            y       = layer_tops[k],
            color   = 'grey',
            lw      = 0.8,
            ls      = '--',
            zorder  = 1,
            label   = f'Interface {k}' if k == 1 else '_nolegend_'
        )
        ax.text(
            1.01, layer_tops[k],
            f'Layer {k} / {k+1}',
            va        = 'center',
            ha        = 'left',
            fontsize  = 7,
            color     = 'grey',
            transform = ax.get_yaxis_transform()
        )

    #draw vertical lines at theta_e and theta_r    
    for k in range(n_layers):
        top    = layer_tops[k]
        bottom = layer_tops[k] + layers[k]['thickness']

        ax.plot(
            [layers[k]['theta_r'], layers[k]['theta_r']], [top, bottom],
            color = colors[k], lw = 1.0, ls = ':',  zorder = 1,
            label = 'θr' if k == 0 else '_nolegend_'
        )
        ax.plot(
            [layers[k]['theta_e'], layers[k]['theta_e']], [top, bottom],
            color = colors[k], lw = 1.0, ls = '-.', zorder = 1,
            label = 'θe' if k == 0 else '_nolegend_'
        )
        ax.plot(
            [layers[k]['theta_init'], layers[k]['theta_init']], [top, bottom],
            color = colors[k], lw = 1.0, ls = '--', zorder = 1,
            label = 'θi' if k == 0 else '_nolegend_'
        )
        

    # One text per interface: surface (top of layer 0) + between layers + bottom
    hp_texts = []
    # Interface positions in absolute depth: surface=0, then layer tops from layer 1 onward
    interface_depths = [0.0] + [layer_tops[k] for k in range(1, n_layers)] + [total_depth]

    for idx, depth in enumerate(interface_depths):
        txt = ax.text(
            0.98, depth,                     # left side of plot, at interface depth
            f'Hp[{idx}] = 0.000 cm',
            va        = 'center',
            ha        = 'right',
            fontsize  = 7,
            color     = 'black',
            bbox      = dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
            zorder    = 3
        )
        hp_texts.append(txt)

    # --- Draw shaded background bands to distinguish layers ---
    for k in range(n_layers):
        top    = layer_tops[k]
        bottom = layer_tops[k] + layers[k]['thickness']
        ax.axhspan(
            top, bottom,
            alpha    = 0.04,
            color    = colors[k],
            zorder   = 0
        )

    # --- Initialise one line per layer ---
    lines = []
    for k in range(n_layers):
        line, = ax.plot(
            [], [],
            color  = colors[k],
            lw     = 1.5,
            marker = 'o',
            ms     = 2,
            label  = f'Layer {k + 1}',
            zorder = 2
        )
        lines.append(line)

    # --- Legend ---
    # Add a dummy handle for the interface line
    interface_handle = mpatches.Patch(
        facecolor = 'none',
        edgecolor = 'grey',
        linestyle = '--',
        linewidth = 0.8,
        label     = 'Layer interface'
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles  = handles + [interface_handle],
        labels   = labels  + ['Layer interface'],
        loc      = 'lower right',
        fontsize = 8
    )

    # --- Animation update function ---
    def update(frame):
        for k in range(n_layers):
            z_k     = z_history[frame, :, k]           # front depths this frame, this layer
            x_k     = theta_raw[k]                    # normalised theta for this layer
            offset  = layer_tops[k]                    # absolute depth offset

            # Only plot bins that have a non-zero front depth.
            # Fronts at zero are inactive; fronts at soil thickness are
            # fully saturated through the layer. Both are plotted:
            #   - zero front: excluded (not yet reached)
            #   - thickness front: plotted at the layer bottom
            active = z_k > 0.0
            if np.any(active):
                lines[k].set_xdata(x_k[active])
                lines[k].set_ydata(z_k[active] + offset)
            else:
                lines[k].set_xdata([])
                lines[k].set_ydata([])

        # Update Hp counters
        for idx, txt in enumerate(hp_texts):
            hp_val = Hp_array[frame, idx] if frame < Hp_array.shape[0] else 0.0
            txt.set_text(f'Hp[{idx}] = {hp_val:.3f} cm')

        ax.set_title(
            f"Profile: {'Zand/B01'}  |  "
            f"time (minutes) {round((frame + 1)/4)} / {round(t_steps/4)}"
        )

        return lines + hp_texts     # include texts in return so blit works if enabled later

    anim = FuncAnimation(
        fig,
        update,
        frames   = t_steps,
        interval = interval,
        blit     = False
    )

    plt.tight_layout()
    plt.show()

    return anim