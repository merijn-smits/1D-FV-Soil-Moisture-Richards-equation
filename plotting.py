import matplotlib.pyplot as plt


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