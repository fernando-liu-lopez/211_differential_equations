import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kirchhoff\'s Current Law", layout="centered")

# -----------------------------
# Helper: draw simple circuit primitives (matplotlib)
# -----------------------------
def draw_resistor(ax, x0, y0, x1, y1, label=None):
    """Draw a zig-zag-ish resistor between (x0,y0) and (x1,y1). Assumes mostly horizontal/vertical."""
    # For simplicity, handle horizontal or vertical.
    if abs(y1 - y0) < 1e-9:
        # horizontal
        xs = np.linspace(x0, x1, 9)
        ys = np.full_like(xs, y0)
        # zigzag in middle segments
        amp = 0.15
        for i in range(2, 7):
            ys[i] = y0 + (amp if i % 2 == 0 else -amp)
        ax.plot(xs, ys, lw=2, color="black")
        if label:
            ax.text((x0 + x1)/2, y0 + 0.35, label, ha="center", va="bottom", fontsize=11)
    elif abs(x1 - x0) < 1e-9:
        # vertical
        ys = np.linspace(y0, y1, 9)
        xs = np.full_like(ys, x0)
        amp = 0.15
        for i in range(2, 7):
            xs[i] = x0 + (amp if i % 2 == 0 else -amp)
        ax.plot(xs, ys, lw=2, color="black")
        if label:
            ax.text(x0 + 0.35, (y0 + y1)/2, label, ha="left", va="center", fontsize=11)
    else:
        # diagonal fallback (simple line)
        ax.plot([x0, x1], [y0, y1], lw=2, color="black")
        if label:
            ax.text((x0 + x1)/2, (y0 + y1)/2, label, ha="center", va="bottom", fontsize=11)


def draw_wire(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], lw=2, color="black")


def draw_node(ax, x, y, label=None, offset = 0.15, position="up"):
    ax.scatter([x], [y], s=60, color="black")
    if not label:
        return
    if position == "up":
            dx, dy = 0.0, offset
            ha, va = "center", "bottom"
    elif position == "down":
        dx, dy = 0.0, -offset
        ha, va = "center", "top"
    elif position == "left":
        dx, dy = -offset, 0.0
        ha, va = "right", "center"
    elif position == "right":
        dx, dy = offset, 0.0
        ha, va = "left", "center"
    else:
        raise ValueError("position must be 'up', 'down', 'left', or 'right'")
    ax.text(
        x + dx,
        y + dy,
        label,
        fontsize=12,
        ha=ha,
        va=va,
    )

def draw_ground(ax, x, y, label=None, orientation="horizontal"):
    """
    Draw a ground symbol centered at (x, y).

    Parameters
    ----------
    ax : matplotlib axis
    x, y : float
        Center of the ground symbol
    label : str or None
        Optional label text
    orientation : {"horizontal", "vertical"}
        Orientation of the ground bars
    """

    if orientation == "horizontal":
        ax.plot([x - 0.30, x + 0.30], [y + 0.12, y + 0.12], lw=2, color="black")
        ax.plot([x - 0.2, x + 0.2], [y, y], lw=2, color="black")

        if label:
            ax.text(x + 0.45, y, label, fontsize=11, va="center")

    elif orientation == "vertical":
        # Three vertical bars, centered at x
        ax.plot([x + 0.12, x + 0.12], [y - 0.20, y + 0.20], lw=2, color="black")
        ax.plot([x, x], [y - 0.3, y + 0.3], lw=2, color="black")

        if label:
            ax.text(x, y - 0.45, label, fontsize=11, ha="center", va="top")

    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")


def draw_current_source(ax, x0, y0, x1, y1, label="I", offset=0.15, position="right"):
    """
    Draw a current source as an arrow on a wire from (x0, y0) to (x1, y1).

    Parameters
    ----------
    ax : matplotlib Axes
    x0, y0, x1, y1 : float
        Endpoints of the wire (arrow points from (x0,y0) to (x1,y1))
    label : str or None
        Label for the current (e.g. r"$I_1$")
    position : {"up", "down", "left", "right"}
        Where to place the label relative to the arrow
    """
    # Draw the wire
    ax.plot([x0, x1], [y0, y1], lw=2, color="black")

    # Direction vector
    dx = x1 - x0
    dy = y1 - y0

    # Place arrow near the middle of the wire
    arrow_x = x0 + 0.4 * dx
    arrow_y = y0 + 0.4 * dy

    ax.arrow(
        arrow_x,
        arrow_y,
        0.12 * dx,
        0.12 * dy,
        head_width=0.12,
        head_length=0.18,
        linewidth=2,
        color="black",
        length_includes_head=True,
    )

    if not label:
        return

    if position == "up":
        lx, ly = arrow_x, arrow_y + offset
        ha, va = "center", "bottom"
    elif position == "down":
        lx, ly = arrow_x, arrow_y - offset
        ha, va = "center", "top"
    elif position == "left":
        lx, ly = arrow_x - offset, arrow_y
        ha, va = "right", "center"
    elif position == "right":
        lx, ly = arrow_x + offset, arrow_y
        ha, va = "left", "center"
    else:
        raise ValueError("position must be 'up', 'down', 'left', or 'right'")

    ax.text(
        lx,
        ly,
        label,
        fontsize=11,
        ha=ha,
        va=va,
    )

def centered_pyplot(fig, width_ratio=2):
    left, center, right = st.columns([1, width_ratio, 1])
    with center:
        st.pyplot(fig,use_container_width=False)
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

st.title("Kirchhoff’s Current Law")
st.markdown(
    """Kirchhoff’s Current Law states that at any node/junction in an electrical circuit, the sum of currents leaving the node equals the sum of currents entering. """)
st.markdown("""It is standard to label currents with the letter 'I', in which case Kirchhoff's Current Law states:""")

st.latex(r"\sum I_{in} = \sum I_{out}")
st.markdown("""For example, the junction drawn below satisfies:""")
st.latex(r"I_1 = I_2 + I_3.")
# draw circuit 1
fig0, ax0 = plt.subplots(figsize=(3,3))
ax0.axis("off")

draw_current_source(ax0, 0, 0, 2, 0,label=r"$I_1$", position='up')
draw_node(ax0, 2, 0, label=r"$J$", position='up')
draw_current_source(ax0, 2, 0, 4, 1, label=r"$I_2$", position='up')
draw_current_source(ax0, 2, 0, 4, -1, label=r"$I_3$", position='down')

centered_pyplot(fig0)
plt.close(fig0)

st.divider()


# -----------------------------
# Example 
# -----------------------------

st.subheader("Turning circuits into systems of linear equations")

st.markdown("""A circuit is drawn below. The parallel vertical lines represent the voltage source, with the different sizes determining the direction in which the current flows (in this case, clockwise).""")

# draw circuit 1
fig1, ax1 = plt.subplots(figsize=(4, 4))
ax1.axis("off")

draw_ground(ax1, 2, 0, orientation="vertical")
draw_wire(ax1, 2, 0, 0, 0)
draw_wire(ax1, 0, 0, 0, 2)
draw_wire(ax1, 0, 2, 0.5, 2)
draw_node(ax1, 0.5, 2, label=r"$J_1$")
draw_current_source(ax1, 0.5, 2, 2, 3, label=r"$I_2$", position='up')
draw_current_source(ax1, 0.5, 2, 2, 1, label=r"$I_3$", position='down')
draw_node(ax1, 2, 3, label=r"$J_2$")
draw_node(ax1, 2, 1, label=r"$J_3$", position="down")
draw_current_source(ax1, 2, 3, 3.5, 2, label=r"$I_5$", position='up')
draw_current_source(ax1, 2, 1, 3.5, 2, label=r"$I_6$", position="down")
draw_current_source(ax1, 2, 1, 2, 3, label=r"$I_4$")
draw_node(ax1, 3.5, 2, label=r"$J_4$")
draw_wire(ax1, 3.5, 2, 4, 2)
draw_current_source(ax1, 4, 2, 4, 0, label=r"$I_1$", position="right")
draw_wire(ax1, 4, 0, 2.15, 0)
draw_current_source(ax1, 0, 0, 0, 2, label=r"$I_1$")

ax1.set_xlim(-0.5, 4.5)
ax1.set_ylim(-0.5, 3.5)
centered_pyplot(fig1)
plt.close(fig1)

st.markdown("""1. Using Kirchhoff's Current Law at each node, write a system of linear equations to describe the currents in this circuit.""")
st.markdown("""2. Write the augmented matrix for the SLEs and transform it to RREF.""")
