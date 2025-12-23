import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -----------------------------
# Stylistic functions
# -----------------------------

def centered_pyplot(fig, width_ratio=2):
    """
    Centered the matplotlib figure in the streamlit.
    """
    left, center, right = st.columns([1, width_ratio, 1])
    with center:
        st.pyplot(fig,use_container_width=False)


# -----------------------------
# Math Functions
# -----------------------------
def make_function(expr: str):
    """
    Given a string expression f(t, y), return a function f(t, y)
    that can be evaluated on numpy arrays or scalars, using numpy math.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Expression for dy/dt is empty.")

    # We'll allow numpy functions: sin, cos, exp, log, etc.
    # t and y will be in the local namespace.
    def f(t, y):
        local_dict = {"t": t, "y": y, "np": np}
        # Expose common numpy functions directly for convenience
        for name in ["sin", "cos", "tan", "exp", "log", "sqrt", "pi"]:
            local_dict[name] = getattr(np, name, None)
        return eval(expr, {"__builtins__": {}}, local_dict)

    return f

def integrate(f_scalar, t0, y0, t_min, t_max, n_steps=1000):
    """
    Runge-Kutta (RK4) integrator that integrates both forward and backward
    from (t0, y0) across [t_min, t_max].
    f_scalar: function(t, y) -> float
    """
    # Choose step size
    h = (t_max - t_min) / n_steps

    # ---- forward from t0 to t_max ----
    if t0 < t_max:
        ts_f = np.arange(t0, t_max + 1e-12, h)
    else:
        ts_f = np.array([t0])
    ys_f = np.empty_like(ts_f)
    if len(ts_f) > 0:
        ys_f[0] = y0
        for i in range(len(ts_f) - 1):
            t, y = ts_f[i], ys_f[i]
            k1 = f_scalar(t, y)
            k2 = f_scalar(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f_scalar(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f_scalar(t + h, y + h * k3)
            ys_f[i + 1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # ---- backward from t0 to t_min ----
    if t0 > t_min:
        ts_b = np.arange(t0, t_min - 1e-12, -h)
    else:
        ts_b = np.array([t0])
    ys_b = np.empty_like(ts_b)
    if len(ts_b) > 0:
        ys_b[0] = y0
        for i in range(len(ts_b) - 1):
            t, y = ts_b[i], ys_b[i]
            # step backwards -> step size is -h
            hm = -h
            k1 = f_scalar(t, y)
            k2 = f_scalar(t + 0.5 * hm, y + 0.5 * hm * k1)
            k3 = f_scalar(t + 0.5 * hm, y + 0.5 * hm * k2)
            k4 = f_scalar(t + hm, y + hm * k3)
            ys_b[i + 1] = y + hm * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # Combine (backward reversed, then forward, without double-counting t0)
    ts_b = ts_b[::-1]
    ys_b = ys_b[::-1]
    if len(ts_f) > 0:
        ts = np.concatenate([ts_b[:-1], ts_f])
        ys = np.concatenate([ys_b[:-1], ys_f])
    else:
        ts, ys = ts_b, ys_b

    return ts, ys



# -----------------------------
# Drawing Circuit Diagrams
# -----------------------------

def draw_wire(ax, x0, y0, x1, y1):
    """
    Draw a wire segment from (x0, y0) to (x1, y1).
    
    Parameters
    ----------
    ax : matplotlib axis
    x0, y0, x1, y1 : float
    """
    ax.plot([x0, x1], [y0, y1], lw=2, color="black")


def draw_node(ax, x, y, label=None, position="up", offset = 0.15):
    """
    Draws a node at (x, y) and optionally labels it at a 
    specified position and offset.
    
    Parameters
    ----------
    ax : matplotlib axis
    x,y : float
        Position of the node.
    label : str or None
        Label for the current
    position : {"up", "down", "left", "right"}
        Positions label relative to the arrow.
    offset : float
        Distance between object and label.
    """
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

def draw_source(ax, x, y, label=None, orientation="horizontal"):
    """
    Draw a voltage source at (x, y) with optional label
    and orientation.
    
    Parameters
    ----------
    ax : matplotlib axis
    x,y : float
        position of the ground symbol
    label : str or None
        Label for the current
    orientation : {"horizontal, "vertical"}
        Draws the ground symbol horizontally or vertically
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


def draw_current_source(ax, x0, y0, x1, y1, label="I", position="right", offset=0.15):
    """
    Draw a wire with an arrow indicating current flow, from (x0, y0) to (x1, y1).
    Optionally adds label at a specified position and offset.

    Parameters
    ----------
    ax : matplotlib axis
    x0, y0, x1, y1 : float
        Endpoints of the wire: from (x0,y0) to (x1,y1)
    label : str or None
        Label for the arrow
    position : {"up", "down", "left", "right"}
        Positions label relative to the arrow
    offset : float
        Distance between object and label.
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
        linewidth=1,
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
        
def draw_resistor(ax, x0, y0, x1, y1, label=None, x_offset=0.2,y_offset=-0.1):
    """Draw a zig-zag-ish resistor from (x0,y0) to (x1,y1).

    Parameters
    ----------
    ax : matplotlib axis
    x0, y0, x1, y1 : float
        Endpoints of the wire: from (x0,y0) to (x1,y1)
    label : str or None
        Label for the arrow
    """
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
    elif abs(x1 - x0) < 1e-9:
        # vertical
        ys = np.linspace(y0, y1, 9)
        xs = np.full_like(ys, x0)
        amp = 0.15
        for i in range(2, 7):
            xs[i] = x0 + (amp if i % 2 == 0 else -amp)
        ax.plot(xs, ys, lw=2, color="black")
    else:
        # diagonal fallback (simple line)
        ax.plot([x0, x1], [y0, y1], lw=2, color="black")


    if not label:
        return
    ax.text(
        x0+x1/2 + x_offset,
        y0+y1/2 + y_offset,
        label,
        fontsize=12,
        ha='center',
        va='center',
    )
    
import numpy as np

def draw_inductor(ax, x0, y0, x1, y1, label=None, n_coils=6,x_offset=0.15, y_offset= 0.01):
    """
    Draw an inductor (coil) between (x0, y0) and (x1, y1).
    """
    dx = x1 - x0
    dy = y1 - y0

    # Determine orientation
    horizontal = abs(dx) >= abs(dy)

    if horizontal:
        sign = np.sign(dx) if dx != 0 else 1
        length = abs(dx)
        # Lead lengths
        lead = 0.15 * length
        coil_length = length - 2 * lead
        # Draw leads
        ax.plot([x0, x0 + sign * lead], [y0, y0], lw=2, color="black")
        ax.plot([x1 - sign * lead, x1], [y1, y1], lw=2, color="black")

        # Coil
        radius = coil_length / (2 * n_coils)
        theta = np.linspace(0, np.pi, 40)

        x_start = x0 + sign * lead
        for i in range(n_coils):
            xc = x_start + sign * (2 * i + 1) * radius
            xs = xc + radius * np.cos(theta) * sign
            ys = y0 + radius * np.sin(theta)
            ax.plot(xs, ys, lw=2, color="black")
    else:
        length = dy
        sign = np.sign(length) if length != 0 else 1
        length = abs(length)

        # Lead lengths
        lead = 0.15 * length
        coil_length = length - 2 * lead

        # Draw leads
        ax.plot([x0, x0], [y0, y0 + sign * lead], lw=2, color="black")
        ax.plot([x1, x1], [y1 - sign * lead, y1], lw=2, color="black")

        # Coil
        radius = coil_length / (2 * n_coils)
        theta = np.linspace(0, np.pi, 40)

        y_start = y0 + sign * lead
        for i in range(n_coils):
            yc = y_start + sign * (2 * i + 1) * radius
            xs = x0 + radius * np.sin(theta)
            ys = yc + radius * np.cos(theta) * sign
            ax.plot(xs, ys, lw=2, color="black")

    # Label
    if label:
        ax.text(x0+x1/2 + x_offset, (y0 + y1) / 2 + y_offset, label,
                ha="center", va="center", fontsize=11)

# -----------------------------
# Mass Spring Systems
# -----------------------------
def mass_spring_solution(t, m, b, k, x0=1.0, v0=0.0):
    """
    Return x(t) for the second-order linear ODE:
        m x'' + b x' + k x = 0
    with initial conditions x(0) = x0, x'(0) = v0.
    Handles underdamped, critically damped, and overdamped cases based on discriminant.
    """
    t = np.asarray(t, dtype=float)
    if m <= 0 or k <= 0:
        return np.zeros_like(t)

    omega0 = np.sqrt(k / m)
    gamma = b / (2.0 * m)  # damping ratio
    disc = gamma**2 - omega0**2

    # Underdamped: gamma < omega0
    if disc < -1e-12:
        omega_d = np.sqrt(omega0**2 - gamma**2)
        A = x0
        B = (v0 + gamma * x0) / omega_d
        x = np.exp(-gamma * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    # Critically damped: gamma == omega0
    elif abs(disc) <= 1e-12:
        A = x0
        B = v0 + gamma * x0
        x = (A + B * t) * np.exp(-gamma * t)

    # Overdamped: gamma > omega0
    else:
        sqrt_disc = np.sqrt(disc)
        r1 = -gamma + sqrt_disc
        r2 = -gamma - sqrt_disc
        denom = (r1 - r2)
        if abs(denom) < 1e-12:
            x = np.zeros_like(t)
        else:
            A = (v0 - r2 * x0) / denom
            B = x0 - A
            x = A * np.exp(r1 * t) + B * np.exp(r2 * t)

    return x

def draw_mass_spring_ax(ax, x_current, x_min, x_max):
    """
    Draw a simple vertical mass–spring cartoon on a Matplotlib Axes.
    y-axis corresponds to x; the mass moves with x_current.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(x_min, x_max)
    ax.axis("off")

    # Ceiling line at top of plot
    anchor_y = x_max
    ax.plot([0.2, 0.8], [anchor_y, anchor_y], color="black", linewidth=3)

    # Mass geometry
    box_height = 0.4 * (x_max - x_min) / 5.0
    box_height = min(box_height, (x_max - x_min) / 3.0)
    margin = box_height
    y_mass = float(np.clip(x_current, x_min + margin, x_max - margin))

    # Spring: zigzag from ceiling down to top of mass
    num_coils = 8
    y_top_spring = anchor_y - 0.2 * (x_max - x_min) / 5.0
    y_bottom_spring = y_mass + box_height / 2.0
    ys = np.linspace(y_top_spring, y_bottom_spring, 2 * num_coils + 1)
    xs = []
    for i in range(len(ys)):
        if i == 0 or i == len(ys) - 1:
            xs.append(0.5)
        else:
            xs.append(0.4 if i % 2 == 0 else 0.6)

    # Vertical rod
    ax.plot([0.5, 0.5], [anchor_y, y_top_spring], color="gray", linewidth=2)
    # Spring
    ax.plot(xs, ys, color="gray", linewidth=2)

    # Mass (rectangle)
    rect_width = 0.4
    rect = plt.Rectangle(
        (0.5 - rect_width / 2.0, y_mass - box_height / 2.0),
        rect_width,
        box_height,
        linewidth=2,
        edgecolor="black",
        facecolor="lightblue",
    )
    ax.add_patch(rect)

def draw_graph_ax(ax, t_vals, x_vals, t_max, x_min, x_max, t_current=None):
    """
    Draw x(t) vs t on a Matplotlib Axes, optionally tracing up to t_current.
    """
    ax.clear()
    ax.set_xlim(0, t_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("time t")
    ax.set_ylabel("position x(t)")
    ax.set_title("Mass–Spring System")

    # Full curve
    ax.plot(t_vals, x_vals, color="navy", linewidth=2, label="x(t)")

    if t_current is not None:
        mask = t_vals <= t_current
        if np.any(mask):
            ax.plot(
                t_vals[mask],
                x_vals[mask],
                color="orange",
                linewidth=3,
                label="current position",
            )
            x_cur = float(np.interp(t_current, t_vals, x_vals))
            ax.scatter([t_current], [x_cur], color="orange", s=40)

    ax.legend(loc="upper right")
