import time

import numpy as np
import sympy as sp
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
#   Analytic solution for mx'' + b x' + k x = 0
#   with x(0) = x0, x'(0) = v0
# -----------------------------
def mass_spring_solution(t, m, b, k, x0=1.0, v0=0.0):
    """
    Return x(t) for the second-order linear ODE:
        m x'' + b x' + k x = 0
    with initial conditions x(0) = x0, x'(0) = v0.
    Handles underdamped, critically damped, and overdamped cases.
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


# -----------------------------
#   Drawing helpers (Matplotlib)
# -----------------------------
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


# -----------------------------
#   Streamlit Page
# -----------------------------
st.set_page_config(page_title="Mass–Spring System", layout="wide")

st.title("Mass–Spring System")

st.markdown(r"""We model the motion of a mass on a vertical spring with damping by:""")
st.latex(r"""m x''(t) + b x'(t) + k x(t) = 0,\hspace{0.2in}\textnormal{where}\quad m: \text{mass} \quad\quad b: \text{damping coefficient} \quad\quad k: \text{spring constant}.""")

st.divider()


col_controls, col_animation = st.columns([1, 4])

with col_controls:
    m = st.number_input("Mass \(m\)", min_value=0.1, value=5.0, step=0.1, format="%.2f")
    b = st.number_input("Damping \(b\)", min_value=0.0, value=2.0, step=0.1, format="%.2f")
    k = st.number_input("Spring constant \(k\)", min_value=0.1, value=4.0, step=0.1, format="%.2f")
    x0 = st.number_input("Initial position", value=1.0, step=0.1, format="%.2f")
    v0 = st.number_input("Initial velocity", value=0.0, step=0.1, format="%.2f")
    t_max = st.number_input("Maximum time", min_value=1.0, value=20.0, step=1.0, format="%.1f")

    animate = st.button("Visualize", type="primary")



# Time grid and solution
t_vals = np.linspace(0.0, float(t_max), 800)
x_vals = mass_spring_solution(t_vals, m, b, k, x0=x0, v0=v0)

# Vertical range for plots
x_min = float(np.min(x_vals)) - 0.2 * (np.max(x_vals) - np.min(x_vals) + 1e-6)
x_max = float(np.max(x_vals)) + 0.2 * (np.max(x_vals) - np.min(x_vals) + 1e-6)
if x_min == x_max:
    x_min -= 1.0
    x_max += 1.0

with col_animation:
    # -----------------------------------------
    # Render the ODE and initial conditions
    # -----------------------------------------
    t = sp.symbols('t')
    x = sp.Function('x')

    # Build the symbolic ODE:
    ode_expr = m * x(t).diff(t, 2)
    ode_expr += b * x(t).diff(t, 1)
    ode_expr += k * x(t)
    # Render:
    st.latex(sp.latex(ode_expr)
        + r" = 0"
    )
    st.latex("x(0)=" + sp.latex(x0) + r", \quad x'(0)=" + sp.latex(v0))
    # Layout for mass-spring (left) and graph (right)
    col_left, col_right = st.columns([1, 2], gap="large")
    left_placeholder = col_left.empty()
    right_placeholder = col_right.empty()


def draw_frame(t_current: float):
    """
    Draw a single frame at time t_current into the two placeholders.
    """
    x_current = float(np.interp(t_current, t_vals, x_vals))

    # --- Left: mass–spring cartoon ---
    fig1, ax1 = plt.subplots(figsize=(3, 4))
    draw_mass_spring_ax(ax1, x_current, x_min, x_max)
    left_placeholder.pyplot(fig1)
    plt.close(fig1)

    # --- Right: x(t) graph with trace ---
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    draw_graph_ax(ax2, t_vals, x_vals, t_max, x_min, x_max, t_current=t_current)
    right_placeholder.pyplot(fig2)
    plt.close(fig2)


# Show either static or animated view
if not animate:
    draw_frame(0.0)
else:
    n_frames = 120
    for t_cur in np.linspace(0.0, float(t_max), n_frames):
        draw_frame(t_cur)
        time.sleep(0.01)
