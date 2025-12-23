import time

import numpy as np
import sympy as sp
import streamlit as st
import matplotlib.pyplot as plt

from helpers import mass_spring_solution, draw_mass_spring_ax, draw_graph_ax

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
    st.latex(f"x(0)={sp.latex(x0)}" + r", \quad x'(0)=" + sp.latex(v0))
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
