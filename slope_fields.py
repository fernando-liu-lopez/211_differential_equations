# pages/slope_fields.py

import numpy as np
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import Legend
from streamlit_bokeh import streamlit_bokeh

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Slope Fields", layout="centered")

st.title("Slope Field Plotter")
st.markdown(
    r"""
Visualize direction fields for first-order ODEs of the form

\[
\frac{dy}{dt} = f(t,y)
\]

and overlay numerical solution curves from chosen initial conditions.
"""
)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def make_function(expr: str):
    """
    Build a vectorized function f(t,y) from a user-entered expression.
    """
    allowed = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
    }

    def f(t, y):
        local = dict(allowed)
        local["t"] = t
        local["y"] = y
        return eval(expr, {"__builtins__": {}}, local)

    return np.vectorize(f)


def rk4_solve(f, t0, y0, t_min, t_max, n_steps=800):
    ts = np.linspace(t_min, t_max, n_steps)
    ys = np.zeros_like(ts)
    ys[0] = y0
    h = ts[1] - ts[0]

    for i in range(len(ts) - 1):
        t = ts[i]
        y = ys[i]
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        ys[i + 1] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ts, ys


# ------------------------------------------------------------
# Form (inputs)
# ------------------------------------------------------------
with st.form("slope_field_form"):
    expr = st.text_input(
        r"Enter $f(t,y)$ for $\frac{dy}{dt} = f(t,y)$",
        value="y - t",
        help=(
            "Use Python syntax. Examples:\n"
            "- y - t\n"
            "- y*(1 - y)\n"
            "- sin(t) - y\n"
            "Use ** for powers, e.g. y**2."
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        t_min = st.number_input("t min", value=0.0)
        t_max = st.number_input("t max", value=5.0)
        n_t = st.number_input("t grid points", min_value=5, value=25, step=5)
    with col2:
        y_min = st.number_input("y min", value=-5.0)
        y_max = st.number_input("y max", value=5.0)
        n_y = st.number_input("y grid points", min_value=5, value=25, step=5)

    st.markdown("### Solution curves")
    show_sol1 = st.checkbox("Show solution 1", value=True)
    show_sol2 = st.checkbox("Show solution 2", value=False)

    col3, col4 = st.columns(2)
    with col3:
        t0_1 = st.number_input("t₀ (curve 1)", value=0.0)
        y0_1 = st.number_input("y₀ (curve 1)", value=1.0)
    with col4:
        t0_2 = st.number_input("t₀ (curve 2)", value=0.0)
        y0_2 = st.number_input("y₀ (curve 2)", value=-1.0)

    submitted = st.form_submit_button("Plot")

# ------------------------------------------------------------
# Stable plot container (IMPORTANT)
# ------------------------------------------------------------
plot_slot = st.container()

# ------------------------------------------------------------
# Always mount a Bokeh component (blank if needed)
# ------------------------------------------------------------
invalid_bounds = (t_max <= t_min) or (y_max <= y_min)

if invalid_bounds or not submitted:
    p_blank = figure(
        width=900,
        height=650,
        x_axis_label="t",
        y_axis_label="y",
    )
    p_blank.sizing_mode = "fixed"
    p_blank.text([], [], text=[])

    with plot_slot:
        streamlit_bokeh(
            p_blank,
            use_container_width=False,
            theme="streamlit",
            key="bokeh:slope_fields:main",
        )

    if invalid_bounds:
        st.warning("Require t_max > t_min and y_max > y_min.")
    if not submitted:
        st.info("Adjust parameters, then press **Plot**.")

    st.stop()

# ------------------------------------------------------------
# Build f(t,y)
# ------------------------------------------------------------
try:
    f = make_function(expr)
    _ = float(f(0.0, 0.0))
except Exception as e:
    st.error(f"Error in f(t,y): {e}")
    st.stop()

# ------------------------------------------------------------
# Compute slope field
# ------------------------------------------------------------
T_vals = np.linspace(t_min, t_max, int(n_t))
Y_vals = np.linspace(y_min, y_max, int(n_y))
T, Y = np.meshgrid(T_vals, Y_vals)

try:
    M = f(T, Y)
except Exception as e:
    st.error(f"Error evaluating f(t,y) on grid: {e}")
    st.stop()

M = np.where(np.isfinite(M), M, 0.0)

dt_grid = (t_max - t_min) / n_t
dy_grid = (y_max - y_min) / n_y
base_len = 0.45 * min(dt_grid, dy_grid)

norm = np.sqrt(1 + M**2)
L = np.where(norm > 0, base_len / norm, 0.0)

dt = L
dy = M * L

x0 = (T - dt / 2).ravel()
x1 = (T + dt / 2).ravel()
y0s = (Y - dy / 2).ravel()
y1s = (Y + dy / 2).ravel()

# ------------------------------------------------------------
# Bokeh figure
# ------------------------------------------------------------
p = figure(
    width=900,
    height=650,
    x_axis_label="t",
    y_axis_label="y",
    x_range=(t_min, t_max),
    y_range=(y_min, y_max),
)
p.sizing_mode = "fixed"

p.segment(x0, y0s, x1, y1s, line_width=1, line_color="gray")

legend_items = []

# Safe scalar wrapper for RK4
def f_scalar(t, y):
    try:
        val = f(np.array(t), np.array(y))
        if isinstance(val, np.ndarray):
            val = val.item()
        if isinstance(val, complex):
            val = val.real
        if not np.isfinite(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


if show_sol1:
    try:
        ts1, ys1 = rk4_solve(f_scalar, t0_1, y0_1, t_min, t_max)
        curve1 = p.line(ts1, ys1, line_width=3, line_color="blue")
        ic1 = p.scatter([t0_1], [y0_1], size=8, color="blue")
        legend_items.append(("Solution 1", [curve1]))
        legend_items.append(("IC 1", [ic1]))
    except Exception as e:
        st.warning(f"Could not compute solution curve 1: {e}")

if show_sol2:
    try:
        ts2, ys2 = rk4_solve(f_scalar, t0_2, y0_2, t_min, t_max)
        curve2 = p.line(ts2, ys2, line_width=3, line_color="red", line_dash="dashed")
        ic2 = p.scatter([t0_2], [y0_2], size=8, color="red")
        legend_items.append(("Solution 2", [curve2]))
        legend_items.append(("IC 2", [ic2]))
    except Exception as e:
        st.warning(f"Could not compute solution curve 2: {e}")

if legend_items:
    legend = Legend(items=legend_items)
    p.add_layout(legend, "right")
    p.legend.click_policy = "hide"

# ------------------------------------------------------------
# Render (same key, same container, always mounted)
# ------------------------------------------------------------
with plot_slot:
    streamlit_bokeh(
        p,
        use_container_width=False,
        theme="streamlit",
        key="bokeh:slope_fields:main",
    )
