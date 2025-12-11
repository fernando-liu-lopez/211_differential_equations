import numpy as np
import streamlit as st
import sympy as sp
from bokeh.plotting import figure
from bokeh.models import Legend
from streamlit_bokeh import streamlit_bokeh

# ================================
#   ODE model and helpers
# ================================
def make_rhs(expr: str):
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


def rk4_solve(f_scalar, t0, y0, t_min, t_max, n_steps=1000):
    """
    Simple RK4 integrator that integrates both forward and backward
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


# ================================
#   Streamlit page
# ================================
st.set_page_config(page_title="Slope Field Plotter", layout="wide")

st.title("Slope Field Plotter")

st.markdown(r"""This is a simple  **direction field / slope field** plotter for a first-order ODE. Enter your differential equation below: """)
col_math, col_input = st.columns([3,20], vertical_alignment="center")
with col_math:
    st.latex(r"\frac{dy}{dt} = f(t, y) = ")
with col_input:
    expr = st.text_input(
        label="",
        value="- t + y**2 + cos(2*t)",
        placeholder="enter f(t, y)",
        help="**Syntax rules:**\n"
        "- Use `t` for the independent variable and `y` for the dependent variable.\n"
        "- Standard arithmetic operations: `+`, `-`, `*`, `/`.\n\n"
        "- Include `*` between multiplied terms (e.g. `2*t`).\n"
        "- Use `**` for exponentiation (e.g. `y**2`, `t**3`).\n"
        "- Use parentheses to group expressions (e.g. `exp(-(t**2))`.\n"
        "- **Available functions:**\n"
        "`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pi`.\n"
    )
st.markdown(r"""You entered:""")
t, y = sp.symbols("t y")
try:
    sympy_expr = sp.sympify(expr, locals={"t": t, "y": y})
    latex_expr = sp.latex(sympy_expr)
    st.latex(rf"\frac{{dy}}{{dt}} = f(t, y) = {latex_expr}.")
except Exception as e:
    st.warning("Could not interpret expression. Check your syntax.")

# -----------------------------
#   Controls
# -----------------------------
with st.form("slope_field_form"):
    col_curves, col_grid = st.columns([2, 2], gap="large")
    
    with col_curves:
        st.markdown("**Solution Curves:**")
        curve_1, curve_2 = st.columns(2)
        with curve_1:
            t0_1 = st.number_input("Curve 1: Initial t₀", value=0.0, step=1.0)
            y0_1 = st.number_input("Curve 1: Initial y₀", value=1.0, step=1.0)
            show_sol1 = st.checkbox("Show solution curve 1", value=True)
        with curve_2:
            t0_2 = st.number_input("Curve 2: Initial t₀", value=0.0, step=1.0)
            y0_2 = st.number_input("Curve 2: Initial y₀", value=-1.0, step=1.0)
            show_sol2 = st.checkbox("Show solution curve 2", value=False)
    with col_grid:
        st.markdown("**Graph Settings:**")
        grid_left, grid_right = st.columns(2)
        with grid_left:
            t_min = st.number_input("Minimum t", value=-5.0, step=1.0)
            t_max = st.number_input("Maximum t", value=5.0, step=1.0)
            n_t = st.slider(
            "Density of t-points",
            min_value=5,
            max_value=40,
            value=20,
            )
        with grid_right:
            y_min = st.number_input("Minimum y", value=-5.0, step=1.0)
            y_max = st.number_input("Maximum y", value=5.0, step=1.0)
            n_y = st.slider(
            "Density of y-points",
            min_value=5,
            max_value=40,
            value=20,
            )

        if t_max <= t_min:
            st.warning("Require t_max > t_min.")
        if y_max <= y_min:
            st.warning("Require y_max > y_min.")
        
    submitted = st.form_submit_button("Plot")

# Stop if bounds invalid
if t_max <= t_min or y_max <= y_min:
    st.stop()

# -----------------------------
#   Build RHS function
# -----------------------------
try:
    f = make_rhs(expr)
    # test evaluation at a scalar point to catch obvious errors early
    _test_val = float(f(0.0, 0.0))
except Exception as e:
    st.error(f"Error in expression for f(t, y): {e}")
    st.stop()

# -----------------------------
#   Compute slope field
# -----------------------------
T_vals = np.linspace(t_min, t_max, n_t)
Y_vals = np.linspace(y_min, y_max, n_y)
T, Y = np.meshgrid(T_vals, Y_vals)

try:
    M = f(T, Y)  # slopes
except Exception as e:
    st.error(f"Error evaluating f(t, y) on the grid: {e}")
    st.stop()

# Avoid NaNs/infs
M = np.where(np.isfinite(M), M, 0.0)

# Direction vectors are proportional to (1, M)
# We'll normalize so arrows have a consistent length relative to the grid.
dt_grid = (t_max - t_min) / n_t
dy_grid = (y_max - y_min) / n_y
base_len = 0.45 * min(dt_grid, dy_grid)  # fraction of grid spacing

norm = np.sqrt(1.0 + M**2)
L = np.where(norm > 0, base_len / norm, 0.0)

dt = L
dy = M * L

x0 = T - dt / 2
x1 = T + dt / 2
y0 = Y - dy / 2
y1 = Y + dy / 2

# Flatten for plotting
x0f = x0.ravel()
x1f = x1.ravel()
y0f = y0.ravel()
y1f = y1.ravel()

# -----------------------------
#   Bokeh figure (fixed axes, square)
# -----------------------------
p = figure(
    width=900,
    height=650,
    x_axis_label="t",
    y_axis_label="y",
    x_range=(t_min, t_max),
    y_range=(y_min, y_max)
)
p.sizing_mode = "fixed"

# Plot slope field
p.segment(
    x0f, y0f, x1f, y1f,
    line_width=1,
    line_color="gray",
)

legend_items = []

# -----------------------------
#   Solution curves
# -----------------------------
def f_scalar(t, y):
    try:
        val = f(np.array(t), np.array(y))

        # If val is an array, get a scalar
        if isinstance(val, np.ndarray):
            val = val.item()

        # If it's complex or nan/inf, fall back to 0
        if not np.isfinite(val):
            return 0.0
        if isinstance(val, complex):
            return float(val.real)

        return float(val)
    except Exception:
        # If evaluating the RHS fails for some (t, y), just return 0 to keep the solver alive.
        return 0.0

if show_sol1:
    try:
        ts1, ys1 = rk4_solve(f_scalar, t0_1, y0_1, t_min, t_max, n_steps=800)
        curve1 = p.line(ts1, ys1, line_width=3, line_color="blue")
        legend_items.append(("Solution 1", [curve1]))
        p.scatter(
        [t0_1], [y0_1],
        marker="circle",
        size=8,
        color="blue")
        legend_items.append(("Initial condition 1", [p.scatter(
            [t0_1], [y0_1],
            marker="circle",
            size=8,
            color="yellow",
        )]))
    except Exception as e:
    st.warning(f"Could not compute solution curve 1: {e}")


if show_sol2:
    try:
        ts2, ys2 = rk4_solve(f_scalar, t0_2, y0_2, t_min, t_max, n_steps=800)
        curve2 = p.line(ts2, ys2, line_width=3, line_color="red", line_dash="dashed")
        legend_items.append(("Solution 2", [curve2]))
        p.scatter(
        [t0_2], [y0_2],
        marker="circle",
        size=8,
        color="red")
        legend_items.append(("Initial condition 2", [p.scatter(
            [t0_2], [y0_2],
            marker="circle",
            size=8,
            color="green",
        )]))
    except Exception as e:
        st.warning(f"Could not compute solution curve 1: {e}")

if legend_items:
    legend = Legend(items=legend_items)
    p.add_layout(legend, "right")
    p.legend.click_policy = "hide"

# -----------------------------
#   Render in Streamlit
# -----------------------------
streamlit_bokeh(p, use_container_width=False, theme="streamlit", key="slope_field_plot")
