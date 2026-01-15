# pages/slope_fields.py

import numpy as np
import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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

# ------------------------------------------------------------
# SymPy parsing (also used for plotting)
# ------------------------------------------------------------
t, y = sp.symbols("t y")

_sympy_locals = {
    "t": t,
    "y": y,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pi": sp.pi,
}

sympy_expr = None
f_np = None

try:
    sympy_expr = sp.sympify(expr, locals=_sympy_locals)
    latex_expr = sp.latex(sympy_expr)
    st.latex(rf"\frac{{dy}}{{dt}} = f(t, y) = {latex_expr}.")
    # Build a numpy-evaluable function f(t,y)
    # modules=["numpy"] makes sin/cos/exp/etc work on arrays
    f_np = sp.lambdify((t, y), sympy_expr, modules=["numpy"])
except Exception:
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
            show_sol2 = st.checkbox("Show solution curve 2", value=True)

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

# Require a valid expression
if sympy_expr is None or f_np is None:
    st.info("Enter a valid function for f(t,y), then press **Plot**.")
    st.stop()

if not submitted:
    st.info("Enter a function, then press **Plot**.")
    st.stop()


# ------------------------------------------------------------
# Numerical helpers (RK4)
# ------------------------------------------------------------
def f_scalar(tt, yy):
    """
    Safe scalar evaluation of f_np(tt,yy) returning a real float.
    Any error/non-finite returns 0.0 to avoid crashing solvers.
    """
    try:
        val = f_np(tt, yy)

        # Convert numpy scalars/arrays
        if isinstance(val, np.ndarray):
            val = val.item()

        # Handle complex outputs
        if isinstance(val, complex):
            val = val.real

        # Guard against nan/inf
        if not np.isfinite(val):
            return 0.0

        return float(val)
    except Exception:
        return 0.0


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    y_next = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y_next


def rk4_from_ic(f, t0, y0, t_min, t_max, n_steps=900):
    """
    Solve y' = f(t,y) with y(t0)=y0 over [t_min, t_max] by integrating
    forward and backward from the initial condition and concatenating.
    """
    # If t0 is outside the plotting window, just integrate across window with that IC
    # (still anchored at t0; the resulting curve may not be meaningful within window)
    t0 = float(t0)
    y0 = float(y0)

    # Forward integration: t0 -> t_max
    if t_max > t0:
        ts_f = np.linspace(t0, t_max, n_steps)
        ys_f = np.empty_like(ts_f, dtype=float)
        ys_f[0] = y0
        hf = ts_f[1] - ts_f[0]
        for i in range(len(ts_f) - 1):
            y_next = rk4_step(f, ts_f[i], ys_f[i], hf)
            if not np.isfinite(y_next):
                ts_f = ts_f[: i + 1]
                ys_f = ys_f[: i + 1]
                break
            ys_f[i + 1] = y_next
    else:
        ts_f = np.array([t0], dtype=float)
        ys_f = np.array([y0], dtype=float)

    # Backward integration: t0 -> t_min
    if t_min < t0:
        ts_b = np.linspace(t0, t_min, n_steps)  # decreasing
        ys_b = np.empty_like(ts_b, dtype=float)
        ys_b[0] = y0
        hb = ts_b[1] - ts_b[0]  # negative
        for i in range(len(ts_b) - 1):
            y_next = rk4_step(f, ts_b[i], ys_b[i], hb)
            if not np.isfinite(y_next):
                ts_b = ts_b[: i + 1]
                ys_b = ys_b[: i + 1]
                break
            ys_b[i + 1] = y_next
        # Reverse backward arrays so they go left-to-right in time
        ts_b = ts_b[::-1]
        ys_b = ys_b[::-1]
    else:
        ts_b = np.array([t0], dtype=float)
        ys_b = np.array([y0], dtype=float)

    # Concatenate, avoiding duplicate t0
    if len(ts_b) > 0 and len(ts_f) > 0:
        ts = np.concatenate([ts_b[:-1], ts_f])
        ys = np.concatenate([ys_b[:-1], ys_f])
    else:
        ts, ys = ts_f, ys_f

    return ts, ys



# ------------------------------------------------------------
# Compute slope field segments
# ------------------------------------------------------------
T_vals = np.linspace(t_min, t_max, n_t)
Y_vals = np.linspace(y_min, y_max, n_y)
T, Y = np.meshgrid(T_vals, Y_vals)

try:
    M = f_np(T, Y)  # slopes
except Exception as e:
    st.error(f"Error evaluating f(t,y) on the grid: {e}")
    st.stop()

# sanitize slopes
M = np.where(np.isfinite(M), M, 0.0)

# segment length relative to grid spacing
dt_grid = (t_max - t_min) / max(n_t, 1)
dy_grid = (y_max - y_min) / max(n_y, 1)
base_len = 0.45 * min(dt_grid, dy_grid)

# normalize direction vectors (1, M)
norm = np.sqrt(1.0 + M**2)
L = np.where(norm > 0, base_len / norm, 0.0)

dt = L
dy = M * L

x0 = (T - dt/2).ravel()
x1 = (T + dt/2).ravel()
y0 = (Y - dy/2).ravel()
y1 = (Y + dy/2).ravel()

segments = np.stack(
    [np.stack([x0, y0], axis=1), np.stack([x1, y1], axis=1)],
    axis=1
)


# ------------------------------------------------------------
# Plot with Matplotlib
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.5))

lc = LineCollection(segments, linewidths=1.0, alpha=0.55)
ax.add_collection(lc)

# fixed axes
ax.set_xlim(t_min, t_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel("t")
ax.set_ylabel("y")

# square plot area
ax.set_box_aspect(1)

# overlay solution curves
if show_sol1:
    try:
        ts1, ys1 = ts1, ys1 = rk4_from_ic(f_scalar, t0_1, y0_1, t_min, t_max, n_steps=900)
        ax.plot(ts1, ys1, linewidth=2.5, label="Solution 1")
        ax.scatter([t0_1], [y0_1], s=35, label="Initial condition 1")
    except Exception as e:
        st.warning(f"Could not compute solution curve 1: {e}")

if show_sol2:
    try:
        ts2, ys2 = ts2, ys2 = rk4_from_ic(f_scalar, t0_2, y0_2, t_min, t_max, n_steps=900)
        ax.plot(ts2, ys2, linewidth=2.5, linestyle="--", label="Solution 2")
        ax.scatter([t0_2], [y0_2], s=35, label="Initial condition 2")
    except Exception as e:
        st.warning(f"Could not compute solution curve 2: {e}")

# title (optional: keep it minimal)
# If you prefer no title, delete the next two lines.
ax.set_title(rf"Slope Fields for: $\frac{{dy}}{{dt}} = {sp.latex(sympy_expr)}$", fontsize=12)

ax.grid(True, alpha=0.25)

handles, labels = ax.get_legend_handles_labels()
if labels:
    ax.legend(loc="upper right")

# Center the figure on the page
left, center, right = st.columns([1, 2, 1])
with center:
    st.pyplot(fig)

plt.close(fig)
