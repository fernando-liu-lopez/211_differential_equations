import numpy as np
import streamlit as st
import sympy as sp
from bokeh.plotting import figure
from bokeh.models import Legend
from streamlit_bokeh import streamlit_bokeh


# -----------------------------
#   Model: Newton's law of cooling
# -----------------------------
def newton_cooling_solution(t, T0, T_env, k):
    """
    Solution of Newton's law of cooling:
        dT/dt = -k (T - T_env),  T(0) = T0
    """
    t = np.asarray(t)
    return T_env + (T0 - T_env) * np.exp(-k * t)


# -----------------------------
#   Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Newton's Law of Cooling",
    layout="centered",
)

st.title("Newton's Law of Cooling")

# sympy expressions
t, T = sp.symbols("t T")

st.markdown("""Start with an object whose temperature at time t is given by:""")
st.latex(r"""T(t),.""")
st.markdown("""Assume the object is in an environment that remains at a constant temperature:""")
st.latex(r"""T_{\text{env}}.""")
st.markdown("""**Newton's law of cooling** states that the rate of change of the temperature of the object:""")
st.latex(r"""\textnormal{i.e.}\qquad\frac{dT}{dt}""")
st.markdown(r"""is proportional to...""")
st.latex(r"""\textnormal{i.e.}\qquad\frac{dT}{dt} = -k\times\square \hspace{0.2in}\textnormal{for some positive constant }k>0,""")
st.markdown(r"""the difference between the object's temperature and the ambient temperature.""")
st.latex(r"""\textnormal{i.e.}\qquad\frac{dT}{dt} = -k\bigl(T(t) - T_{\text{env}}\bigr).""")
st.markdown(r"""This is a first-order linear ODE! The constant k is called the **cooling constant** and depends on the characteristics of the object.""")

st.markdown(r"""The grapher below will plot the temperature of two objects over time.""")
st.markdown(r"""You'll need to give it an initial condition (i.e. an initial temperature for the object):""")
st.latex(r"""T(0) = T_0,""")
st.markdown(r"""as well as a value for the cooling constant.""")
# -----------------------------
#   Global ranges (for sliders + fixed axes)
# -----------------------------
with st.form("newton_form"):

    st.markdown("**Parameters:**")
    T_MAX = st.number_input("Graph until time t =", value=20.0, step=1.0)
    Tenv = st.number_input("Ambient Temperature:", value=67.0, step=1.0)
    obj1, obj2 = st.columns(2)
    with obj1:
        st.markdown("**Object 1:**")
        T0_1 = st.number_input("Initial Temperature 1", value=80.0, step=1.0)
        k_1 = st.number_input("Cooling Constant 1 (k)", value=0.2, step=0.1)
        show_curve_1 = st.checkbox("Show Object 1", value=True)
    with obj2:
        st.markdown("**Object 2:**")
        T0_2 = st.number_input("Initial Temperature 2", value=120.0, step=1.0)
        k_2 = st.number_input("Cooling Constant 2 (k)", value=0.7, step=0.01)
        show_curve_2 = st.checkbox("Show Object 2", value=True)        
    
    submitted = st.form_submit_button("Plot")

# Fixed y-axis range based on slider ranges
Y_MIN = min(T0_1,T0_2, Tenv)
Y_MAX = max(T0_1,T0_2, Tenv)

# Time grid (fixed for all curves)
t = np.linspace(0.0, T_MAX, 600)


# -----------------------------
#   Compute curves
# -----------------------------
T1 = newton_cooling_solution(t, T0_1, Tenv, k_1)
T2 = newton_cooling_solution(t, T0_2, Tenv, k_2)

# -----------------------------
#   Bokeh figure (square, fixed axes)
# -----------------------------
p = figure(
    width=700,
    height=500,
    x_axis_label="Time t",
    y_axis_label="Temperature",
    x_range=(0, T_MAX),
    y_range=(Y_MIN, Y_MAX)
)

# Keep plot shape square-ish
p.sizing_mode = "fixed"

# Curve 1 + ambient line
env = p.line(
    [0, T_MAX],
    [Tenv, Tenv],
    line_width=2,
    line_dash="dashed",
    line_color="yellow",
    level='overlay'
)

legend_items = [("Ambient Temp", [env])]

if show_curve_1:
    curve1 = p.line(
        t,
        T1,
        line_width=3,
        line_color="blue",
    )
    legend_items.extend([("Object 1 Temp", [curve1])])

# Curve 2 + ambient line
if show_curve_2:
    curve2 = p.line(
        t,
        T2,
        line_width=3,
        line_color="red",
    )
    legend_items.extend([("Object 2 Temp", [curve2])])

legend = Legend(items=legend_items, location="center")
p.add_layout(legend, "right")
p.legend.click_policy = "hide"

# -----------------------------
#   Render Bokeh in Streamlit
# -----------------------------
streamlit_bokeh(p, use_container_width=False, theme="streamlit", key="newton_plot")

