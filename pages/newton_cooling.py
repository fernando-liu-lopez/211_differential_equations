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

st.markdown("""**Newton's law of cooling** describes the rate at which an object cools down (or heats up) when exposed to a different temperature environment. The law states that the rate of change of the temperature of an object is proportional to the difference between its own temperature and the temperature of its surroundings.""")
st.markdown("""Here, we will (1) breakdown how to write down an ODE that models Newton's law, and (2) visualize the solutions to the ODE in different scenarios, and (3) explore different scenarios where we can apply Newton's law.""")
st.divider()

# sympy expressions
t, T = sp.symbols("t T")

st.subheader("1. Modeling Newton's Law:")
st.markdown("""Start with an object whose temperature at time t is given by:""")
st.latex(r"""T(t),.""")
st.markdown("""Assume the object is in an environment that remains at a constant temperature:""")
st.latex(r"""T_{\text{env}}\in\mathbb{R}.""")
st.markdown("""**Newton's law** states that the rate of change of the temperature of the object:""")
st.latex(r"""\textnormal{i.e.}\qquad\frac{dT}{dt}""")
st.markdown("""is proportional to:""")
st.latex(r"""\textnormal{i.e.}\qquad\frac{dT}{dt} = \pm k\cdot\square \hspace{0.2in}\textnormal{for some positive constant }k>0,""")
st.markdown("""the difference between the object's temperature and the ambient temperature:""")
st.latex(r"""\textnormal{i.e.}\qquad\frac{dT}{dt} = \pm k\cdot \bigl(T(t) - T_{\text{env}}\bigr).""")
st.markdown(r"""If the object is **cooling**, we assume the object's temperature is higher than the ambient temperature:""")
st.latex(r"""\textnormal{i.e.}\qquad\bigl(T(t) - T_{\text{env}}\bigr) > 0,""")
st.markdown(r"""and therefore we choose the minus sign in front of k to make the rate of change negative.""")
st.latex(r"""\frac{dT}{dt} = - k\cdot \bigl(T(t) - T_{\text{env}}\bigr).""")
st.markdown("""This is a first-order linear ODE!""")
st.markdown("""The constant k is called the **cooling constant** and depends on the characteristics of the object.""")
st.divider()

st.subheader("2. Visualizing solutions:")
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
        k_1 = st.number_input("Cooling Constant 1 (k)", value=0.2, step=0.1, format="%0.4f")
        show_curve_1 = st.checkbox("Show Object 1", value=True)
    with obj2:
        st.markdown("**Object 2:**")
        T0_2 = st.number_input("Initial Temperature 2", value=100.0, step=1.0)
        k_2 = st.number_input("Cooling Constant 2 (k)", value=0.7, step=0.01, format="%0.4f")
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

st.divider()

st.subheader("3. Exploration and Discussion Questions:")

st.markdown("""1. In words, how would you describe the behavior of the solutions to Newton's law of cooling?""")
st.markdown("""2. How would you expect solutions to look if the ambient temperature is higher than the initial temperature of the object? Check your intuition by changing the ambient temperature in the grapher above.""")
st.markdown("""3. How does increasing/decreasing the cooling constant affect the rate at which the temperature changes? What sort of objects might have a high/low cooling constants?""")
st.markdown("""4. Your house thermostat is set to 70 degrees Fahrenheit. You have soup cooking at 170 degrees Fahrenheit for a dinner party. You put some of the soup in a bowl to cool for 5 minutes and notice a temperature drop of around 30 degrees. Using the grapher above, approximate the cooling constant for the soup you'll serve. If the soup is best enjoyed between 120 and 140 degrees Fahrenheit, how long should you let the soup cool before serving? How long do your guests have to eat before the soup is too cold?""")