import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kirchhoff\'s Laws", layout="centered")

from helpers import centered_pyplot, draw_current_source, draw_source, draw_node, draw_wire


st.title("Kirchhoff’s Circuit Laws")

st.write("This page will cover Kirchhoff’s Current Law and Kirchhoff’s Voltage Law. For a crash course on basic circuit terminology, see the page below:")
st.page_link("pages/circuit_terms.py", label="Basic Circuit Terms")

st.header("1. Kirchhoff’s Current Law")
st.markdown(
    """Kirchhoff’s Current Law states that at any junction in an electrical circuit, the sum of currents leaving the node equals the sum of currents entering. """)
st.markdown("""It is standard to label currents with the letter 'I', in which case Kirchhoff's Current Law states:""")

st.latex(r"\sum I_{in} = \sum I_{out}")
st.markdown("""For example, the junction drawn below satisfies:""")
st.latex(r"I_1 = I_2 + I_3.")
# draw circuit 0
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

draw_source(ax1, 2, 0, orientation="vertical")
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
st.markdown("""3. Suppose you had a device that allowed you to measure the current flowing through the wires at any point in the circuit. What's the minimum number of points you would you need to measure to determine all of the currents in this circuit? How does that number relate to the RREF of the augmented matrix?""")
