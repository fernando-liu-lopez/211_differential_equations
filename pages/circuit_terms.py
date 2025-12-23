import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from helpers import centered_pyplot, draw_wire, draw_node, draw_resistor, draw_source, draw_inductor

st.set_page_config(page_title="Basic Circuit Terminology", layout="wide")

st.header("Terminology and Diagrammatic Representation")

col1,col2 = st.columns(2)
with col1:
    st.subheader("Conductor")
    st.write("A material that allows the flow of electric current through it (e.g. metal wires). Think of conductors as pipes carrying water (in this case, electric charge).")
    st.write("We draw these as solid lines/wires.")
    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis("off")

    draw_wire(ax, 0, 0, 2, 0)

    centered_pyplot(fig)
    plt.close(fig)
    
    #----------------------------

with col2:
    st.subheader("Junctions (J)")

    st.write("A point where two or more conductors meet. Think of these as intersections of multiple pipes carrying water.")
    st.write("We draw these as nodes.")

    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis("off")

    draw_wire(ax, 0, 0, 1, 0)
    draw_wire(ax, 1, 0, 2, 1)
    draw_wire(ax, 1, 0, 2, -1)
    draw_node(ax, 1, 0, label=r"$J$", offset=0.25)

    centered_pyplot(fig)
    plt.close(fig)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Current (I)")

    st.write("""The rate of flow of electric charge through a conductor. 
    Think of current as \'how fast water is flowing through the pipes\'.
    We draw these as arrows along the wires, pointing in the direction of current flow.
    """)

    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis("off")
    
    draw_wire(ax, 0, 0, 2, 0)
    arrow = FancyArrowPatch(
        (0,0), (1.3,0),
        arrowstyle='-|>',
        mutation_scale=20,   # ↓ makes arrowhead smaller/thinner
        linewidth=1.5,       # shaft thickness
        color='black',
    )
    ax.add_patch(arrow)
    ax.text(1, 0.04, r'$I$', fontsize=11, ha='center', va='top')
    centered_pyplot(fig)
    plt.close(fig)
    st.write("""***Kirkhoff's current law*** states that currents are conserved at junctions, 
    i.e. the sum of currents entering a junction must equal the sum of currents leaving it.
    """)
with col2:
    st.subheader("Voltage (V or E)")

    st.write("""The electrical potential difference between two points in a circuit. 
    If you think of two tanks connected by a pipe (one filled, one half-empty), then voltage is the 'water pressure' that drives current flow.
    A ***voltage source*** is a device that creates and maintains a voltage (e.g. batteries or generators). 
    We draw voltage sources as short and long parallel lines with a gap between them.
    As in the water tank analogy, we flow from the longer line (fuller tank) to the shorter tank (emptier tank).
    """)

    fig, ax = plt.subplots(figsize=(2,1))
    ax.axis("off")

    draw_wire(ax, 0, 0, 1, 0)
    draw_wire(ax, 0, 0, 0, 1)
    draw_wire(ax, 0, 1, 2, 1)
    draw_wire(ax, 2, 1, 2, 0)
    draw_source(ax, 1, 0,orientation='vertical')
    draw_wire(ax, 1.15, 0, 2, 0)

    centered_pyplot(fig)
    plt.close(fig)

st.divider()

col1,col2 = st.columns(2)
with col1:
    st.subheader("Resistors (R)")
    
    st.write("""A component that reduces the voltage/flow of electric current. 
    Think of these as thinner pipes that allow us to slow down the flow of water (electric current).
    The 
    We draw these as zig-zag lines along the wires.""")

    fig, ax = plt.subplots(figsize=(1,1))
    ax.axis("off")
    draw_wire(ax, 0,0, 0.25,0)
    draw_resistor(ax, 0.25,0, 0.75, 0, label=r"$R$")
    draw_wire(ax, 0.75,0, 1,0)
    centered_pyplot(fig)
    plt.close(fig)
    st.write("""***Ohm's law*** states that the voltage across a resistor is proportional to the current passing through it.
    Mathematically, this is expressed as:""")
    st.latex("V = I \cdot R,")
    st.write("where R is a constant (called the ***resistance***) that depends on the material and dimensions of the resistor.")
    
with col2:  
    st.subheader("Inductors (L)")
    
    st.write("""A component (usually insulated wire wound in a coil) that stores energy when a current flows through it.""")

    fig, ax = plt.subplots(figsize=(2,1))
    ax.axis("off")
    draw_wire(ax, 0,0, 0.25,0)
    draw_inductor(ax, 0.25,0, 0.75, 0, label=r"$L$",n_coils=3)
    draw_wire(ax, 0.75,0, 1,0)
    centered_pyplot(fig)
    plt.close(fig)
    st.write("""***Faraday's law*** and ***Lenz's law*** combined show that the voltage across an inductor is proportional to the rate of change of the current passing through it.
    Mathematically, this is expressed as:""")
    st.latex("V = L \cdot \\frac{dI}{dt},")
    st.write("where L is a constant (called the ***inductance***).")

st.divider()

    
    
