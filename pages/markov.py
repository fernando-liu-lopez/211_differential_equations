import streamlit as st
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, patches
from matplotlib.patches import FancyArrowPatch

from helpers import centered_pyplot, draw_current_source, draw_node, draw_wire

st.set_page_config(page_title="Markov Chains", layout="centered")

st.title("Markov Chains")

st.write('(in progress)')

fig, ax = plt.subplots(figsize=(3,3))
ax.axis("off")

circle = matplotlib.patches.Circle((-2, 0),radius= 1, color='black', fill=False)
ax.add_patch(circle)
plt.text(-2.7,-0.2, 'Heads', fontsize=10)
circle = matplotlib.patches.Circle((2, 0),radius= 1, color='black', fill=False)
ax.add_patch(circle)
plt.text(1.5,-0.2, 'Tails', fontsize=10)
ax.arrow(-1,0.5,2,0,
head_width=0.12,
    head_length=0.18,
    linewidth=1,
    color="black",
    length_includes_head=True,
)
ax.text(0,0.8,
    '0.3',
    fontsize=11,
    ha='center',
    va='center'
)
ax.arrow(1,-0.5,-2,0,
head_width=0.12,
    head_length=0.18,
    linewidth=1,
    color="black",
    length_includes_head=True,
)
ax.text(0,-0.9,
    '0.7',
    fontsize=11,
    ha='center',
    va='center'
)
arrow = FancyArrowPatch(
    (-2.6, 1), (-1.6, 1),
    connectionstyle="arc3,rad=-2.6",   # positive → counterclockwise
    arrowstyle='-|>',
    mutation_scale=12,
    linewidth=1,
    color="black"
)
ax.add_patch(arrow)
arrow = FancyArrowPatch(
    (2.6, 1), (1.6, 1),
    connectionstyle="arc3,rad=2.6",   # positive → counterclockwise
    arrowstyle='-|>',
    mutation_scale=12,
    linewidth=1,
    color="black"
)
ax.add_patch(arrow)
ax.text(-2,2.7,
    '0.7',
    fontsize=11,
    ha='center',
    va='center'
)
ax.text(2,2.7,
    '0.3',
    fontsize=11,
    ha='center',
    va='center'
)

plt.xlim([-4, 4])
plt.ylim([-4, 4])
centered_pyplot(fig)
plt.close(fig)

st.latex(r"\begin{matrix}P = \begin{bmatrix}0.7 & 0.3 \\ 0.3 & 0.6 \end{bmatrix}\end{matrix}")