import streamlit as st
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, patches
from matplotlib.patches import FancyArrowPatch

from helpers import centered_pyplot, draw_current_source, draw_node, draw_wire

st.set_page_config(page_title="Markov Chains", layout="centered")

st.title("Markov Chains")

st.subheader('1. What is a Markov Chain?')

st.write('Suppose you had an unfair coin that had a 70% chance of landing on heads and a 30% chance of landing on tails. We might model the probabilities  of flipping heads or tails as follows:')

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

st.write('Our coin can be in one of two states: heads or tails. We represent these above as labeled vertices/circles. The arrows represent the probabilities of transitioning from one state to another. For example, if we are currently in the heads state, we have a 70% chance of remaining in the heads state and a 30% chance of transitioning to the tails state.')

st.write('Alternatively, we can represent the probabilities of transitioning between these states using a ***transition matrix***, where the entry in the i-th row and j-th column represents the probability of transitioning from state i to state j. For our unfair coin, the transition matrix would look like:')

st.latex(r"\begin{matrix}P = \begin{matrix}\textnormal{from heads}\\\textnormal{from tails}\end{matrix} \stackrel{\begin{matrix}\textnormal{to heads} & \textnormal{to tails}\end{matrix} }{\begin{bmatrix}\phantom{mm}0.7\phantom{mmmm} & 0.3\phantom{mm} \\ 0.3\phantom{mm} & 0.7\phantom{mm} \end{bmatrix}} \end{matrix}")

st.write('While we are at it, we can also represent our states using a ***state vectors***, where the i-th entry represents the probability of being in state i. For example:')

st.latex(r"\textnormal{last flipped heads:}\quad \begin{bmatrix}1 \\ 0\end{bmatrix} \qquad\qquad\textnormal{last flipped tails:}\quad \begin{bmatrix}0 \\ 1\end{bmatrix}")

st.write('Now, if we wanted to compute the probability of flipping heads or tails on the next flip, we could multiply our transition matrix by our state vector. For example, if we just flipped heads, we would compute:')

st.latex(r"\begin{bmatrix}0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix} \begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}0.7 \\ 0.3\end{bmatrix}")

st.write('which tells us that we have a 70% chance of flipping heads again and a 30% chance of flipping tails.')

st.divider()

st.subheader('2. Markov Chains for Predictive Text Generation')

st.write('Suppose you work with a group of monkeys that know four words: ``cats``, ``dogs``, ``like``, and ``don\'t``. Despite their limited vocabulary, the monkeys are tech-savy and each have cellphones. Unfortunately, their cell data is leaked, and you find the following trends:')

st.write(r"""- When the last word a monkey typed was `cats`, the chance of the next word also being `cats` was pretty low (say $10\%=0.1=\frac{10}{100}$). The probabilities for ``dogs``, ``like``, and ``don't`` were $0.1$, $0.3$, and  $0.5$ respectively. You encode these into a probability vector:""")

st.latex(r"\begin{matrix}\textnormal{cats} \\ \textnormal{dogs} \\ \textnormal{like} \\ \textnormal{don't} \end{matrix}\begin{bmatrix} 0.1 \\ 0.1 \\ 0.3 \\ 0.5\end{bmatrix}")

st.write(r"""- Similarly, if the last word typed was `dogs`, the probabilities for the next word were $(0.1, 0.2, 0.6, 0.1)$.  """)

st.write(r"- For `like`, the probabilities were $(0.35, 0.35, 0.1, 0.2)$.")

st.write(r"- Finally, for `don't`, the probabilities  $(0.25, 0.25, 0.5, 0)$.")

st.write(r"Suppose you're texting one of these monkeys (why?). Your conversation is described by the vector $\mathbf{\vec{x}}=(0,0,1,0)$, meaning the last word typed was ``like`` (with $100\%$ probability).")

st.write(r"(a) Find a matrix $\mathbf{A}$ such that the entries of $\mathbf{A}\mathbf{\vec{x}}$ give the probabilities for what the next word will be, no matter what $\mathbf{\vec{x}}$ is.")

st.write(r"(b) Find a matrix such that, given a vector $\mathbf{\vec{x}}$ as in (a), computes the probabilities for the \textit{second} word a user is likely to type.")

st.write(r"(c) With an online tool or calculator, find $\mathbf{A}^{500}$. What interpretation can you give to the columns of this matrix?")

st.divider()

st.subheader("3. Markov Chain Simulator")

def normalize_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = A.astype(float)
    col_sums = A.sum(axis=0)
    col_sums = np.where(np.abs(col_sums) < eps, 1.0, col_sums)
    return A / col_sums

def normalize_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(float)
    s = x.sum()
    if abs(s) < eps:
        return np.ones_like(x) / len(x)
    return x / s

def apply_steps(A: np.ndarray, x0: np.ndarray, n: int) -> np.ndarray:
    x = x0.copy()
    for _ in range(n):
        x = A @ x
    return x

def simulate(A: np.ndarray, x0: np.ndarray, steps: int) -> np.ndarray:
    k = len(x0)
    X = np.zeros((steps + 1, k), dtype=float)
    X[0] = x0
    x = x0.copy()
    for i in range(1, steps + 1):
        x = A @ x
        X[i] = x
    return X

k = st.number_input("Number of states", min_value=2, max_value=10, value=3, step=1)

st.markdown(
    r"""
Enter your transition matrix and initial state vector below. 
"""
)

# A default example (3-state)
default_A = np.array([
    [0.0, 1.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.0, 0.0],
], dtype=float)

if k != 3:
    A_init = np.eye(k) * 0.7 + (np.ones((k, k)) - np.eye(k)) * (0.3 / (k - 1))
else:
    A_init = default_A

with st.expander("Edit transition matrix.", expanded=True):
    cols = st.columns(k)
    A = np.zeros((k, k), dtype=float)
    for j in range(k):
        with cols[j]:
            st.markdown(f"State {j+1}")
            for i in range(k):
                A[i, j] = st.number_input(
                    f"A[{i+1},{j+1}]",
                    min_value=0.0,
                    value=float(A_init[i, j]) if i < A_init.shape[0] and j < A_init.shape[1] else (1.0 if i == j else 0.0),
                    step=0.05,
                    format="%.2f",
                    key=f"A_{i}_{j}",
                )

auto_normalize = st.checkbox("Auto-normalize columns to sum to 1", value=True)
if auto_normalize:
    A = normalize_columns(A)

col_sums = A.sum(axis=0)
# st.markdown("Column sums (should be 1): " + ", ".join([f"{s:.3f}" for s in col_sums]))

labels = [f"{i+1}" for i in range(k)]

x0 = np.zeros(k, dtype=float)
with st.expander("Initial state vector x₀", expanded=True):
    xcols = st.columns(k)
    for i in range(k):
        with xcols[i]:
            x0[i] = st.number_input(
                f"State {i+1}",
                min_value=0.0,
                value=1.0 if i == 0 else 0.0,
                step=0.05,
                format="%.2f",
                key=f"x0_{i}",
            )

x0 = normalize_vec(x0)

n = st.slider("Step number to compute (xₙ = Aⁿ x₀)", min_value=0, max_value=50, value=2)
steps_for_plot = st.slider("Number of steps to graph", min_value=max(5, n), max_value=200, value=max(10, n))

show_steady = st.checkbox("Also estimate a long-run (steady) distribution by iterating", value=True)
steady_steps = st.number_input("Iterations for long-run estimate", min_value=50, max_value=5000, value=500, step=50)

xn = apply_steps(A, x0, n)
X = simulate(A, x0, steps=int(steps_for_plot))

if show_steady:
    x_long = apply_steps(A, x0, int(steady_steps))
else:
    x_long = None

st.divider()

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.markdown(rf"**Probability Distribution at Step n={n}**")
    fig_b, ax_b = plt.subplots(figsize=(5.5, 4.0))
    bars = ax_b.bar(labels, xn)
    ax_b.set_ylim(0, 1)
    for rect in bars:
        height = rect.get_height()
        ax_b.text(
            rect.get_x() + rect.get_width() / 2,
            height+0.1,
            f"{height:.3f}",
            ha="center",
            va="center",
            fontsize=20,
            clip_on=False,
        )
    ax_b.set_xlabel("State")
    ax_b.set_ylabel("Probability")
    ax_b.grid(True, alpha=0.2, axis="y")
    st.pyplot(fig_b)
    plt.close(fig_b)

with c2:
    if x_long is not None:
        st.markdown(rf"**Long-run Estimate after {int(steady_steps)} steps**")
        fig_s, ax_s = plt.subplots(figsize=(5.5, 4.0))
        long_bars = ax_s.bar(labels, x_long)
        ax_s.set_ylim(0, 1)
        for rect in long_bars:
            height = rect.get_height()
            ax_s.text(
                rect.get_x() + rect.get_width() / 2,
                height+0.1,
                f"{height:.3f}",
                ha="center",
                va="center",
                fontsize=20,
                clip_on=False,
            )
        ax_s.set_xlabel("State")
        ax_s.set_ylabel("Probability")
        ax_s.grid(True, alpha=0.2, axis="y")
        st.pyplot(fig_s)
        plt.close(fig_s)
    else:
        st.markdown("**Long-run estimate**")
        st.info("Enable the long-run estimate checkbox to compute a steady pattern.")

st.markdown("**Evolution over time**")
fig_l, ax_l = plt.subplots(figsize=(11, 4.5))
t_steps = np.arange(X.shape[0])
for i in range(k):
    ax_l.plot(t_steps, X[:, i], linewidth=2, label=f"State {i+1}")
ax_l.set_xlabel("Step")
ax_l.set_ylabel("Probability")
ax_l.set_ylim(0, 1)
ax_l.grid(True, alpha=0.25)
ax_l.legend(ncol=min(k, 4))
st.pyplot(fig_l)
plt.close(fig_l)
