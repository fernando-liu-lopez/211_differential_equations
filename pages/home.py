import streamlit as st


st.set_page_config(
    page_title="Home",
    layout="wide",
)

st.title("Differential Equations and Linear Algebra")

st.markdown(r"""Welcome to our course on Differential Equations and Linear Algebra! This app contains a collection of **interactive visualizations**
for differential equations, linear algebra, and their applications.""")


st.markdown(
    """
Click any of the links below to navigate to the appropriate interactive tool.
"""
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("First-order ODEs")

    st.markdown(
        """
        **[Newton's Law of Cooling](./newton_cooling)**  
        Visualize the cooling of an object over time according to Newton's Law of Cooling.
        """
    )
    st.markdown(
        """
        **[Slope Field Plotter](./Slope_Fields)**  
        Visualize direction fields and solution curves for first-order ODEs with initial conditions.
        """
    )
    
    st.subheader("Linear Algebra")
    st.markdown(
        """
        **[Kirchhoff's Circuit Laws](./kirchhoff_currents)**  
        Find current flows in circuits using Kirchhoff's Current Law and systems of linear equations.
        """
    )
    st.markdown(
        """
        **[Markov Chains](./markov)**  
        Learn about how matrices and matrix multiplication can model probabilistic processes that find applications in  word prediction, finance, and more.
        """
    )
    
with col2:
    st.subheader("Second-order ODEs")

    st.markdown(
        """
        **[Mass–Spring–Damper System](./Mass_Spring)**  
        Animate the motion governed by a mass-spring-damper system, including underdamped, critically damped, and overdamped cases.
        """
    )