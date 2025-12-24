import streamlit as st
import numpy as np

# Pages
pages = {
    'Home': [
        st.Page('pages/home.py', title = 'Home')
    ],
    'First Order ODEs': [
        st.Page('pages/newton_cooling.py', title = 'Newton\'s Law of Cooling'),
        st.Page('pages/slope_fields.py', title = 'Slope Fields')
    ],
    'Linear Algebra': [
        st.Page('pages/kirchhoff_currents.py', title = 'Kirchhoff\'s Circuit Laws'),
        st.Page('pages/markov.py', title = 'Markov Chains')
    ],
    'Second Order Linear ODEs': [
        st.Page('pages/mass_spring.py',title='Mass-Spring Systems')
    ],
    'Miscellaneous': [
        st.Page('pages/circuit_terms.py',title='Basic Circuit Terms')
    ]
}


pg = st.navigation(pages)
pg.run()