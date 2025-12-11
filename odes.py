import streamlit as st
import numpy as np

# Pages
pages = {
    'Home': [
        st.Page('pages/home.py', title = 'Math 211 Differential Equations')
    ],
    'First Order ODEs': [
        st.Page('pages/newton_cooling.py', title = 'Newton\'s Law of Cooling'),
        st.Page('pages/slope_fields.py', title = 'Slope Fields')
    ],
    'Linear Algebra': [
        st.Page('pages/rc_circuits.py', title = 'RC Circuits')
    ],
    'Second Order Linear ODEs': [
        st.Page('pages/harmonic_oscillators.py', title = 'Harmonic Oscillators'),
        st.Page('pages/mass_spring.py',title='Mass-Spring Systems'),
        st.Page('pages/rlc_circuits.py', title = 'RLC Circuits'),
    ]
}


pg = st.navigation(pages)
pg.run()