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
        st.Page('pages/kirchhoff_currents.py', title = 'Circuits: Kirchhoff\'s Current Law')
    ],
    'Second Order Linear ODEs': [
        st.Page('pages/mass_spring.py',title='Mass-Spring Systems')
    ]
}


pg = st.navigation(pages)
pg.run()