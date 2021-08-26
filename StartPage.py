# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 20:23:42 2021

@author: chanchanchan
"""


import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def app():
    
   st.header('Summary of Shear Wave Velocity')

   
   st.write('The data used in this bender element anaylsis were obtained based on samples of stiff and overconsolidated London Clay.The measurements are taken in the vertical direction with horizontal polarisation.')
   
   st.subheader('Bender Element Parameters:')
   
   st.write('Soil mass density (kg/m3): 2000')
   st.write('Distance between two BE components (mm):98.36')
   st.write('Bender element length (mm):6')
   st.write('Travel distance (mm):93.36')
   
   st.subheader('Time Domain Interpretation:')
   
   st.write('The results obtained via the arrival time identification method and cross correlation method are shown below:')
   
   st.write('The time identification method is based on the start-to-start arrival time of the signal.')
   
   fig = go.Figure(data=[go.Table(
    header=dict(values=['Input signal frequency (kHz)', 'Interpretation method','Shear wave arrival time (ms)', 'Shear wave velocity (m/s)', 'Shear modulus (MPa)','L/ λ'],
                line_color='darkslategray',
                fill_color='light blue',
                align='center'),
    cells=dict(values=[[3, 3, 4, 4, 5, 5, 6, 6, 7, 7], # 1st column
                       ['Arrival time', 'Cross correlation','Arrival time', 'Cross correlation','Arrival time', 'Cross correlation','Arrival time', 'Cross correlation','Arrival time', 'Cross correlation'],# 2nd column
                       [0.55, 0.533, 0.53, 0.525, 0.50, 0.518, 0.48, 0.518, 0.47, 0.516],
                       [169.75, 175.16, 176.15, 177.83, 186.72, 180.23, 193.50, 180.23, 198.64, 180.93],
                       [57.63, 61.36, 62.06, 63.25, 69.73, 64.97, 74.88, 64.97, 78.92, 65.47],
                       [1.65, 1.60, 2.12, 2.10, 2.50, 2.59, 2.88, 3.11, 3.29, 3.61]], 
               line_color='darkslategray',
               fill_color='white',
               align='center'))
])

   fig.update_layout(width=700, height=600,)
   st.write(fig)
   
   st.subheader('Frequency Domain Interpretation:')
   
   st.write('The results obtained via the transfer function are shown below:')
   
   fig = go.Figure(data=[go.Table(
    header=dict(values=['Input signal frequency (kHz)', 'Interpretation method','Shear wave arrival time (ms)', 'Shear wave velocity (m/s)', 'Shear modulus (MPa)','L/ λ'],
                line_color='darkslategray',
                fill_color='light blue',
                align='center'),
    cells=dict(values=[[3, 4, 5, 6, 7], # 1st column
                       ['Transfer function', 'Transfer function','Transfer function','Transfer function','Transfer function'],# 2nd column
                       [0.396, 0.402, 0.402, 0.390, 0.406],
                       [235.76, 232.24, 232.24, 239.38, 229.95],
                       [111.17, 107.87, 107.87, 114.61, 105.75],
                       [1.58, 1.61, 1.61, 1.56, 1.62]], 
               line_color='darkslategray',
               fill_color='white',
               align='center'))
])
   
   fig.update_layout(width=700, height=500,)
   st.write(fig)
   
   st.subheader('Shear Wave Velocity Summary:')
   
   st.write('The shear wave velocity  obtained via the TD and FD methods are shown below:')
   
   
   fig = go.Figure()

   fig.add_trace(go.Scatter(
       x=[3, 4, 5 ,6, 7],
       y=[169.75, 176.15, 186.72, 193.50, 198.64],
       name="S-S arrival method"       
))


   fig.add_trace(go.Scatter(
       x=[3, 4, 5 ,6, 7],
       y=[175.16, 177.83, 180.23, 180.23, 180.93],
       name="Cross correlation"
))
   
   fig.add_trace(go.Scatter(
       x=[3, 4, 5 ,6, 7],
       y=[235.76, 232.24, 232.24, 239.38, 229.95],
       name="Transfer function"
))

   fig.update_layout(
       xaxis_title="Input signal frequency (kHz)",
       yaxis_title="Shear wave velocity (m/s)",
)
   
   
   st.write(fig)
   
   
