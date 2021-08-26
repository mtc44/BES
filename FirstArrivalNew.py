# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:06:21 2021

@author: chanchanchan
"""

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import DissertationPlotwithDataMain as main 



def app():
    
   st.header('Time Domain Interpretation')
   st.subheader('Arrival Time Identification Methods')
   

   input_fa = go.Figure()

   input_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data3_input_new, mode = 'lines', name = '3 kHz'))
   input_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data4_input_new, mode = 'lines', name = '4 kHz'))
   input_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data5_input_new, mode = 'lines', name = '5 kHz'))
   input_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data6_input_new, mode = 'lines', name = '6 kHz'))
   input_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data7_input_new, mode = 'lines', name = '7 kHz'))
   input_fa.update_layout( title={'text':"Input Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time (ms)", yaxis_title="Output Voltage (Arbitary Units)")

   st.write(input_fa)
   
   st.sidebar.write('Output Signal:')

   output_3_checkbox = st.sidebar.checkbox("3kHz", value=True)
   output_4_checkbox = st.sidebar.checkbox("4kHz", value = True)
   output_5_checkbox = st.sidebar.checkbox("5kHz", value = True)
   output_6_checkbox = st.sidebar.checkbox("6kHz", value = True)
   output_7_checkbox = st.sidebar.checkbox("7kHz", value = True)
   
   output_fa = go.Figure()
   
   if output_3_checkbox:       
       output_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data3_output_new, mode = 'lines', name = '3 kHz'))
          
   if output_4_checkbox:
       output_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data4_output_new, mode = 'lines', name = '4 kHz'))
       
   if output_5_checkbox:
       output_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data5_output_new, mode = 'lines', name = '5 kHz'))
    
   if output_6_checkbox:
       output_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data6_output_new, mode = 'lines', name = '6 kHz'))
        
   if output_7_checkbox:
       output_fa.add_trace(go.Scatter(x = main.data3_time_new, y = main.data7_output_new, mode = 'lines', name = '7 kHz'))
       
   output_fa.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time (ms)", yaxis_title="Output Voltage (Arbitary Units)")
   st.write(output_fa)
   

   
   
   
