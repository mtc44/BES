# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 21:46:48 2021

@author: chanchanchan
"""

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import DissertationPlotwithDataMain as main 
import FastFouriorTransform as faft


def app():
    
    st.header('Fast Fourior Transfrom')
    
    #generate different select tabs with frequencies
    signal= st.sidebar.selectbox('Frequency of Input Signal:', ['3kHz', '4kHz', '5kHz', '6kHz', '7kHz'])

    #plotting graphs
    if signal == '3kHz':
        
        input3_faft = go.Figure()
        input3_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data3_input, mode = 'lines', name = '3 kHz'))
        input3_faft.update_layout( title={'text':"Input Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        input3_faft.update_xaxes(range = [0, 20])
        input3_faft.update_yaxes(range = [0, 700])
        st.write(input3_faft)
        
        output3_faft = go.Figure()
        output3_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data3_output, mode = 'lines', name = '3 kHz'))
        output3_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        output3_faft.update_xaxes(range = [0, 20])
        output3_faft.update_yaxes(range = [0, 700])
        st.write(output3_faft)
        
        stacked3_faft = go.Figure()
        stacked3_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.stacked_phase3out, mode = 'markers', name = '3 kHz'))
        stacked3_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        stacked3_faft.update_xaxes(range = [0, 20])
        stacked3_faft.update_yaxes(range = [0, 8000])
        st.write(stacked3_faft)
        
    elif signal == '4kHz':
        
        input4_faft = go.Figure()
        input4_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data4_input, mode = 'lines', name = '4 kHz'))
        input4_faft.update_layout( title={'text':"Input Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        input4_faft.update_xaxes(range = [0, 20])
        #input4_faft.update_yaxes(range = [0, 700])
        st.write(input4_faft)
        
        output4_faft = go.Figure()
        output4_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data4_output, mode = 'lines', name = '4 kHz'))
        output4_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        output4_faft.update_xaxes(range = [0, 20])
        #output4_faft.update_yaxes(range = [0, 700])
        st.write(output4_faft)
        
        stacked4_faft = go.Figure()
        stacked4_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.stacked_phase4out, mode = 'markers', name = '4 kHz'))
        stacked4_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        stacked4_faft.update_xaxes(range = [0, 20])
        stacked4_faft.update_yaxes(range = [0, 8000])
        st.write(stacked4_faft)
        
    elif signal == '5kHz':
        
        input5_faft = go.Figure()
        input5_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data5_input, mode = 'lines', name = '5 kHz'))
        input5_faft.update_layout( title={'text':"Input Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        input5_faft.update_xaxes(range = [0, 20])
        #input4_faft.update_yaxes(range = [0, 700])
        st.write(input5_faft)
        
        output5_faft = go.Figure()
        output5_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data5_output, mode = 'lines', name = '5 kHz'))
        output5_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        output5_faft.update_xaxes(range = [0, 20])
        #output4_faft.update_yaxes(range = [0, 700])
        st.write(output5_faft)
        
        stacked5_faft = go.Figure()
        stacked5_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.stacked_phase5out, mode = 'markers', name = '5 kHz'))
        stacked5_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        stacked5_faft.update_xaxes(range = [0, 20])
        stacked5_faft.update_yaxes(range = [0, 8000])
        st.write(stacked5_faft)
        
    elif signal == '6kHz':
        
        input6_faft = go.Figure()
        input6_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data6_input, mode = 'lines', name = '6 kHz'))
        input6_faft.update_layout( title={'text':"Input Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        input6_faft.update_xaxes(range = [0, 20])
        #input6_faft.update_yaxes(range = [0, 700])
        st.write(input6_faft)
        
        output6_faft = go.Figure()
        output6_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data6_output, mode = 'lines', name = '6 kHz'))
        output6_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        output6_faft.update_xaxes(range = [0, 20])
        #output6_faft.update_yaxes(range = [0, 700])
        st.write(output6_faft)
        
        stacked6_faft = go.Figure()
        stacked6_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.stacked_phase6out, mode = 'markers', name = '6 kHz'))
        stacked6_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        stacked6_faft.update_xaxes(range = [0, 20])
        stacked6_faft.update_yaxes(range = [0, 8000])
        st.write(stacked6_faft)
        
    elif signal == '7kHz':
        
        input7_faft = go.Figure()
        input7_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data7_input, mode = 'lines', name = '7 kHz'))
        input7_faft.update_layout( title={'text':"Input Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        input7_faft.update_xaxes(range = [0, 20])
        #input7_faft.update_yaxes(range = [0, 700])
        st.write(input7_faft)
        
        output7_faft = go.Figure()
        output7_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.mag_fft_data7_output, mode = 'lines', name = '7 kHz'))
        output7_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        output7_faft.update_xaxes(range = [0, 20])
        #output7_faft.update_yaxes(range = [0, 700])
        st.write(output7_faft)
        
        stacked7_faft = go.Figure()
        stacked7_faft.add_trace(go.Scatter(x = faft.change_in_frequency, y = faft.stacked_phase7out, mode = 'markers', name = '7 kHz'))
        stacked7_faft.update_layout( title={'text':"Output Signals",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        stacked7_faft.update_xaxes(range = [0, 20])
        stacked7_faft.update_yaxes(range = [0, 8000])
        st.write(stacked7_faft)
        
