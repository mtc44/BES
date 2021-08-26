# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:33:01 2021

@author: chanchanchan
"""

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import DissertationPlotwithDataMain as main 
import FastFouriorTransform as faft
import TransferFunction as tf

def app():
    
    st.header('Frequency Domain Interpretation')
    st.subheader('Transfer Function')
    
    #generate different select tabs with frequencies
    signal= st.sidebar.selectbox('Frequency of Input Signal:', ['3kHz', '4kHz', '5kHz', '6kHz', '7kHz'])
    
    
    if signal == '3kHz':
        
        #input and output gain 
        gaininout = go.Figure()
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data3_input[0:23], mode = 'lines+markers', name = 'Input'))
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data3_output[0:23], mode = 'lines+markers', name = 'Output'))
        gaininout.update_layout( title={'text':"Input and Output Signal Magnitude",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        gaininout.update_xaxes(range = [0, 11])
        gaininout.update_yaxes(range = [0, 1000])
        
        st.write(gaininout)
        
        
        #input and output stacked phase
        phaseinout = go.Figure()
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase3in[0:23], mode = 'markers', name = 'Input'))
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase3out[0:23], mode = 'markers', name = 'Output'))
        phaseinout.update_layout( title={'text':"Input and Output Signal Stacked Phase",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        phaseinout.update_xaxes(range = [0, 11])
        phaseinout.update_yaxes(range = [0, 4000])
        
        st.write(phaseinout)
        
        #stacked phase vs L/lambda
        phaseout = go.Figure()
        phaseout.add_trace(go.Scatter(x = tf.D_lambda3[0:23], y = tf.stacked_phase3_trans[0:23], mode = 'lines+markers', name = 'Output'))
        phaseout.update_layout( title={'text':"Stacked Phase of Output Signal against L/lambda",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="L/lamda", yaxis_title="Stacked Phase (Degrees)")
        phaseout.update_xaxes(range = [0, 5.5])
        phaseout.update_yaxes(range = [-500, 2500])
        
        st.write(phaseout)
        
        #Gain and Phase plot vs frequnecy
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.stacked_phase3_trans[0:23], mode = 'lines+markers', name="Stacked Phase"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.mag3_trans[0:23], mode = 'lines+markers', name="Gain Factor"),secondary_y=True,)
        fig.update_xaxes(title_text="Frequency (kHz)")
        fig.update_yaxes(title_text="Phase (degrees)", secondary_y=False)
        fig.update_yaxes(title_text="Gain (Arbitary Units)", secondary_y=True)
        fig.update_layout( title={'text':'Gain Factor and Stacked Phase of the Bender Element Signal','y':0.85,'x':0.43,'xanchor': 'center','yanchor': 'top'})
        fig.update_xaxes(range = [0, 11])
        fig.update_yaxes(range = [-500, 2000], secondary_y=False)
        fig.update_yaxes(range = [0, 4.5], secondary_y=True)
        
        st.write(fig)
        
        #arrival time vs frequency 
        arrival = go.Figure()
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[1:23], y = tf.slope3_trans[0:22], mode ='lines+markers', name = 'Frequency Interval = 0.49kHz'))
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[7:12], y = tf.Tarr3_list, mode = 'lines', name = 'Frequency Interval = 1.96kHz'))
        arrival.update_layout( title={'text':"Shear Wave Arrival Time",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Arrival Time (ms)")
        arrival.update_xaxes(range = [0, 11])
        arrival.update_yaxes(range = [-200, 1600])
        
        st.write(arrival)
        
    if signal == '4kHz':
        
        #input and output gain 
        gaininout = go.Figure()
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data4_input[0:23], mode = 'lines+markers', name = 'Input'))
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data4_output[0:23], mode = 'lines+markers', name = 'Output'))
        gaininout.update_layout( title={'text':"Input and Output Signal Magnitude",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        gaininout.update_xaxes(range = [0, 11])
        gaininout.update_yaxes(range = [0, 1000])
        
        st.write(gaininout)
        
        
        #input and output stacked phase
        phaseinout = go.Figure()
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase4in[0:23], mode = 'markers', name = 'Input'))
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase4out[0:23], mode = 'markers', name = 'Output'))
        phaseinout.update_layout( title={'text':"Input and Output Signal Stacked Phase",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        phaseinout.update_xaxes(range = [0, 11])
        phaseinout.update_yaxes(range = [0, 4000])
        
        st.write(phaseinout)
        
        #stacked phase vs L/lambda
        phaseout = go.Figure()
        phaseout.add_trace(go.Scatter(x = tf.D_lambda4[0:19], y = tf.stacked_phase4_trans[0:19], mode = 'lines+markers', name = 'Output'))
        phaseout.update_layout( title={'text':"Stacked Phase of Output Signal against L/lambda",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="L/lamda", yaxis_title="Stacked Phase (Degrees)")
        phaseout.update_xaxes(range = [0, 5.5])
        phaseout.update_yaxes(range = [-500, 2500])
        
        st.write(phaseout)
        
        #Gain and Phase plot vs frequnecy
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.stacked_phase4_trans[0:23], mode = 'lines+markers', name="Stacked Phase"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.mag4_trans[0:23], mode = 'lines+markers', name="Gain Factor"),secondary_y=True,)
        fig.update_xaxes(title_text="Frequency (kHz)")
        fig.update_yaxes(title_text="Phase (degrees)", secondary_y=False)
        fig.update_yaxes(title_text="Gain (Arbitary Units)", secondary_y=True)
        fig.update_layout( title={'text':'Gain Factor and Stacked Phase of the Bender Element Signal','y':0.85,'x':0.43,'xanchor': 'center','yanchor': 'top'})
        fig.update_xaxes(range = [0, 11])
        fig.update_yaxes(range = [-500, 1600], secondary_y=False)
        fig.update_yaxes(range = [0, 1.6], secondary_y=True)
        
        st.write(fig)
        
        #arrival time vs frequency 
        
        arrival = go.Figure()
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[1:23], y = tf.slope4_trans[0:22], mode = 'lines+markers', name = 'Frequency Interval = 0.49kHz'))
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[7:12], y = tf.Tarr4_list, mode = 'lines', name = 'Frequency Interval = 1.96kHz'))
        arrival.update_layout( title={'text':"Shear Wave Arrival Time",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Arrival Time (ms)")
        arrival.update_xaxes(range = [0, 11])
        arrival.update_yaxes(range = [-400, 1600])
        
        st.write(arrival)
        
    if signal == '5kHz':
        
        #input and output gain 
        gaininout = go.Figure()
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data5_input[0:23], mode = 'lines+markers', name = 'Input'))
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data5_output[0:23], mode = 'lines+markers', name = 'Output'))
        gaininout.update_layout( title={'text':"Input and Output Signal Magnitude",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        gaininout.update_xaxes(range = [0, 11])
        gaininout.update_yaxes(range = [0, 1000])
        
        st.write(gaininout)
        
        
        #input and output stacked phase
        phaseinout = go.Figure()
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase5in[0:23], mode = 'markers', name = 'Input'))
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase5out[0:23], mode = 'markers', name = 'Output'))
        phaseinout.update_layout( title={'text':"Input and Output Signal Stacked Phase",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        phaseinout.update_xaxes(range = [0, 11])
        phaseinout.update_yaxes(range = [0, 4000])
        
        st.write(phaseinout)
        
        #stacked phase vs L/lambda
        phaseout = go.Figure()
        phaseout.add_trace(go.Scatter(x = tf.D_lambda5[0:19], y = tf.stacked_phase5_trans[0:19], mode = 'lines+markers', name = 'Output'))
        phaseout.update_layout( title={'text':"Stacked Phase of Output Signal against L/lambda",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="L/lamda", yaxis_title="Stacked Phase (Degrees)")
        phaseout.update_xaxes(range = [0, 5.5])
        phaseout.update_yaxes(range = [-500, 2500])
        
        st.write(phaseout)
        
        #Gain and Phase plot vs frequnecy
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.stacked_phase5_trans[0:23], mode = 'lines+markers', name="Stacked Phase"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.mag5_trans[0:23], mode = 'lines+markers', name="Gain Factor"),secondary_y=True,)
        fig.update_xaxes(title_text="Frequency (kHz)")
        fig.update_yaxes(title_text="Phase (degrees)", secondary_y=False)
        fig.update_yaxes(title_text="Gain (Arbitary Units)", secondary_y=True)
        fig.update_layout( title={'text':'Gain Factor and Stacked Phase of the Bender Element Signal','y':0.85,'x':0.43,'xanchor': 'center','yanchor': 'top'})
        fig.update_xaxes(range = [0, 11])
        fig.update_yaxes(range = [-500, 2000], secondary_y=False)
        fig.update_yaxes(range = [0, 3], secondary_y=True)
        
        st.write(fig)
        
        #arrival time vs frequency 
        
        arrival = go.Figure()
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[1:23], y = tf.slope5_trans[0:22], mode = 'lines+markers', name = 'Frequency Interval = 0.49kHz'))
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[7:12], y = tf.Tarr5_list, mode = 'lines', name = 'Frequency Interval = 1.96kHz'))
        arrival.update_layout( title={'text':"Shear Wave Arrival Time",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Arrival Time (ms)")
        arrival.update_xaxes(range = [0, 11])
        arrival.update_yaxes(range = [-200, 1600])
        
        st.write(arrival)
        
    if signal == '6kHz':
        
        #input and output gain 
        gaininout = go.Figure()
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data6_input[0:23], mode = 'lines+markers', name = 'Input'))
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data6_output[0:23], mode = 'lines+markers', name = 'Output'))
        gaininout.update_layout( title={'text':"Input and Output Signal Magnitude",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        gaininout.update_xaxes(range = [0, 11])
        gaininout.update_yaxes(range = [0, 1000])
        
        st.write(gaininout)
        
        
        #input and output stacked phase
        phaseinout = go.Figure()
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase6in[0:23], mode = 'markers', name = 'Input'))
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase6out[0:23], mode = 'markers', name = 'Output'))
        phaseinout.update_layout( title={'text':"Input and Output Signal Stacked Phase",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        phaseinout.update_xaxes(range = [0, 11])
        phaseinout.update_yaxes(range = [0, 4000])
        
        st.write(phaseinout)
        
        #stacked phase vs L/lambda
        phaseout = go.Figure()
        phaseout.add_trace(go.Scatter(x = tf.D_lambda3[0:19], y = tf.stacked_phase6_trans[0:19], mode = 'lines+markers', name = 'Output'))
        phaseout.update_layout( title={'text':"Stacked Phase of Output Signal against L/lambda",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="L/lamda", yaxis_title="Stacked Phase (Degrees)")
        phaseout.update_xaxes(range = [0, 5.5])
        phaseout.update_yaxes(range = [-500, 2500])
        
        st.write(phaseout)
        
        #Gain and Phase plot vs frequnecy
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.stacked_phase6_trans[0:23], mode = 'lines+markers', name="Stacked Phase"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.mag6_trans[0:23], mode = 'lines+markers', name="Gain Factor"),secondary_y=True,)
        fig.update_xaxes(title_text="Frequency (kHz)")
        fig.update_yaxes(title_text="Phase (degrees)", secondary_y=False)
        fig.update_yaxes(title_text="Gain (Arbitary Units)", secondary_y=True)
        fig.update_layout( title={'text':'Gain Factor and Stacked Phase of the Bender Element Signal','y':0.85,'x':0.43,'xanchor': 'center','yanchor': 'top'})
        fig.update_xaxes(range = [0, 11])
        fig.update_yaxes(range = [-500, 2500], secondary_y=False)
        fig.update_yaxes(range = [0, 1.6], secondary_y=True)
        
        st.write(fig)
        
        #arrival time vs frequency 
        
        arrival = go.Figure()
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[1:23], y = tf.slope6_trans[0:22], mode = 'lines+markers', name = 'Frequency Interval = 0.49kHz'))
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[7:12], y = tf.Tarr6_list, mode = 'lines', name = 'Frequency Interval = 1.96kHz'))
        arrival.update_layout( title={'text':"Shear Wave Arrival Time",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Arrival Time (ms)")
        arrival.update_xaxes(range = [0, 11])
        arrival.update_yaxes(range = [-200, 1600])
        
        st.write(arrival)
        
    if signal == '7kHz':
        
        #input and output gain 
        gaininout = go.Figure()
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data7_input[0:23], mode = 'lines+markers', name = 'Input'))
        gaininout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.mag_fft_data7_output[0:23], mode = 'lines+markers', name = 'Output'))
        gaininout.update_layout( title={'text':"Input and Output Signal Magnitude",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude (Arbitary Units)")
        gaininout.update_xaxes(range = [0, 11])
        gaininout.update_yaxes(range = [0, 1000])
        
        st.write(gaininout)
        
        
        #input and output stacked phase
        phaseinout = go.Figure()
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase7in[0:23], mode = 'markers', name = 'Input'))
        phaseinout.add_trace(go.Scatter(x = main.change_in_frequency[0:23], y = faft.stacked_phase7out[0:23], mode = 'markers', name = 'Output'))
        phaseinout.update_layout( title={'text':"Input and Output Signal Stacked Phase",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Stacked Phase (Degrees)")
        phaseinout.update_xaxes(range = [0, 11])
        phaseinout.update_yaxes(range = [0, 4000])
        
        st.write(phaseinout)
        
        #stacked phase vs L/lambda
        phaseout = go.Figure()
        phaseout.add_trace(go.Scatter(x = tf.D_lambda7[0:19], y = tf.stacked_phase7_trans[0:19], mode = 'lines', name = 'Output'))
        phaseout.update_layout( title={'text':"Stacked Phase of Output Signal against L/lambda",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="L/lamda", yaxis_title="Stacked Phase (Degrees)")
        phaseout.update_xaxes(range = [0, 5.5])
        phaseout.update_yaxes(range = [-500, 2500])
        
        st.write(phaseout)
        
        #Gain and Phase plot vs frequnecy
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.stacked_phase7_trans[0:23], mode = 'lines+markers', name="Stacked Phase"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=main.change_in_frequency[0:23], y=tf.mag7_trans[0:23], mode = 'lines+markers', name="Gain Factor"),secondary_y=True,)
        fig.update_xaxes(title_text="Frequency (kHz)")
        fig.update_yaxes(title_text="Phase (degrees)", secondary_y=False)
        fig.update_yaxes(title_text="Gain (Arbitary Units)", secondary_y=True)
        fig.update_layout( title={'text':'Gain Factor and Stacked Phase of the Bender Element Signal','y':0.85,'x':0.43,'xanchor': 'center','yanchor': 'top'})
        fig.update_xaxes(range = [0, 11])
        fig.update_yaxes(range = [-500, 2000], secondary_y=False)
        fig.update_yaxes(range = [0, 1.6], secondary_y=True)
        
        st.write(fig)
        
        #arrival time vs frequency 
        
        arrival = go.Figure()
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[1:23], y = tf.slope7_trans[0:22], mode = 'lines+markers', name = 'Frequency Interval = 0.49kHz'))
        arrival.add_trace(go.Scatter(x = main.change_in_frequency[7:12], y = tf.Tarr7_list, mode = 'lines', name = 'Frequency Interval = 1.96kHz'))
        arrival.update_layout( title={'text':"Shear Wave Arrival Time",'y':0.85,'x':0.45,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Frequency (kHz)", yaxis_title="Arrival Time (ms)")
        arrival.update_xaxes(range = [0, 11])
        arrival.update_yaxes(range = [-200, 1600])
        
        st.write(arrival)

