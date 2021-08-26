# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 01:26:15 2021

@author: chanchanchan
"""

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import DissertationPlotwithDataMain as main 
import FastFouriorTransform as faft
import CrossCorrelation as cc

def app():
    
    st.header('Time Domain Interpretation')
    st.subheader('Cross Correlation')
    
    #generate different select tabs
    signal= st.sidebar.selectbox('Frequency of Input Signal:', ['3kHz', '4kHz', '5kHz', '6kHz', '7kHz'])
    
    #generate slider
    level_of_crossing = st.sidebar.slider('Level of Crossing (%):', value = 50, min_value = 0, max_value = 100)
    
    if signal == '3kHz':
        
             
       max_mag_peak = max(max(abs(cc.peaks3_min)),max(cc.peaks3_max))
       upper_bound = level_of_crossing*0.01*max_mag_peak  # %Level
       lower_bound = -upper_bound


       peaks_level_max = []

       for peaks_level in cc.df_peak3['y']: 
           if peaks_level > upper_bound or peaks_level < lower_bound:
               peaks_level_max.append(peaks_level)
           else:
               pass

       peaks_level_max_point = cc.df_peak3.loc[cc.df_peak3['y'].isin(peaks_level_max)]
   
       cc3 = go.Figure()
       cc3.add_trace(go.Scatter(x = cc.data_time, y = cc.mag_CCsignal_data3, mode = 'lines', name = '3 kHz'))
       cc3.add_trace(go.Scatter(x = peaks_level_max_point['x'], y = peaks_level_max_point['y'], mode = 'markers', name = 'peaks'))
       cc3.update_layout(title={'text':"Cross Correlation Function",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time Delay (ms)", yaxis_title="Magnitude (Arbitary Units)")
       cc3.update_xaxes(range = [0, 2])
       cc3.update_yaxes(range = [-4, 4])
      
       st.write(cc3)
       

       peaks_table = go.Figure(data=[go.Table(header=dict(values=['Time (ms)', 'Magnitude'],line_color='darkslategray', fill_color='light blue'),
                     cells=dict(values=[peaks_level_max_point['x'], peaks_level_max_point['y']],line_color='darkslategray', fill_color='white'))
                     ])

       peaks_table.update_layout(title={'text':"Peaks from Cross Correlation Function:",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'})
       st.write(peaks_table)
   
    elif signal == '4kHz':
    
       max_mag_peak = max(max(abs(cc.peaks4_min)),max(cc.peaks4_max))
       upper_bound = level_of_crossing*0.01*max_mag_peak  # %Level
       lower_bound = -upper_bound


       peaks_level_max = []

       for peaks_level in cc.df_peak4['y']: 
           if peaks_level > upper_bound or peaks_level < lower_bound:
               peaks_level_max.append(peaks_level)
           else:
               pass

       peaks_level_max_point = cc.df_peak4.loc[cc.df_peak4['y'].isin(peaks_level_max)]
     
       cc4 = go.Figure()
       cc4.add_trace(go.Scatter(x = cc.data_time, y = cc.mag_CCsignal_data4, mode = 'lines', name = '4 kHz'))
       cc4.add_trace(go.Scatter(x = peaks_level_max_point['x'], y = peaks_level_max_point['y'], mode = 'markers', name = 'peaks'))
       cc4.update_layout(title={'text':"Cross Correlation Function",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time Delay (ms)", yaxis_title="Magnitude (Arbitary Units)")
       cc4.update_xaxes(range = [0, 2])
       cc4.update_yaxes(range = [-5, 5])
      
       st.write(cc4)
       
       peaks_table = go.Figure(data=[go.Table(header=dict(values=['Time (ms)', 'Magnitude'],line_color='darkslategray', fill_color='light blue'),
                     cells=dict(values=[peaks_level_max_point['x'], peaks_level_max_point['y']],line_color='darkslategray', fill_color='white'))
                     ])

       peaks_table.update_layout(title={'text':"Peaks from Cross Correlation Function:",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'})
       st.write(peaks_table)

    elif signal == '5kHz':
    
       max_mag_peak = max(max(abs(cc.peaks5_min)),max(cc.peaks5_max))
       upper_bound = level_of_crossing*0.01*max_mag_peak  # %Level
       lower_bound = -upper_bound


       peaks_level_max = []

       for peaks_level in cc.df_peak5['y']: 
           if peaks_level > upper_bound or peaks_level < lower_bound:
               peaks_level_max.append(peaks_level)
           else:
               pass

       peaks_level_max_point = cc.df_peak5.loc[cc.df_peak5['y'].isin(peaks_level_max)]
   
       cc5 = go.Figure()
       cc5.add_trace(go.Scatter(x = cc.data_time, y = cc.mag_CCsignal_data5, mode = 'lines', name = '5 kHz'))
       cc5.add_trace(go.Scatter(x = peaks_level_max_point['x'], y = peaks_level_max_point['y'], mode = 'markers', name = 'peaks'))
       cc5.update_layout(title={'text':"Cross Correlation Function",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time Delay (ms)", yaxis_title="Magnitude (Arbitary Units)")
       cc5.update_xaxes(range = [0, 2])
       cc5.update_yaxes(range = [-5, 5])
      
       st.write(cc5)
       
       peaks_table = go.Figure(data=[go.Table(header=dict(values=['Time (ms)', 'Magnitude'],line_color='darkslategray', fill_color='light blue'),
                     cells=dict(values=[peaks_level_max_point['x'], peaks_level_max_point['y']],line_color='darkslategray', fill_color='white'))
                     ])

       peaks_table.update_layout(title={'text':"Peaks from Cross Correlation Function:",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'})
       st.write(peaks_table)
       

    elif signal == '6kHz':
    
       max_mag_peak = max(max(abs(cc.peaks6_min)),max(cc.peaks6_max))
       upper_bound = level_of_crossing*0.01*max_mag_peak  # %Level
       lower_bound = -upper_bound


       peaks_level_max = []  

       for peaks_level in cc.df_peak6['y']: 
           if peaks_level > upper_bound or peaks_level < lower_bound:
               peaks_level_max.append(peaks_level)
           else:
               pass

       peaks_level_max_point = cc.df_peak6.loc[cc.df_peak6['y'].isin(peaks_level_max)]
   
       cc6 = go.Figure()
       cc6.add_trace(go.Scatter(x = cc.data_time, y = cc.mag_CCsignal_data6, mode = 'lines', name = '6 kHz'))
       cc6.add_trace(go.Scatter(x = peaks_level_max_point['x'], y = peaks_level_max_point['y'], mode = 'markers', name = 'peaks'))
       cc6.update_layout(title={'text':"Cross Correlation Function",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time Delay (ms)", yaxis_title="Magnitude (Arbitary Units)")
       cc6.update_xaxes(range = [0, 2])
       cc6.update_yaxes(range = [-5, 5])
      
       st.write(cc6)
       
       
       peaks_table = go.Figure(data=[go.Table(header=dict(values=['Time (ms)', 'Magnitude'],line_color='darkslategray', fill_color='light blue'),
                     cells=dict(values=[peaks_level_max_point['x'], peaks_level_max_point['y']],line_color='darkslategray', fill_color='white'))
                     ])

       peaks_table.update_layout(title={'text':"Peaks from Cross Correlation Function:",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'})
       st.write(peaks_table)
 
   
    elif signal == '7kHz':
    
       max_mag_peak = max(max(abs(cc.peaks7_min)),max(cc.peaks7_max))
       upper_bound = level_of_crossing*0.01*max_mag_peak  # %Level
       lower_bound = -upper_bound


       peaks_level_max = []

       for peaks_level in cc.df_peak7['y']: 
           if peaks_level > upper_bound or peaks_level < lower_bound:
               peaks_level_max.append(peaks_level)
           else:
               pass

       peaks_level_max_point = cc.df_peak7.loc[cc.df_peak7['y'].isin(peaks_level_max)]
   
       cc7 = go.Figure()
       cc7.add_trace(go.Scatter(x = cc.data_time, y = cc.mag_CCsignal_data7, mode = 'lines', name = '7 kHz'))
       cc7.add_trace(go.Scatter(x = peaks_level_max_point['x'], y = peaks_level_max_point['y'], mode = 'markers', name = 'peaks'))
       cc7.update_layout(title={'text':"Cross Correlation Function",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Time Delay (ms)", yaxis_title="Magnitude (Arbitary Units)")
       cc7.update_xaxes(range = [0, 2])
       cc7.update_yaxes(range = [-5, 4])
      
       st.write(cc7)
       
       
       peaks_table = go.Figure(data=[go.Table(header=dict(values=['Time (ms)', 'Magnitude'],line_color='darkslategray', fill_color='light blue'),
                     cells=dict(values=[peaks_level_max_point['x'], peaks_level_max_point['y']],line_color='darkslategray', fill_color='white'))
                     ])

       peaks_table.update_layout(title={'text':"Peaks from Cross Correlation Function:",'y':0.85,'x':0.5,'xanchor': 'center','yanchor': 'top'})
       st.write(peaks_table)
  