# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 23:31:18 2021

@author: chanchanchan
"""


import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from MultiappFrame import MultiApp #MultiappFrame set up the framework for different tabs 
import FirstArrivalNew, FastFouriorTransformNew, CrossCorrelationNew, StartPage, TransferFunctionNew # import app modules to generate multiapp (different tabs)

import DissertationPlotwithDataMain as main 



app = MultiApp()

st.title("Bender Element Analysis ")

st.sidebar.write('Interpretation of Bender Element:')

app.add_app("Summary of Bender Element Analysis", StartPage.app)
app.add_app("First Arrival", FirstArrivalNew.app)
app.add_app("Fast Fourior Transform", FastFouriorTransformNew.app)
app.add_app("Cross Correlation", CrossCorrelationNew.app)
app.add_app("Transfer Function", TransferFunctionNew.app)

# The main app
app.run()



