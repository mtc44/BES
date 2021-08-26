# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:29:58 2021

@author: chanchanchan
"""
#Variables in Bender Element Analyisis:

#Fast Fourior Transform:

#Input Signal
Input_Signal_kHz = 3



from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import DissertationPlotwithDataMain as main 

change_in_frequency=[]

for frequency in np.arange(0, 2046, 1):
    change_in_time = 0.000001
    n = 2048
    k = 1/n/change_in_time
    change_in_frequency.append(frequency*k*0.001)

#FFT Graphs variables
#3
df_data3 = pd.DataFrame({'x':main.data3_input_new, 'y':main.data3_output_new})

fft_data3_input = np.fft.fft(df_data3['x']) #S1-F1
fft_data3_output_pre = [0] +(np.fft.fft(df_data3['y'])).tolist()
del fft_data3_output_pre[1]
fft_data3_output = np.array(fft_data3_output_pre)

mag_fft_data3_input = np.abs(fft_data3_input)
mag_fft_data3_output = np.abs(fft_data3_output)


#4
df_data4 = pd.DataFrame({'x':main.data4_input_new, 'y':main.data4_output_new})

fft_data4_input = np.fft.fft(df_data4['x'])
fft_data4_output_pre = [0] + (np.fft.fft(df_data4['y'])).tolist() #turn to list to set the first value to 0
del fft_data4_output_pre[1]
fft_data4_output = np.array(fft_data4_output_pre)

mag_fft_data4_input = np.abs(fft_data4_input)
mag_fft_data4_output = np.abs(fft_data4_output)


#5
df_data5 = pd.DataFrame({'x':main.data5_input_new, 'y':main.data5_output_new})

fft_data5_input = np.fft.fft(df_data5['x']) #S1-F1
fft_data5_output_pre = [0] +(np.fft.fft(df_data5['y'])).tolist()
del fft_data5_output_pre[1]
fft_data5_output = np.array(fft_data5_output_pre)

mag_fft_data5_input = np.abs(fft_data5_input)
mag_fft_data5_output = np.abs(fft_data5_output)


#6
df_data6 = pd.DataFrame({'x':main.data6_input_new, 'y':main.data6_output_new})

fft_data6_input = np.fft.fft(df_data6['x']) #S1-F1
fft_data6_output_pre = [0] +(np.fft.fft(df_data6['y'])).tolist()
del fft_data6_output_pre[1]
fft_data6_output = np.array(fft_data6_output_pre)

mag_fft_data6_input = np.abs(fft_data6_input)
mag_fft_data6_output = np.abs(fft_data6_output)


#7
df_data7 = pd.DataFrame({'x':main.data7_input_new, 'y':main.data7_output_new})

fft_data7_input = np.fft.fft(df_data7['x']) #S1-F1
fft_data7_output_pre = [0] +(np.fft.fft(df_data7['y'])).tolist()
del fft_data7_output_pre[1]
fft_data7_output = np.array(fft_data7_output_pre)

mag_fft_data7_input = np.abs(fft_data7_input)
mag_fft_data7_output = np.abs(fft_data7_output)


#Stacked phase variables
#3 output
real_fft_data3out = fft_data3_output.real
imag_fft_data3out = fft_data3_output.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data3out, imag_fft_data3out):
    
    TangentI = np.arctan(imag_data/real_data)
    
    np.seterr(invalid='ignore')
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
    
    
deg3_phaseout = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj3_1 = []

for deg in deg3_phaseout:
    
    if deg > 0:
        adj3_1.append(deg)
    elif deg <= 0:
        adj3_1.append(deg+360)


adj3_2 = []

for deg in deg3_phaseout:
    
    if deg > 0:
        adj3_2.append(deg)
    elif deg < 0:
        adj3_2.append(deg+360)
        

#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap3 = []

for adj_a, adj_b in zip(adj3_1, adj3_2):
    
    if adj_b > adj_a:
        unwrap3.append(adj_a+(360-adj_b))
    else:
        unwrap3.append(adj_a-adj_b)
        
stacked_phase3_pre = [0] + unwrap3

#cumulative summation for staced_phase
stacked_phase3out = np.cumsum(stacked_phase3_pre)



#3 input
real_fft_data3in = fft_data3_input.real
imag_fft_data3in = fft_data3_input.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data3in, imag_fft_data3in):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
    
    np.seterr(invalid='ignore')
    
deg3_phasein = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj3_1 = []

for deg in deg3_phasein:
    
    if deg > 0:
        adj3_1.append(deg)
    elif deg <= 0:
        adj3_1.append(deg+360)

del adj3_1[0]
adj3_1 = [360] + adj3_1

adj3_2 = []
    
for deg in deg3_phasein:
    
    if deg > 0:
        adj3_2.append(deg)
    elif deg < 0:
        adj3_2.append(deg+360)

del adj3_2[0] #for plotting with the same length

#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap3 = []

for adj_a, adj_b in zip(adj3_1, adj3_2):
    
    if adj_b > adj_a:
        unwrap3.append(adj_a+(360-adj_b))
    else:
        unwrap3.append(adj_a-adj_b)
        
stacked_phase3_pre = [0] + unwrap3

#cumulative summation for staced_phase
stacked_phase3in = np.cumsum(stacked_phase3_pre)



#4output
real_fft_data4out = fft_data4_output.real
imag_fft_data4out = fft_data4_output.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data4out, imag_fft_data4out):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
    np.seterr(invalid='ignore')
    
deg4_phaseout = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj4_1 = []

for deg in deg4_phaseout:
    
    if deg > 0:
        adj4_1.append(deg)
    elif deg <= 0:
        adj4_1.append(deg+360)



adj4_2 = []

for deg in deg4_phaseout:
    
    if deg > 0:
        adj4_2.append(deg)
    elif deg < 0:
        adj4_2.append(deg+360)

#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap4 = []

for adj_a, adj_b in zip(adj4_1, adj4_2):
    
    if adj_b > adj_a:
        unwrap4.append(adj_a+(360-adj_b))
    else:
        unwrap4.append(adj_a-adj_b)
        
stacked_phase4_pre = [0] + unwrap4

#cumulative summation for staced_phase
stacked_phase4out = np.cumsum(stacked_phase4_pre)



#4input
real_fft_data4in = fft_data4_input.real
imag_fft_data4in = fft_data4_input.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data4in, imag_fft_data4in):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
deg4_phasein = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj4_1 = []

for deg in deg4_phasein:
    
    if deg > 0:
        adj4_1.append(deg)
    elif deg <= 0:
        adj4_1.append(deg+360)
        
del adj4_1[0]
adj4_1 = [360] + adj4_1


adj4_2 = []

for deg in deg4_phasein:
    
    if deg > 0:
        adj4_2.append(deg)
    elif deg < 0:
        adj4_2.append(deg+360)
        
del adj4_2[0]

#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap4 = []

for adj_a, adj_b in zip(adj4_1, adj4_2):
    
    if adj_b > adj_a:
        unwrap4.append(adj_a+(360-adj_b))
    else:
        unwrap4.append(adj_a-adj_b)
        
stacked_phase4_pre = [0] + unwrap4

#cumulative summation for staced_phase
stacked_phase4in = np.cumsum(stacked_phase4_pre)



#5output
real_fft_data5out = fft_data5_output.real
imag_fft_data5out = fft_data5_output.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data5out, imag_fft_data5out):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
    
    np.seterr(invalid='ignore')
    
deg5_phaseout = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj5_1 = []

for deg in deg5_phaseout:
    
    if deg > 0:
        adj5_1.append(deg)
    elif deg <= 0:
        adj5_1.append(deg+360)


adj5_2 = []

for deg in deg5_phaseout:
    
    if deg > 0:
        adj5_2.append(deg)
    elif deg < 0:
        adj5_2.append(deg+360)


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap5 = []

for adj_a, adj_b in zip(adj5_1, adj5_2):
    
    if adj_b > adj_a:
        unwrap5.append(adj_a+(360-adj_b))
    else:
        unwrap5.append(adj_a-adj_b)
        
stacked_phase5_pre = [0] + unwrap5

#cumulative summation for staced_phase
stacked_phase5out = np.cumsum(stacked_phase5_pre)


#5input
real_fft_data5in = fft_data5_input.real
imag_fft_data5in = fft_data5_input.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data5in, imag_fft_data5in):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
deg5_phasein = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj5_1 = []

for deg in deg5_phasein:
    
    if deg > 0:
        adj5_1.append(deg)
    elif deg <= 0:
        adj5_1.append(deg+360)
        
del adj5_1[0]
adj5_1 = [360] + adj5_1


adj5_2 = []

for deg in deg5_phasein:
    
    if deg > 0:
        adj5_2.append(deg)
    elif deg < 0:
        adj5_2.append(deg+360)
        
del adj5_2[0] 


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap5 = []

for adj_a, adj_b in zip(adj5_1, adj5_2):
    
    if adj_b > adj_a:
        unwrap5.append(adj_a+(360-adj_b))
    else:
        unwrap5.append(adj_a-adj_b)
        
stacked_phase5_pre = [0] + unwrap5

#cumulative summation for staced_phase
stacked_phase5in = np.cumsum(stacked_phase5_pre)



#6output
real_fft_data6out = fft_data6_output.real
imag_fft_data6out = fft_data6_output.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data6out, imag_fft_data6out):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
deg6_phaseout = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj6_1 = []

for deg in deg6_phaseout:
    
    if deg > 0:
        adj6_1.append(deg)
    elif deg <= 0:
        adj6_1.append(deg+360)


adj6_2 = []

for deg in deg6_phaseout:
    
    if deg > 0:
        adj6_2.append(deg)
    elif deg < 0:
        adj6_2.append(deg+360)


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap6 = []

for adj_a, adj_b in zip(adj6_1, adj6_2):
    
    if adj_b > adj_a:
        unwrap6.append(adj_a+(360-adj_b))
    else:
        unwrap6.append(adj_a-adj_b)
        
stacked_phase6_pre = [0] + unwrap6

#cumulative summation for staced_phase
stacked_phase6out = np.cumsum(stacked_phase6_pre)


#6input
real_fft_data6in = fft_data6_input.real
imag_fft_data6in = fft_data6_input.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data6in, imag_fft_data6in):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
deg6_phasein = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj6_1 = []

for deg in deg6_phasein:
    
    if deg > 0:
        adj6_1.append(deg)
    elif deg <= 0:
        adj6_1.append(deg+360)

del adj6_1[0]
adj6_1 = [360] + adj6_1


adj6_2 = []

for deg in deg6_phasein:
    
    if deg > 0:
        adj6_2.append(deg)
    elif deg < 0:
        adj6_2.append(deg+360)
        
del adj6_2[0]


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap6 = []

for adj_a, adj_b in zip(adj6_1, adj6_2):
    
    if adj_b > adj_a:
        unwrap6.append(adj_a+(360-adj_b))
    else:
        unwrap6.append(adj_a-adj_b)
        
stacked_phase6_pre = [0] + unwrap6

#cumulative summation for staced_phase
stacked_phase6in = np.cumsum(stacked_phase6_pre)



#7output
real_fft_data7out = fft_data7_output.real
imag_fft_data7out = fft_data7_output.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data7out, imag_fft_data7out):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
deg7_phaseout = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj7_1 = []

for deg in deg7_phaseout:
    
    if deg > 0:
        adj7_1.append(deg)
    elif deg <= 0:
        adj7_1.append(deg+360)


adj7_2 = []

for deg in deg7_phaseout:
    
    if deg > 0:
        adj7_2.append(deg)
    elif deg < 0:
        adj7_2.append(deg+360)


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap7 = []

for adj_a, adj_b in zip(adj7_1, adj7_2):
    
    if adj_b > adj_a:
        unwrap7.append(adj_a+(360-adj_b))
    else:
        unwrap7.append(adj_a-adj_b)
        
stacked_phase7_pre = [0] + unwrap7

#cumulative summation for staced_phase
stacked_phase7out = np.cumsum(stacked_phase7_pre)



#7input
real_fft_data7in = fft_data7_input.real
imag_fft_data7in = fft_data7_input.imag
PIvalue = 4*np.arctan(1)

phase = []

for real_data, imag_data in zip(real_fft_data7in, imag_fft_data7in):
    
    TangentI = np.arctan(imag_data/real_data)
    
    if real_data > 0 and imag_data == 0:
        phase.append(0)
    elif real_data > 0 and imag_data > 0:   
        phase.append(TangentI)
    elif real_data > 0 and imag_data < 0:
        phase.append(TangentI + 2*PIvalue)
    
    elif real_data == 0 and imag_data == 0:
        phase.append(0)
    elif real_data == 0 and imag_data > 0:
        phase.append(PIvalue/2)
    elif real_data == 0 and imag_data < 0:
        phase.append(3*PIvalue/2)
    
    elif real_data < 0 and imag_data == 0:
        phase.append(PIvalue)
    elif real_data< 0 and imag_data > 0:
        phase.append(TangentI + PIvalue)
    elif real_data< 0 and imag_data < 0:
        phase.append(TangentI + PIvalue )
        
deg7_phasein = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj7_1 = []

for deg in deg7_phasein:
    
    if deg > 0:
        adj7_1.append(deg)
    elif deg <= 0:
        adj7_1.append(deg+360)
        
del adj7_1[0]
adj7_1 = [360] + adj7_1


adj7_2 = []

for deg in deg7_phasein:
    
    if deg > 0:
        adj7_2.append(deg)
    elif deg < 0:
        adj7_2.append(deg+360)
        
del adj7_2[0]


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap7 = []

for adj_a, adj_b in zip(adj7_1, adj7_2):
    
    if adj_b > adj_a:
        unwrap7.append(adj_a+(360-adj_b))
    else:
        unwrap7.append(adj_a-adj_b)
        
stacked_phase7_pre = [0] + unwrap7

#cumulative summation for staced_phase
stacked_phase7in = np.cumsum(stacked_phase7_pre)




if __name__ == "__main__" :
   
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 20])

if __name__ == "__main__" :

   if Input_Signal_kHz == 3:
    
       plt.plot(change_in_frequency, mag_fft_data3_input, label = '3kHz')
       plt.title('Fast Fourier Transform (Input)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.ylim([0, 700])
       plt.show()
       
       plt.plot(change_in_frequency, mag_fft_data3_output, label = '3kHz')
       plt.title('Fast Fourier Transform (Output)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.xlim([0, 20])
       plt.ylim([0, 700])
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.title('Stacked Phase')
       plt.ylabel('Stacked Phase (Degree)')
       plt.xlabel('Frequency (kHz)')
       plt.ylim([0, 8000])
       plt.xlim([0, 20])
       plt.scatter(change_in_frequency, stacked_phase3out, label = '3kHz')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
    
   elif Input_Signal_kHz == 4:
       plt.plot(change_in_frequency, mag_fft_data4_input, label = '4kHz')
       plt.title('Fast Fourier Transform (Input)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.plot(change_in_frequency, mag_fft_data4_output, label = '4kHz')
       plt.title('Fast Fourier Transform (Output)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.xlim([0, 20])
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.title('Stacked Phase')
       plt.ylabel('Stacked Phase (Degree)')
       plt.xlabel('Frequency (kHz)')
       plt.ylim([0, 8000])
       plt.xlim([0, 20])
       plt.scatter(change_in_frequency, stacked_phase4out, label = '4kHz')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()    

   elif Input_Signal_kHz == 5:
       plt.plot(change_in_frequency, mag_fft_data5_input, label = '5kHz')
       plt.title('Fast Fourier Transform (Input)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.plot(change_in_frequency, mag_fft_data5_output, label = '5kHz')
       plt.title('Fast Fourier Transform (Output)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.xlim([0, 20])
       plt.ylim([0, 700])
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.title('Stacked Phase')
       plt.ylabel('Stacked Phase (Degree)')
       plt.xlabel('Frequency (kHz)')
       plt.ylim([0, 8000])
       plt.xlim([0, 20])
       plt.scatter(change_in_frequency, stacked_phase5out, label = '5kHz')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()  
    
   elif Input_Signal_kHz == 6:
       plt.plot(change_in_frequency, mag_fft_data6_input, label = '6kHz')
       plt.title('Fast Fourier Transform (Input)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.plot(change_in_frequency, mag_fft_data6_output, label = '6kHz')
       plt.title('Fast Fourier Transform (Output)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.xlim([0, 20])
       plt.ylim([0, 700])
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.title('Stacked Phase')
       plt.ylabel('Stacked Phase (Degree)')
       plt.xlabel('Frequency (kHz)')
       plt.ylim([0, 8000])
       plt.xlim([0, 20])
       plt.scatter(change_in_frequency, stacked_phase6out, label = '6kHz')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()  
    
   elif Input_Signal_kHz == 7:
       plt.plot(change_in_frequency, mag_fft_data7_input, label = '7kHz')
       plt.title('Fast Fourier Transform (Input)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.plot(change_in_frequency, mag_fft_data7_output, label = '7kHz')
       plt.title('Fast Fourier Transform (Output)')
       plt.ylabel('Magnitude (Arbitary Units)')
       plt.xlim([0, 20])
       plt.ylim([0, 700])
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       plt.title('Stacked Phase')
       plt.ylabel('Stacked Phase (Degree)')
       plt.xlabel('Frequency (kHz)')
       plt.ylim([0, 8000])
       plt.xlim([0, 20])
       plt.scatter(change_in_frequency, stacked_phase7out, label = '7kHz')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()  
    

    
    
    