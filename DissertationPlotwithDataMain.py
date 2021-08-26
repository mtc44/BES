# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:09:40 2021

@author: chanchanchan
"""
#Variables in Bender Element Analyisis:
    
#First Arrival:
    
#Zoom-in function
Time_1 = 0 #unit: ms
Time_2 = 2 #unit: ms

Amplitude_1 = 3   #unit:V
Amplitude_2 = -3  #unit:V

#Boundary for first arrival peak 
Boundary_1 = 0.50 #unit: ms
Boundary_2 = 0.75 #unit: ms


    
#Fast Fourior Transform:


#Cross Correlation:
Level = 10 #%


from matplotlib import pyplot as plt
import pandas as pd


#Import Experimental Data From csv Files:

data3 = pd.read_csv('https://github.com/mtc44/BES/blob/main/3-khz.csv', delimiter = ',', index_col='Time in s')
data4 = pd.read_csv('https://github.com/mtc44/BES/blob/main/4-khz.csv', delimiter = ',', index_col='Time in s')
data5 = pd.read_csv('https://github.com/mtc44/BES/blob/main/5-khz.csv', delimiter = ',', index_col='Time in s')
data6 = pd.read_csv('https://github.com/mtc44/BES/blob/main/6-khz.csv', delimiter = ',', index_col='Time in s')
data7 = pd.read_csv('https://github.com/mtc44/BES/blob/main/7-khz.csv', delimiter = ',', index_col='Time in s')



data3_input = data3['CH I in voltage']
data4_input = data4['CH I in voltage']
data5_input = data5['CH I in voltage']
data6_input = data6['CH I in voltage']
data7_input = data7['CH I in voltage']

data3_output = data3['CH II in voltage']
data4_output = data4['CH II in voltage']
data5_output = data5['CH II in voltage']
data6_output = data6['CH II in voltage']
data7_output = data7['CH II in voltage']



#Max, Mean and S.D of Input and Output Signals:
import numpy as np

Input3_mean = data3_input.mean()
Output3_mean = data3_output.mean()

Input4_mean = data4_input.mean()
Output4_mean = data4_output.mean()

Input5_mean = data5_input.mean()
Output5_mean = data5_output.mean()

Input6_mean = data6_input.mean()
Output6_mean = data6_output.mean()

Input7_mean = data7_input.mean()
Output7_mean = data7_output.mean()


Input3_SD = np.std(data3_input)
Output3_SD = np.std(data3_output)

Input4_SD = np.std(data4_input)
Output4_SD = np.std(data4_output)

Input5_SD = np.std(data5_input)
Output5_SD = np.std(data5_output)

Input6_SD = np.std(data6_input)
Output6_SD = np.std(data6_output)

Input7_SD = np.std(data7_input)
Output7_SD = np.std(data7_output)


Input3_max = max(np.abs(data3_input))
Output3_max = max(np.abs(data3_output))

Input4_max = max(np.abs(data4_input))
Output4_max = max(np.abs(data4_output))

Input5_max = max(np.abs(data5_input))
Output5_max = max(np.abs(data5_output))

Input6_max = max(np.abs(data6_input))
Output6_max = max(np.abs(data6_output))

Input7_max = max(np.abs(data7_input))
Output7_max = max(np.abs(data7_output))




#Normalise Input and Output Signals:

for inputsignal in np.arange(Input3_max):
    data3_input_new = (data3_input - Input3_mean)/Input3_SD

for inputsignal in np.arange(Input4_max):
    data4_input_new = (data4_input - Input4_mean)/Input4_SD
    
for inputsignal in np.arange(Input5_max):
    data5_input_new = (data5_input - Input5_mean)/Input5_SD
    
for inputsignal in np.arange(Input6_max):
    data6_input_new = (data6_input - Input6_mean)/Input6_SD
    
for inputsignal in np.arange(Input7_max):
    data7_input_new = (data7_input - Input7_mean)/Input7_SD


for outputsignal in np.arange(0, Output3_max, 0.01):
    data3_output_new = (data3_output - Output3_mean)/Output3_SD

for outputsignal in np.arange(0, Output4_max, 0.01):
    data4_output_new = (data4_output - Output4_mean)/Output4_SD
    
for outputsignal in np.arange(0, Output5_max, 0.01):
    data5_output_new = (data5_output - Output5_mean)/Output5_SD

for outputsignal in np.arange(0, Output6_max, 0.01):
    data6_output_new = (data6_output - Output6_mean)/Output6_SD
    
for outputsignal in np.arange(0, Output7_max, 0.01):
    data7_output_new = (data7_output - Output7_mean)/Output7_SD




#Signal Offseting to Make the Input Signal Starts at t=0:
#plotting first arrival:
#applying boundary and zoom in on first arrival:

data3_time_new = []

for n in range(0,2046,1):
    data3_time_new.append(n*0.001-0.11)


if __name__ == "__main__" :

#Input signal    
  plt.plot(data3_time_new, data3_input_new, label = '3kHz')
  plt.plot(data3_time_new, data4_input_new, label = '4kHz')
  plt.plot(data3_time_new, data5_input_new, label = '5kHz')
  plt.plot(data3_time_new, data6_input_new, label = '6kHz')
  plt.plot(data3_time_new, data7_input_new, label = '7kHz')
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
  plt.title('Input Signal')
  plt.ylabel('Amplitude (V)')
  plt.xlabel('Time (ms)')
  plt.xlim([0, 1.0])
  plt.show()


#Output signal
#full image
if __name__ == "__main__" :
   plt.plot(data3_time_new, data3_output_new, label = '3kHz')
   plt.plot(data3_time_new, data4_output_new, label = '4kHz')
   plt.plot(data3_time_new, data5_output_new, label = '5kHz')
   plt.plot(data3_time_new, data6_output_new, label = '6kHz')
   plt.plot(data3_time_new, data7_output_new, label = '7kHz')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.title('Output Signal (overview)')
   plt.ylabel('Amplitude (V)')
   plt.xlabel('Time (ms)')
   plt.xlim([0,2])
   plt.show()
#zoom in image 
   plt.plot(data3_time_new, data3_output_new, label = '3kHz')
   plt.plot(data3_time_new, data4_output_new, label = '4kHz')
   plt.plot(data3_time_new, data5_output_new, label = '5kHz')
   plt.plot(data3_time_new, data6_output_new, label = '6kHz')
   plt.plot(data3_time_new, data7_output_new, label = '7kHz')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.title('Output Signal (zoom in)')
   plt.ylabel('Amplitude (V)')
   plt.xlabel('Time (ms)')
   plt.xlim([Time_1, Time_2]) #change to zoom in with specific time
   plt.ylim([Amplitude_1,Amplitude_2]) #change to zoom in with different amplitude
   plt.axvline(x=Boundary_1, ymin=0.05, ymax=0.95, color = 'black') # boundary for specific time period 
   plt.axvline(x=Boundary_2, ymin=0.05, ymax=0.95, color = 'black')
   plt.show()




#Application of FFT:

fft_data3_input = np.fft.fft(data3_input_new) #S1-F2
fft_data3_output = np.fft.fft(data3_output_new) #S1-F1

mag_fft_data3_input = np.abs(fft_data3_input)
mag_fft_data3_output = np.abs(fft_data3_output)


change_in_frequency=[]

for frequency in np.arange(0, 2046, 1):
    change_in_time = 0.000001
    n = 2048
    k = 1/n/change_in_time
    change_in_frequency.append(frequency*k*0.001)

#plot FFT 
if __name__ == "__main__" :
    
   plt.title('Fast Fourier Transform')
   plt.ylabel('Magnitude (Arbitary Units)')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 20])
   plt.plot(change_in_frequency, mag_fft_data3_output)
   plt.show()

#Stacked Phase Degree:

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
        
deg3out_phase = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap 
adj3out_1 = []

for deg in deg3out_phase:
    
    if deg > 0:
        adj3out_1.append(deg)
    elif deg <= 0:
        adj3out_1.append(deg+360)


adj3out_2 = []

for deg in deg3out_phase:
    
    if deg > 0:
        adj3out_2.append(deg)
    elif deg < 0:
        adj3out_2.append(deg+360)


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap3out = []

for adj_a, adj_b in zip(adj3out_1, adj3out_2):
    
    if adj_b > adj_a:
        unwrap3out.append(adj_a+(360-adj_b))
    else:
        unwrap3out.append(adj_a-adj_b)
        
stacked_phase3out_pre = [0] + unwrap3out

#cumulative summation for staced_phase
stacked_phase3out = np.cumsum(stacked_phase3out_pre)


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
        
deg3in_phase = np.degrees(phase)

#adj_1 and adj_2 created for the summation of unwrap
adj3in_1 = []

for deg in deg3in_phase:
    
    if deg > 0:
        adj3in_1.append(deg)
    elif deg <= 0:
        adj3in_1.append(deg+360)


adj3in_2 = []

for deg in deg3in_phase:
    
    if deg > 0:
        adj3in_2.append(deg)
    elif deg < 0:
        adj3in_2.append(deg+360)
        
del adj3in_2[0]


#unwrap and stacked_phase_pre created for the summation of stacked_phase_d
unwrap3in = []

for adj_a, adj_b in zip(adj3in_1, adj3in_2):
    
    if adj_b > adj_a:
        unwrap3in.append(adj_a+(360-adj_b))
    else:
        unwrap3in.append(adj_a-adj_b)
        
stacked_phase3in_pre = [0] + unwrap3in

#cumulative summation for staced_phase
stacked_phase3in = np.cumsum(stacked_phase3in_pre)


#plot stacked phase 

if __name__ == "__main__" : 
    
   plt.title('Stacked Phase')
   plt.ylabel('Stacked Phase (Degree)')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 20])
   plt.ylim([0, 8000])
   plt.scatter(change_in_frequency, stacked_phase3out)
   plt.show()






#Cross Spectra of Signals:

conj_fft_data3_input = np.conj(fft_data3_input)

CrossSpectra_data3 = conj_fft_data3_input*fft_data3_output 

fft_CrossSpectra_data3 = np.fft.ifft(CrossSpectra_data3)

real_fft_CS_data3 = fft_CrossSpectra_data3.real




#Max Magnitude of Output Signal:

max_data3_output_new = max(np.abs(data3_output_new))

normalise_CCsignal_data3 = max(np.abs(real_fft_CS_data3))/max_data3_output_new




#Find CC of Signal Output Variables:

mag_CCsignal_data3 = []

for magnitudes in real_fft_CS_data3:
    mag_CCsignal_data3.append(magnitudes/normalise_CCsignal_data3)
    
data3_time = []

for time in range(0,2046,1):
    data3_time.append(time*0.001)



#Find Minumum and Maximum of the CC Graphs:

from scipy.signal import find_peaks

array_mag_CCsignal_data3 = np.array(mag_CCsignal_data3) #turn list into array for math operation

array_mag_CCsignal_data3_2 = array_mag_CCsignal_data3*-1 #mirror to find minimum  

peaks_max_dict= find_peaks(array_mag_CCsignal_data3, height = -4) #indices and values of maximum peaks
peaks_max_height = peaks_max_dict[1]
peaks_max_time = peaks_max_dict[0]*0.001 #time of peaks


peaks_min_mirror = find_peaks(array_mag_CCsignal_data3_2, height = -4)#indices and values of minimum peaks
peaks_min_height = peaks_min_mirror[1] #mirrored values of minimum peaks
peaks_min_time = peaks_min_mirror[0]*0.001 #time of peaks

#turn dict into array for maths operation
peaks_min_turn = 1

for peak in peaks_min_height:
    peaks_min_turn = peaks_min_turn*peaks_min_height[peak]

peaks_max_turn = 1

for peak in peaks_max_height:
    peaks_max_turn = peaks_max_turn*peaks_max_height[peak]

peaks_min = peaks_min_turn*-1 #values of minimum peaks in array
peaks_max = peaks_max_turn    #values of maximum peaks in array

peaks_total = np.concatenate([peaks_min, peaks_max])
peaks_time_total = np.concatenate([peaks_min_time, peaks_max_time])


import pandas as pd

df_peak = pd.DataFrame({'x':peaks_time_total, 'y':peaks_total}) #turn into dataframe
#df_peak.plot('x', 'y', kind='scatter') #plot all peaks as scattered points



#% Level of the Peak Magnitude:

max_mag_peak = max(max(abs(peaks_min)),max(peaks_max))
upper_bound = Level*0.01*max_mag_peak  # %Level
lower_bound = -upper_bound


peaks_level_max = []

for peaks_level in df_peak['y']: 
    if peaks_level > upper_bound or peaks_level < lower_bound:
        peaks_level_max.append(peaks_level)
    else:
        pass

peaks_level_max_point = df_peak.loc[df_peak['y'].isin(peaks_level_max)]

#plot peaks with %level

if __name__ == "__main__" :
    
   peaks_level_max_point.plot.scatter(x = "x", y = "y", label = 'peaks', color = 'orange') 




#Plot CC Graphs:

if __name__ == "__main__" :    

   plt.title('Cross Correlation')
   plt.ylabel('Magnitude (Arbitary Units)')
   plt.xlabel('Time delay (ms)')
   plt.xlim([0, 2])
   plt.plot(data3_time, mag_CCsignal_data3)
   plt.legend()
   plt.show()



#Transfer Function:

mag_trans_pre = [0] + (mag_fft_data3_output/mag_fft_data3_input).tolist() #set the first value of the series = 0 
del mag_trans_pre [1]
mag_trans = mag_trans_pre
stacked_phase_trans = stacked_phase3out - stacked_phase3in



from scipy.stats import linregress

#input
slopein = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase3in[x-1] 
    x_2 = stacked_phase3in[x] 
    x_3 = stacked_phase3in[x+1] 
        
    x_4 = change_in_frequency[x-1]
    x_5 = change_in_frequency[x]
    x_6 = change_in_frequency[x+1]
        
    slopein.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope_input = np.array(slopein)/360*1000

#output
slopeout = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase3out[x-1] 
    x_2 = stacked_phase3out[x] 
    x_3 = stacked_phase3out[x+1] 
        
    x_4 = change_in_frequency[x-1]
    x_5 = change_in_frequency[x]
    x_6 = change_in_frequency[x+1]
        
    slopeout.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope_output = np.array(slopeout)/360*1000

#transfer 

pretrigger = 110
slopetrans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase_trans[x-1] 
    x_2 = stacked_phase_trans[x] 
    x_3 = stacked_phase_trans[x+1] 
        
    x_4 = change_in_frequency[x-1]
    x_5 = change_in_frequency[x]
    x_6 = change_in_frequency[x+1]
        
    slopetrans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope_trans = np.array(slopetrans)/360*1000 - pretrigger


# D/lambda

slope_x = change_in_frequency[8:10]
slope_y = stacked_phase_trans[8:10]

line_regress = linregress(slope_x, slope_y)
Tarr = line_regress[0]/360*1000

D_lambda = []

for value in change_in_frequency:
    D_lambda.append(value*Tarr/1000)
    

    

#Plot Transfer Function Graphs:

if __name__ == "__main__" :    
    
#Slope Transfer
   plt.title('Slope Transfer vs Frequency')
   plt.ylabel('Arrival time (ms)')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 10])
   plt.plot(change_in_frequency[1:20], slope_trans[0:19], label = 'Transfer')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.show()
   
   
#Transfer Phase/Mag  
   fig,ax_phase = plt.subplots()
   ax_phase.plot(change_in_frequency[0:20], stacked_phase_trans[0:20], label = 'Stacked Phase', color = 'red')
   plt.legend(bbox_to_anchor=(1.1, 1.1), loc= 'upper left')
   plt.ylabel('Phase (degrees)', color = 'red')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 10])
   plt.ylim([-400, 1600])
   plt.title('Transfer Phase/Mag vs Frequency')
   
   ax_mag = ax_phase.twinx()
   ax_mag.plot(change_in_frequency[0:20], mag_trans[0:20], label = 'Magnitude', color = 'blue')
   plt.legend(bbox_to_anchor=(1.1, 1), loc= 'upper left')
   plt.ylabel('Gain (Arbitary Units)', color = 'blue')
   plt.show()
   
#Input/Output Mag
   plt.title('Input/Output Magnitude vs Frequency')
   plt.ylabel('Magnitude (arb.unit)')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 10])
   plt.ylim([0, 1200])
   plt.plot(change_in_frequency[0:20], mag_fft_data3_input[0:20], label = 'Input')
   plt.plot(change_in_frequency[0:20], mag_fft_data3_output[0:20], label = 'Output')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.show()

#Transfer Mag
   plt.title('Transfer Magnitude vs Frequency')
   plt.ylabel('Gain (arbitary unit)')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 10])
   plt.plot(change_in_frequency[0:20], mag_trans[0:20], label = 'Transfer')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.show()

#Trans Stacked phase vs D/lambda
   plt.title('Transfer Stacked Phase vs D/lambda')
   plt.ylabel('Phase (degrees)')
   plt.xlabel('D/lambda')
   plt.xlim([0, 5])
   plt.plot(D_lambda[0:19], stacked_phase_trans[0:19], label = 'Transfer')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.show()


#Input/Output Phase
   plt.title('Input/Output Phase vs Frequency')
   plt.ylabel('Stacked Phase (degrees)')
   plt.xlabel('Frequency (kHz)')
   plt.xlim([0, 10])
   plt.ylim([0, 3000])
   plt.scatter(change_in_frequency[0:20], stacked_phase3in[0:20], label = 'Input')
   plt.scatter(change_in_frequency[0:20], stacked_phase3out[0:20], label = 'Output')
   plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
   plt.show()
   
   
