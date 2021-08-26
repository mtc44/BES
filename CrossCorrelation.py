# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:16:53 2021

@author: chanchanchan
"""

#Variables in Bender Element Analyisis:

#Cross Correlation:

#Input Signal
Input_Signal_kHz = 3
Level_of_Crossing = 80  # in %


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import DissertationPlotwithDataMain as main 
import FastFouriorTransform as faft



#Cross Spectra of Signals:

#3
conj_fft_data3_input = np.conj(main.fft_data3_input)
CrossSpectra_data3 = conj_fft_data3_input*main.fft_data3_output 
fft_CrossSpectra_data3 = np.fft.ifft(CrossSpectra_data3)
real_fft_CS_data3 = fft_CrossSpectra_data3.real

#4
conj_fft_data4_input = np.conj(faft.fft_data4_input)
CrossSpectra_data4 = conj_fft_data4_input*faft.fft_data4_output 
fft_CrossSpectra_data4 = np.fft.ifft(CrossSpectra_data4)
real_fft_CS_data4 = fft_CrossSpectra_data4.real

#5
conj_fft_data5_input = np.conj(faft.fft_data5_input)
CrossSpectra_data5 = conj_fft_data5_input*faft.fft_data5_output 
fft_CrossSpectra_data5 = np.fft.ifft(CrossSpectra_data5)
real_fft_CS_data5 = fft_CrossSpectra_data5.real

#6
conj_fft_data6_input = np.conj(faft.fft_data6_input)
CrossSpectra_data6 = conj_fft_data6_input*faft.fft_data6_output 
fft_CrossSpectra_data6 = np.fft.ifft(CrossSpectra_data6)
real_fft_CS_data6 = fft_CrossSpectra_data6.real

#7
conj_fft_data7_input = np.conj(faft.fft_data7_input)
CrossSpectra_data7 = conj_fft_data7_input*faft.fft_data7_output 
fft_CrossSpectra_data7 = np.fft.ifft(CrossSpectra_data7)
real_fft_CS_data7 = fft_CrossSpectra_data7.real



#Max Magnitude of Output Signal:

#3
max_data3_output_new = max(np.abs(main.data3_output_new))
normalise_CCsignal_data3 = max(np.abs(real_fft_CS_data3))/max_data3_output_new

#4
max_data4_output_new = max(np.abs(main.data4_output_new))
normalise_CCsignal_data4 = max(np.abs(real_fft_CS_data4))/max_data4_output_new

#5
max_data5_output_new = max(np.abs(main.data5_output_new))
normalise_CCsignal_data5 = max(np.abs(real_fft_CS_data5))/max_data5_output_new

#6
max_data6_output_new = max(np.abs(main.data6_output_new))
normalise_CCsignal_data6 = max(np.abs(real_fft_CS_data6))/max_data6_output_new

#7
max_data7_output_new = max(np.abs(main.data7_output_new))
normalise_CCsignal_data7 = max(np.abs(real_fft_CS_data7))/max_data7_output_new



#Find CC of Signal Output Variables:

data_time = []

for time in range(0,2046,1):
    data_time.append(time*0.001)    

#3
mag_CCsignal_data3 = []

for magnitudes in real_fft_CS_data3:
    mag_CCsignal_data3.append(magnitudes/normalise_CCsignal_data3)
    
#4
mag_CCsignal_data4 = []

for magnitudes in real_fft_CS_data4:
    mag_CCsignal_data4.append(magnitudes/normalise_CCsignal_data4)

#5
mag_CCsignal_data5 = []

for magnitudes in real_fft_CS_data5:
    mag_CCsignal_data5.append(magnitudes/normalise_CCsignal_data5)

#6
mag_CCsignal_data6 = []

for magnitudes in real_fft_CS_data6:
    mag_CCsignal_data6.append(magnitudes/normalise_CCsignal_data6)

#7
mag_CCsignal_data7 = []

for magnitudes in real_fft_CS_data7:
    mag_CCsignal_data7.append(magnitudes/normalise_CCsignal_data7)





#Find Minumum and Maximum of the CC Graphs:

from scipy.signal import find_peaks

#3
array_mag_CCsignal_data3 = np.array(mag_CCsignal_data3) #turn list into array for math operation

array_mag_CCsignal_data3_2 = array_mag_CCsignal_data3*-1 #mirror to find minimum  

peaks3_max_dict= find_peaks(array_mag_CCsignal_data3, height = -4) #indices and values of maximum peaks
peaks3_max_height = peaks3_max_dict[1]
peaks3_max_time = peaks3_max_dict[0]*0.001 #time of peaks


peaks3_min_mirror = find_peaks(array_mag_CCsignal_data3_2, height = -4)#indices and values of minimum peaks
peaks3_min_height = peaks3_min_mirror[1] #mirrored values of minimum peaks
peaks3_min_time = peaks3_min_mirror[0]*0.001 #time of peaks

#turn dict into array for maths operation
peaks3_min_turn = 1

for peak in peaks3_min_height:
    peaks3_min_turn = peaks3_min_turn*peaks3_min_height[peak]

peaks3_max_turn = 1

for peak in peaks3_max_height:
    peaks3_max_turn = peaks3_max_turn*peaks3_max_height[peak]

peaks3_min = peaks3_min_turn*-1 #values of minimum peaks in array
peaks3_max = peaks3_max_turn    #values of maximum peaks in array

peaks3_total = np.concatenate([peaks3_min, peaks3_max])
peaks3_time_total = np.concatenate([peaks3_min_time, peaks3_max_time])

df_peak3 = pd.DataFrame({'x':peaks3_time_total, 'y':peaks3_total})


#4
array_mag_CCsignal_data4 = np.array(mag_CCsignal_data4) #turn list into array for math operation

array_mag_CCsignal_data4_2 = array_mag_CCsignal_data4*-1 #mirror to find minimum  

peaks4_max_dict= find_peaks(array_mag_CCsignal_data4, height = -5) #indices and values of maximum peaks
peaks4_max_height = peaks4_max_dict[1]
peaks4_max_time = peaks4_max_dict[0]*0.001 #time of peaks


peaks4_min_mirror = find_peaks(array_mag_CCsignal_data4_2, height = -5)#indices and values of minimum peaks
peaks4_min_height = peaks4_min_mirror[1] #mirrored values of minimum peaks
peaks4_min_time = peaks4_min_mirror[0]*0.001 #time of peaks

#turn dict into array for maths operation
peaks4_min_turn = 1

for peak in peaks4_min_height:
    peaks4_min_turn = peaks4_min_turn*peaks4_min_height[peak]

peaks4_max_turn = 1

for peak in peaks4_max_height:
    peaks4_max_turn = peaks4_max_turn*peaks4_max_height[peak]

peaks4_min = peaks4_min_turn*-1 #values of minimum peaks in array
peaks4_max = peaks4_max_turn    #values of maximum peaks in array

peaks4_total = np.concatenate([peaks4_min, peaks4_max])
peaks4_time_total = np.concatenate([peaks4_min_time, peaks4_max_time])

df_peak4 = pd.DataFrame({'x':peaks4_time_total, 'y':peaks4_total})


#5
array_mag_CCsignal_data5 = np.array(mag_CCsignal_data5) #turn list into array for math operation

array_mag_CCsignal_data5_2 = array_mag_CCsignal_data5*-1 #mirror to find minimum  

peaks5_max_dict= find_peaks(array_mag_CCsignal_data5, height = -5) #indices and values of maximum peaks
peaks5_max_height = peaks5_max_dict[1]
peaks5_max_time = peaks5_max_dict[0]*0.001 #time of peaks


peaks5_min_mirror = find_peaks(array_mag_CCsignal_data5_2, height = -5)#indices and values of minimum peaks
peaks5_min_height = peaks5_min_mirror[1] #mirrored values of minimum peaks
peaks5_min_time = peaks5_min_mirror[0]*0.001 #time of peaks

#turn dict into array for maths operation
peaks5_min_turn = 1

for peak in peaks5_min_height:
    peaks5_min_turn = peaks5_min_turn*peaks5_min_height[peak]

peaks5_max_turn = 1

for peak in peaks5_max_height:
    peaks5_max_turn = peaks5_max_turn*peaks5_max_height[peak]

peaks5_min = peaks5_min_turn*-1 #values of minimum peaks in array
peaks5_max = peaks5_max_turn    #values of maximum peaks in array

peaks5_total = np.concatenate([peaks5_min, peaks5_max])
peaks5_time_total = np.concatenate([peaks5_min_time, peaks5_max_time])

df_peak5 = pd.DataFrame({'x':peaks5_time_total, 'y':peaks5_total})


#6
array_mag_CCsignal_data6 = np.array(mag_CCsignal_data6) #turn list into array for math operation

array_mag_CCsignal_data6_2 = array_mag_CCsignal_data6*-1 #mirror to find minimum  

peaks6_max_dict= find_peaks(array_mag_CCsignal_data6, height = -5) #indices and values of maximum peaks
peaks6_max_height = peaks6_max_dict[1]
peaks6_max_time = peaks6_max_dict[0]*0.001 #time of peaks


peaks6_min_mirror = find_peaks(array_mag_CCsignal_data6_2, height = -5)#indices and values of minimum peaks
peaks6_min_height = peaks6_min_mirror[1] #mirrored values of minimum peaks
peaks6_min_time = peaks6_min_mirror[0]*0.001 #time of peaks

#turn dict into array for maths operation
peaks6_min_turn = 1

for peak in peaks6_min_height:
    peaks6_min_turn = peaks6_min_turn*peaks6_min_height[peak]

peaks6_max_turn = 1

for peak in peaks6_max_height:
    peaks6_max_turn = peaks6_max_turn*peaks6_max_height[peak]

peaks6_min = peaks6_min_turn*-1 #values of minimum peaks in array
peaks6_max = peaks6_max_turn    #values of maximum peaks in array

peaks6_total = np.concatenate([peaks6_min, peaks6_max])
peaks6_time_total = np.concatenate([peaks6_min_time, peaks6_max_time])

df_peak6 = pd.DataFrame({'x':peaks6_time_total, 'y':peaks6_total})


#7
array_mag_CCsignal_data7 = np.array(mag_CCsignal_data7) #turn list into array for math operation

array_mag_CCsignal_data7_2 = array_mag_CCsignal_data7*-1 #mirror to find minimum  

peaks7_max_dict= find_peaks(array_mag_CCsignal_data7, height = -5) #indices and values of maximum peaks
peaks7_max_height = peaks7_max_dict[1]
peaks7_max_time = peaks7_max_dict[0]*0.001 #time of peaks


peaks7_min_mirror = find_peaks(array_mag_CCsignal_data7_2, height = -5)#indices and values of minimum peaks
peaks7_min_height = peaks7_min_mirror[1] #mirrored values of minimum peaks
peaks7_min_time = peaks7_min_mirror[0]*0.001 #time of peaks

#turn dict into array for maths operation
peaks7_min_turn = 1

for peak in peaks7_min_height:
    peaks7_min_turn = peaks7_min_turn*peaks7_min_height[peak]

peaks7_max_turn = 1

for peak in peaks7_max_height:
    peaks7_max_turn = peaks7_max_turn*peaks7_max_height[peak]

peaks7_min = peaks7_min_turn*-1 #values of minimum peaks in array
peaks7_max = peaks7_max_turn    #values of maximum peaks in array

peaks7_total = np.concatenate([peaks7_min, peaks7_max])
peaks7_time_total = np.concatenate([peaks7_min_time, peaks7_max_time])

df_peak7 = pd.DataFrame({'x':peaks7_time_total, 'y':peaks7_total})



#variable if statement

if __name__ == "__main__" :

   if Input_Signal_kHz == 3:
       
      max_mag_peak = max(max(abs(peaks3_min)),max(peaks3_max))
      upper_bound = Level_of_Crossing*0.01*max_mag_peak  # %Level
      lower_bound = -upper_bound


      peaks_level_max = []

      for peaks_level in df_peak3['y']: 
          if peaks_level > upper_bound or peaks_level < lower_bound:
              peaks_level_max.append(peaks_level)
          else:
              pass

      peaks_level_max_point = df_peak3.loc[df_peak3['y'].isin(peaks_level_max)]
   
      peaks_level_max_point.plot.scatter(x = "x", y = "y", label = 'peaks', color = 'orange')
   
      plt.title('Cross Correlation')
      plt.ylabel('Magnitude (Arbitary Units)')
      plt.xlabel('Time delay (ms)')
      plt.xlim([0, 2])
      plt.ylim([-4, 4])
      plt.plot(data_time, mag_CCsignal_data3, label='3kHz')
      plt.legend()
      plt.show()

   elif Input_Signal_kHz == 4:
    
      max_mag_peak = max(max(abs(peaks4_min)),max(peaks4_max))
      upper_bound = Level_of_Crossing*0.01*max_mag_peak  # %Level
      lower_bound = -upper_bound


      peaks_level_max = []

      for peaks_level in df_peak4['y']: 
          if peaks_level > upper_bound or peaks_level < lower_bound:
              peaks_level_max.append(peaks_level)
          else:
              pass

      peaks_level_max_point = df_peak4.loc[df_peak4['y'].isin(peaks_level_max)]
   
      peaks_level_max_point.plot.scatter(x = "x", y = "y", label = 'peaks', color = 'orange')
   
      plt.title('Cross Correlation')
      plt.ylabel('Magnitude (Arbitary Units)')
      plt.xlabel('Time delay (ms)')
      plt.xlim([0, 2])
      plt.plot(data_time, mag_CCsignal_data4, label='4kHz')
      plt.legend()
      plt.show()
   

   elif Input_Signal_kHz == 5:
    
      max_mag_peak = max(max(abs(peaks5_min)),max(peaks5_max))
      upper_bound = Level_of_Crossing*0.01*max_mag_peak  # %Level
      lower_bound = -upper_bound


      peaks_level_max = []

      for peaks_level in df_peak5['y']: 
          if peaks_level > upper_bound or peaks_level < lower_bound:
              peaks_level_max.append(peaks_level)
          else:
              pass

      peaks_level_max_point = df_peak5.loc[df_peak5['y'].isin(peaks_level_max)]
   
      peaks_level_max_point.plot.scatter(x = "x", y = "y", label = 'peaks', color = 'orange')
   
      plt.title('Cross Correlation')
      plt.ylabel('Magnitude (Arbitary Units)')
      plt.xlabel('Time delay (ms)')
      plt.xlim([0, 2])
      plt.plot(data_time, mag_CCsignal_data5, label='5kHz')
      plt.legend()
      plt.show()
   
   elif Input_Signal_kHz == 6:
    
      max_mag_peak = max(max(abs(peaks6_min)),max(peaks6_max))
      upper_bound = Level_of_Crossing*0.01*max_mag_peak  # %Level
      lower_bound = -upper_bound


      peaks_level_max = []

      for peaks_level in df_peak6['y']: 
          if peaks_level > upper_bound or peaks_level < lower_bound:
              peaks_level_max.append(peaks_level)
          else:
              pass

      peaks_level_max_point = df_peak6.loc[df_peak6['y'].isin(peaks_level_max)]
   
      peaks_level_max_point.plot.scatter(x = "x", y = "y", label = 'peaks', color = 'orange')
   
      plt.title('Cross Correlation')
      plt.ylabel('Magnitude (Arbitary Units)')
      plt.xlabel('Time delay (ms)')
      plt.xlim([0, 2])
      plt.ylim([-5, 5])
      plt.plot(data_time, mag_CCsignal_data6, label='6kHz')
      plt.legend()
      plt.show()
   
   elif Input_Signal_kHz == 7:
    
      max_mag_peak = max(max(abs(peaks7_min)),max(peaks7_max))
      upper_bound = Level_of_Crossing*0.01*max_mag_peak  # %Level
      lower_bound = -upper_bound


      peaks_level_max = []

      for peaks_level in df_peak7['y']: 
          if peaks_level > upper_bound or peaks_level < lower_bound:
              peaks_level_max.append(peaks_level)
          else:
              pass

      peaks_level_max_point = df_peak7.loc[df_peak7['y'].isin(peaks_level_max)]
   
      peaks_level_max_point.plot.scatter(x = "x", y = "y", label = 'peaks', color = 'orange')
   
      plt.title('Cross Correlation')
      plt.ylabel('Magnitude (Arbitary Units)')
      plt.xlabel('Time delay (ms)')
      plt.xlim([0, 2])
      plt.ylim([-5, 4])
      plt.plot(data_time, mag_CCsignal_data7, label='7kHz')
      plt.legend()
      plt.show()
   
   