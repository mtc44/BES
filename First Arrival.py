# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:56:06 2021

@author: chanchanchan
"""

#First Arrival:
    
#Zoom-in function
Time_1 = 1 #unit: ms
Time_2 = 2 #unit: ms

Amplitude_1 = -3   #unit:V
Amplitude_2 = 5  #unit:V

#Boundary for first arrival peak 
Boundary_1 = 0.50 #unit: ms
Boundary_2 = 0.75 #unit: ms



from matplotlib import pyplot as plt
import DissertationPlotwithDataMain as main 

#Input Signal
plt.plot(main.data3_time_new, main.data3_input_new, label = '3kHz')
plt.plot(main.data3_time_new, main.data4_input_new, label = '4kHz')
plt.plot(main.data3_time_new, main.data5_input_new, label = '5kHz')
plt.plot(main.data3_time_new, main.data6_input_new, label = '6kHz')
plt.plot(main.data3_time_new, main.data7_input_new, label = '7kHz')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
plt.title('Input Signal')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (ms)')
plt.xlim([0, 1.0])
plt.show()


#Output Signal

#overview
plt.plot(main.data3_time_new, main.data3_output_new, label = '3kHz')
plt.plot(main.data3_time_new, main.data4_output_new, label = '4kHz')
plt.plot(main.data3_time_new, main.data5_output_new, label = '5kHz')
plt.plot(main.data3_time_new, main.data6_output_new, label = '6kHz')
plt.plot(main.data3_time_new, main.data7_output_new, label = '7kHz')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
plt.title('Output Signal (overview)')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (ms)')
plt.xlim([0,2])
plt.show()

#zoom in image 
plt.plot(main.data3_time_new, main.data3_output_new, label = '3kHz')
plt.plot(main.data3_time_new, main.data4_output_new, label = '4kHz')
plt.plot(main.data3_time_new, main.data5_output_new, label = '5kHz')
plt.plot(main.data3_time_new, main.data6_output_new, label = '6kHz')
plt.plot(main.data3_time_new, main.data7_output_new, label = '7kHz')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
plt.title('Output Signal (zoom in)')
plt.ylabel('Amplitude (V)')
plt.xlabel('Time (ms)')
plt.xlim([Time_1,Time_2]) #change to zoom in with specific time
plt.ylim([Amplitude_1,Amplitude_2])
plt.axvline(x=Boundary_1, ymin=0.05, ymax=0.95, color = 'black') # boundary for specific time period 
plt.axvline(x=Boundary_2, ymin=0.05, ymax=0.95, color = 'black')
plt.show()