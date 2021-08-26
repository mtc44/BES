# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 20:30:05 2021

@author: chanchanchan
"""

#Transfer Function:
  
Input_Signal_kHz = 4
  
    
from matplotlib import pyplot as plt
import numpy as np
import DissertationPlotwithDataMain as main 
import FastFouriorTransform as faft

#Transfer Function
#3
mag3_trans = [0] + (main.mag_fft_data3_output/main.mag_fft_data3_input).tolist() #set the first value of the series = 0 
del mag3_trans[1]
#Transfer Function Mag

stacked_phase3_trans = faft.stacked_phase3out - faft.stacked_phase3in #Transfer Function phase 

#4
mag4_trans = [0] + (faft.mag_fft_data4_output/faft.mag_fft_data4_input).tolist() #set the first value of the series = 0 
del mag4_trans[1]
#Transfer Function Mag

stacked_phase4_trans = faft.stacked_phase4out - faft.stacked_phase4in #Transfer Function phase 

#5
mag5_trans = [0] + (faft.mag_fft_data5_output/faft.mag_fft_data5_input).tolist() #set the first value of the series = 0 
del mag5_trans[1]
#Transfer Function Mag

stacked_phase5_trans = faft.stacked_phase5out - faft.stacked_phase5in #Transfer Function phase 

#6
mag6_trans = [0] + (faft.mag_fft_data6_output/faft.mag_fft_data6_input).tolist() #set the first value of the series = 0 
del mag6_trans[1]
#Transfer Function Mag

stacked_phase6_trans = faft.stacked_phase6out - faft.stacked_phase6in #Transfer Function phase 

#7
mag7_trans = [0] + (faft.mag_fft_data7_output/faft.mag_fft_data7_input).tolist() #set the first value of the series = 0 
del mag7_trans[1]
#Transfer Function Mag

stacked_phase7_trans = faft.stacked_phase7out - faft.stacked_phase7in #Transfer Function phase 



#Line Regression 

from scipy.stats import linregress

#3
#input
slope3in = []

for x in np.arange(1, 60, 1):
    x_1 = main.stacked_phase3in[x-1] 
    x_2 = main.stacked_phase3in[x] 
    x_3 = main.stacked_phase3in[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope3in.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope3_input = np.array(slope3in)/360*1000

#output
slope3out = []

for x in np.arange(1, 60, 1):
    x_1 = main.stacked_phase3out[x-1] 
    x_2 = main.stacked_phase3out[x] 
    x_3 = main.stacked_phase3out[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope3out.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope3_output = np.array(slope3out)/360*1000

#transfer 

pretrigger = 110
slope3trans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase3_trans[x-1] 
    x_2 = stacked_phase3_trans[x] 
    x_3 = stacked_phase3_trans[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope3trans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope3_trans = np.array(slope3trans)/360*1000 - pretrigger


#4
#input
slope4in = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase4in[x-1] 
    x_2 = faft.stacked_phase4in[x] 
    x_3 = faft.stacked_phase4in[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope4in.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope4_input = np.array(slope4in)/360*1000

#output
slope4out = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase4out[x-1] 
    x_2 = faft.stacked_phase4out[x] 
    x_3 = faft.stacked_phase4out[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope4out.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope4_output = np.array(slope4out)/360*1000

#transfer 

pretrigger = 110
slope4trans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase4_trans[x-1] 
    x_2 = stacked_phase4_trans[x] 
    x_3 = stacked_phase4_trans[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope4trans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope4_trans = np.array(slope4trans)/360*1000 - pretrigger 


#5
#input
slope5in = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase5in[x-1] 
    x_2 = faft.stacked_phase5in[x] 
    x_3 = faft.stacked_phase5in[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope5in.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope5_input = np.array(slope5in)/360*1000

#output
slope5out = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase5out[x-1] 
    x_2 = faft.stacked_phase5out[x] 
    x_3 = faft.stacked_phase5out[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope5out.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope5_output = np.array(slope5out)/360*1000

#transfer 

pretrigger = 110
slope5trans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase5_trans[x-1] 
    x_2 = stacked_phase5_trans[x] 
    x_3 = stacked_phase5_trans[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope5trans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope5_trans = np.array(slope5trans)/360*1000 - pretrigger



#5
#input
slope5in = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase5in[x-1] 
    x_2 = faft.stacked_phase5in[x] 
    x_3 = faft.stacked_phase5in[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope5in.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope5_input = np.array(slope5in)/360*1000

#output
slope5out = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase5out[x-1] 
    x_2 = faft.stacked_phase5out[x] 
    x_3 = faft.stacked_phase5out[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope5out.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope5_output = np.array(slope5out)/360*1000

#transfer 

pretrigger = 110
slope5trans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase5_trans[x-1] 
    x_2 = stacked_phase5_trans[x] 
    x_3 = stacked_phase5_trans[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope5trans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope5_trans = np.array(slope5trans)/360*1000 - pretrigger



#6
#input
slope6in = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase6in[x-1] 
    x_2 = faft.stacked_phase6in[x] 
    x_3 = faft.stacked_phase6in[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope6in.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope6_input = np.array(slope6in)/360*1000

#output
slope6out = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase6out[x-1] 
    x_2 = faft.stacked_phase6out[x] 
    x_3 = faft.stacked_phase6out[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope6out.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope6_output = np.array(slope6out)/360*1000

#transfer 

pretrigger = 110
slope6trans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase6_trans[x-1] 
    x_2 = stacked_phase6_trans[x] 
    x_3 = stacked_phase6_trans[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope6trans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope6_trans = np.array(slope6trans)/360*1000 - pretrigger


#7
#input
slope7in = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase7in[x-1] 
    x_2 = faft.stacked_phase7in[x] 
    x_3 = faft.stacked_phase7in[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope7in.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope7_input = np.array(slope7in)/360*1000

#output
slope7out = []

for x in np.arange(1, 60, 1):
    x_1 = faft.stacked_phase7out[x-1] 
    x_2 = faft.stacked_phase7out[x] 
    x_3 = faft.stacked_phase7out[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope7out.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope7_output = np.array(slope7out)/360*1000

#transfer 

pretrigger = 110
slope7trans = []

for x in np.arange(1, 60, 1):
    x_1 = stacked_phase7_trans[x-1] 
    x_2 = stacked_phase7_trans[x] 
    x_3 = stacked_phase7_trans[x+1] 
        
    x_4 = main.change_in_frequency[x-1]
    x_5 = main.change_in_frequency[x]
    x_6 = main.change_in_frequency[x+1]
        
    slope7trans.append(linregress([x_4,x_5,x_6], [x_1,x_2,x_3])[0])
        
slope7_trans = np.array(slope7trans)/360*1000 - pretrigger



#3 D/lambda

slope3_x = main.change_in_frequency[7:12]
slope3_y = stacked_phase3_trans[7:12]

line_regress3 = linregress(slope3_x, slope3_y)
Tarr3 = line_regress3[0]/360*1000 - pretrigger

D_lambda3 = []

for value in main.change_in_frequency:
    D_lambda3.append(value*Tarr3/1000)
    
Tarr3_list = []

for value in np.arange(0,5,1):
    
    Tarr3_list.append((value+1-value)*Tarr3)
    
    

#4 D/lambda

slope4_x = main.change_in_frequency[7:12]
slope4_y = stacked_phase4_trans[7:12]

line_regress4 = linregress(slope4_x, slope4_y)
Tarr4 = line_regress4[0]/360*1000 - pretrigger

D_lambda4 = []

for value in main.change_in_frequency:
    D_lambda4.append(value*Tarr4/1000)
    
Tarr4_list = []

for value in np.arange(0,5,1):
    
    Tarr4_list.append((value+1-value)*Tarr4)
    
    
#5 D/lambda

slope5_x = main.change_in_frequency[7:12]
slope5_y = stacked_phase5_trans[7:12]

line_regress5 = linregress(slope5_x, slope5_y)
Tarr5 = line_regress5[0]/360*1000 - pretrigger

D_lambda5 = []

for value in main.change_in_frequency:
    D_lambda5.append(value*Tarr5/1000)

Tarr5_list = []

for value in np.arange(0,5,1):
    
    Tarr5_list.append((value+1-value)*Tarr5)
    
    

#6 D/lambda

slope6_x = main.change_in_frequency[7:12]
slope6_y = stacked_phase6_trans[7:12]

line_regress6 = linregress(slope6_x, slope6_y)
Tarr6 = line_regress6[0]/360*1000 - pretrigger

D_lambda6 = []

for value in main.change_in_frequency:
    D_lambda6.append(value*Tarr6/1000)

Tarr6_list = []

for value in np.arange(0,5,1):
    
    Tarr6_list.append((value+1-value)*Tarr6)
    
    
    
#7 D/lambda

slope7_x = main.change_in_frequency[7:12]
slope7_y = stacked_phase7_trans[7:12]

line_regress7 = linregress(slope7_x, slope7_y)
Tarr7 = line_regress7[0]/360*1000 - pretrigger

D_lambda7 = []

for value in main.change_in_frequency:
    D_lambda7.append(value*Tarr7/1000)

Tarr7_list = []

for value in np.arange(0,5,1):
    
    Tarr7_list.append((value+1-value)*Tarr7)




if __name__ == "__main__" :    
    
    if Input_Signal_kHz == 3:
    
#Slope Transfer
       plt.title('Slope Transfer vs Frequency')
       plt.ylabel('Arrival time (ms)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[1:20], slope3_trans[0:19], label = 'Transfer')
       plt.plot(main.change_in_frequency[7:11], Tarr3_list)
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
   
   
#Transfer Phase/Mag  
       fig,ax_phase = plt.subplots()
       ax_phase.plot(main.change_in_frequency[0:20], stacked_phase3_trans[0:20], label = 'Stacked Phase', color = 'red')
       plt.legend(bbox_to_anchor=(1.1, 1.1), loc= 'upper left')
       plt.ylabel('Phase (degrees)', color = 'red')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([-400, 1600])
       plt.title('Transfer Phase/Mag vs Frequency')
   
       ax_mag = ax_phase.twinx()
       ax_mag.plot(main.change_in_frequency[0:20], mag3_trans[0:20], label = 'Magnitude', color = 'blue')
       plt.legend(bbox_to_anchor=(1.1, 1), loc= 'upper left')
       plt.ylabel('Gain (Arbitary Units)', color = 'blue')
       plt.show()
   
#Input/Output Mag
       plt.title('Input/Output Magnitude vs Frequency')
       plt.ylabel('Magnitude (arb.unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 1200])
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data3_input[0:20], label = 'Input')
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data3_output[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Transfer Mag
       plt.title('Transfer Magnitude vs Frequency')
       plt.ylabel('Gain (arbitary unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[0:20], mag3_trans[0:20], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Trans Stacked phase vs D/lambda
       plt.title('Transfer Stacked Phase vs D/lambda')
       plt.ylabel('Phase (degrees)')
       plt.xlabel('D/lambda')
       plt.xlim([0, 5])
       plt.plot(D_lambda3[0:19], stacked_phase3_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()


#Input/Output Phase
       plt.title('Input/Output Phase vs Frequency')
       plt.ylabel('Stacked Phase (degrees)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 3000])
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase3in[0:20], label = 'Input')
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase3out[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       
    if Input_Signal_kHz == 4:
    
#Slope Transfer
       plt.title('Slope Transfer vs Frequency')
       plt.ylabel('Arrival time (ms)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[1:20], slope4_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
   
   
#Transfer Phase/Mag  
       fig,ax_phase = plt.subplots()
       ax_phase.plot(main.change_in_frequency[0:20], stacked_phase4_trans[0:20], label = 'Stacked Phase', color = 'red')
       plt.legend(bbox_to_anchor=(1.1, 1.1), loc= 'upper left')
       plt.ylabel('Phase (degrees)', color = 'red')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([-400, 1600])
       plt.title('Transfer Phase/Mag vs Frequency')
   
       ax_mag = ax_phase.twinx()
       ax_mag.plot(main.change_in_frequency[0:20], mag4_trans[0:20], label = 'Magnitude', color = 'blue')
       plt.legend(bbox_to_anchor=(1.1, 1), loc= 'upper left')
       plt.ylabel('Gain (Arbitary Units)', color = 'blue')
       plt.show()
   
#Input/Output Mag
       plt.title('Input/Output Magnitude vs Frequency')
       plt.ylabel('Magnitude (arb.unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 1200])
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data4_input[0:20], label = 'Input')
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data4_output[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Transfer Mag
       plt.title('Transfer Magnitude vs Frequency')
       plt.ylabel('Gain (arbitary unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[0:20], mag4_trans[0:20], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Trans Stacked phase vs D/lambda
       plt.title('Transfer Stacked Phase vs D/lambda')
       plt.ylabel('Phase (degrees)')
       plt.xlabel('D/lambda')
       plt.xlim([0, 5])
       plt.plot(D_lambda4[0:19], stacked_phase4_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()


#Input/Output Phase
       plt.title('Input/Output Phase vs Frequency')
       plt.ylabel('Stacked Phase (degrees)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 3000])
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase4in[0:20], label = 'Input')
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase4out[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       
    if Input_Signal_kHz == 5:
    
#Slope Transfer
       plt.title('Slope Transfer vs Frequency')
       plt.ylabel('Arrival time (ms)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[1:20], slope5_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
   
   
#Transfer Phase/Mag  
       fig,ax_phase = plt.subplots()
       ax_phase.plot(main.change_in_frequency[0:20], stacked_phase5_trans[0:20], label = 'Stacked Phase', color = 'red')
       plt.legend(bbox_to_anchor=(1.1, 1.1), loc= 'upper left')
       plt.ylabel('Phase (degrees)', color = 'red')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([-400, 1600])
       plt.title('Transfer Phase/Mag vs Frequency')
   
       ax_mag = ax_phase.twinx()
       ax_mag.plot(main.change_in_frequency[0:20], mag5_trans[0:20], label = 'Magnitude', color = 'blue')
       plt.legend(bbox_to_anchor=(1.1, 1), loc= 'upper left')
       plt.ylabel('Gain (Arbitary Units)', color = 'blue')
       plt.show()
   
#Input/Output Mag
       plt.title('Input/Output Magnitude vs Frequency')
       plt.ylabel('Magnitude (arb.unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 1200])
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data5_input[0:20], label = 'Input')
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data5_output[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Transfer Mag
       plt.title('Transfer Magnitude vs Frequency')
       plt.ylabel('Gain (arbitary unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[0:20], mag5_trans[0:20], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Trans Stacked phase vs D/lambda
       plt.title('Transfer Stacked Phase vs D/lambda')
       plt.ylabel('Phase (degrees)')
       plt.xlabel('D/lambda')
       plt.xlim([0, 5])
       plt.plot(D_lambda5[0:19], stacked_phase5_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()


#Input/Output Phase
       plt.title('Input/Output Phase vs Frequency')
       plt.ylabel('Stacked Phase (degrees)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 3000])
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase5in[0:20], label = 'Input')
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase5out[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       
    if Input_Signal_kHz == 6:
    
#Slope Transfer
       plt.title('Slope Transfer vs Frequency')
       plt.ylabel('Arrival time (ms)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[1:20], slope6_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
   
   
#Transfer Phase/Mag  
       fig,ax_phase = plt.subplots()
       ax_phase.plot(main.change_in_frequency[0:20], stacked_phase6_trans[0:20], label = 'Stacked Phase', color = 'red')
       plt.legend(bbox_to_anchor=(1.1, 1.1), loc= 'upper left')
       plt.ylabel('Phase (degrees)', color = 'red')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([-400, 1600])
       plt.title('Transfer Phase/Mag vs Frequency')
   
       ax_mag = ax_phase.twinx()
       ax_mag.plot(main.change_in_frequency[0:20], mag6_trans[0:20], label = 'Magnitude', color = 'blue')
       plt.legend(bbox_to_anchor=(1.1, 1), loc= 'upper left')
       plt.ylabel('Gain (Arbitary Units)', color = 'blue')
       plt.show()
   
#Input/Output Mag
       plt.title('Input/Output Magnitude vs Frequency')
       plt.ylabel('Magnitude (arb.unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 1200])
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data6_input[0:20], label = 'Input')
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data6_output[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Transfer Mag
       plt.title('Transfer Magnitude vs Frequency')
       plt.ylabel('Gain (arbitary unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[0:20], mag6_trans[0:20], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Trans Stacked phase vs D/lambda
       plt.title('Transfer Stacked Phase vs D/lambda')
       plt.ylabel('Phase (degrees)')
       plt.xlabel('D/lambda')
       plt.xlim([0, 5])
       plt.ylim([0,2000])
       plt.plot(D_lambda6[0:19], stacked_phase6_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()


#Input/Output Phase
       plt.title('Input/Output Phase vs Frequency')
       plt.ylabel('Stacked Phase (degrees)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 3000])
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase6in[0:20], label = 'Input')
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase6out[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       

    if Input_Signal_kHz == 7:
    
#Slope Transfer
       plt.title('Slope Transfer vs Frequency')
       plt.ylabel('Arrival time (ms)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[1:20], slope7_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
   
   
#Transfer Phase/Mag  
       fig,ax_phase = plt.subplots()
       ax_phase.plot(main.change_in_frequency[0:20], stacked_phase7_trans[0:20], label = 'Stacked Phase', color = 'red')
       plt.legend(bbox_to_anchor=(1.1, 1.1), loc= 'upper left')
       plt.ylabel('Phase (degrees)', color = 'red')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([-400, 1600])
       plt.title('Transfer Phase/Mag vs Frequency')
   
       ax_mag = ax_phase.twinx()
       ax_mag.plot(main.change_in_frequency[0:20], mag7_trans[0:20], label = 'Magnitude', color = 'blue')
       plt.legend(bbox_to_anchor=(1.1, 1), loc= 'upper left')
       plt.ylabel('Gain (Arbitary Units)', color = 'blue')
       plt.show()
   
#Input/Output Mag
       plt.title('Input/Output Magnitude vs Frequency')
       plt.ylabel('Magnitude (arb.unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 1200])
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data7_input[0:20], label = 'Input')
       plt.plot(main.change_in_frequency[0:20], faft.mag_fft_data7_output[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Transfer Mag
       plt.title('Transfer Magnitude vs Frequency')
       plt.ylabel('Gain (arbitary unit)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.plot(main.change_in_frequency[0:20], mag7_trans[0:20], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()

#Trans Stacked phase vs D/lambda
       plt.title('Transfer Stacked Phase vs D/lambda')
       plt.ylabel('Phase (degrees)')
       plt.xlabel('D/lambda')
       plt.xlim([0, 5])
       plt.ylim([-500, 2000])
       plt.plot(D_lambda7[0:19], stacked_phase7_trans[0:19], label = 'Transfer')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()


#Input/Output Phase
       plt.title('Input/Output Phase vs Frequency')
       plt.ylabel('Stacked Phase (degrees)')
       plt.xlabel('Frequency (kHz)')
       plt.xlim([0, 10])
       plt.ylim([0, 3000])
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase7in[0:20], label = 'Input')
       plt.scatter(main.change_in_frequency[0:20], faft.stacked_phase7out[0:20], label = 'Output')
       plt.legend(bbox_to_anchor=(1, 1), loc='upper left' )
       plt.show()
       
       