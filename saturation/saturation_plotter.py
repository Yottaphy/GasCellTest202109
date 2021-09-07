import numpy as np
import scipy as sc
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#functions defined for exponential decay (expo) and double exponential (expo2)
def expo(t, N_0, tau):
    return N_0*(np.exp(-t/tau))
def expo2(t, N_0, N_1, N_2, tau1, tau2):
    return N_0*expo(t, N_1, tau1)*expo(t, N_2, t-tau2)


#read the file and output the correct count array and plot limits    
def txtreader(filename):
    x,y,z = np.genfromtxt( filename, unpack=True )
    return x,y,z


#read 3 columns in order
power1, counts1, error1 = txtreader("1step.txt")
power2, counts2, error2 = txtreader("2step.txt")

#make a smooth array of times for the plot
# smooth = np.arange(0.5*min(time),1.1*max(time))

#define some nicer format for the plots
k='C0'
plt.rcParams['font.size'] = 18


#FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.

#plot the experimental data (from the text file) as dots with error bars. 
plt.errorbar(power1, counts1, yerr=error1, fmt='o-', color='C0', label='1st Step')


#more format for the plots
#plt.yscale('log')
plt.grid(which='major', axis='both', linestyle='--', linewidth=1)
#plt.ylim(0, 1.1)
plt.ylabel("Counts")
plt.xlabel("Laser Power [mW]")
plt.legend(loc = 'upper left')
# plt.yscale("log")


#save plot as a pdf (vector images are superior, change my mind) with transparency
plt.savefig("SaturationCurve1.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
plt.savefig("SaturationCurve1.png", bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
plt.errorbar(power2, counts2, yerr=error2, fmt='o-', color='C1', label='2nd Step')

#more format for the plots
#plt.yscale('log')
plt.grid(which='major', axis='both', linestyle='--', linewidth=1)
#plt.ylim(0, 1.1)
plt.ylabel("Counts")
plt.xlabel("Laser Power [mW]")
plt.legend(loc = 'upper left')
# plt.yscale("log")
plt.savefig("SaturationCurve2.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
plt.savefig("SaturationCurve2.png", bbox_inches = 'tight', pad_inches = 0.1)
