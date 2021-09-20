import numpy as np
from numpy.lib.scimath import log, sqrt
import scipy as sc
from scipy.special import wofz
import sys
import csv
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PI = 3.14159265359

#functions defined for exponential decay (expo) and double exponential (expo2)
def gaus(x, n0, mu, sigma):
    return n0*(1/sqrt(2*PI))*(1/sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
def lorentz(x, n0, mu, gamma):
    return n0*(PI*gamma*(1+((x-mu)/gamma)**2))**(-1)
def voigta(x, n, mu, sigma, gamma):
    fg = 2*sigma*sqrt(2*log(2))
    fl = 2*gamma
    f = (fg**5 + 2.69269*fl*(fg**4) + 2.42843*(fg**3)*(fl**2) + 4.47163*(fg**2)*(fl**3) + 0.07842*fg*(fl**4) + fl**5)**(1/5)
    eta = 1.36603*(fl/f) - 0.47719*(fl/f)**2 + 0.11116*(fl/f)**3
    return eta*lorentz(x, n, mu, f) + (1-eta)*gaus(x, n, mu, f)
def voigt(x, n, mu, sigma, gamma):
    return n*np.real(wofz((x-mu+1j*gamma)/sigma/sqrt(2)))/sigma/sqrt(2*PI)


#read the file and output the correct count array and plot limits    
def txtreader(filename):
    x,y,z = np.genfromtxt( filename, unpack= True, delimiter=',', skip_header=1)
    return x,y


#read 3 columns in order
in1 = sys.argv[1]
# in2 = sys.argv[2]
finalname = sys.argv[2]
waveno1, counts1 = txtreader(in1)
# waveno2, counts2 = txtreader(in2)

#make a smooth array of times for the plot
# smooth = np.arange(0.5*min(time),1.1*max(time))

#define some nicer format for the plots
plt.rcParams['font.size'] = 18


#FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.

#plot the experimental data (from the text file) as dots with error bars. 
minimum = waveno1.min()
maximum = waveno1.max()

midpoint = 13085.683

for i in range(len(waveno1)):
    waveno1[i] -= midpoint

# for i in range(len(waveno2)):
#     waveno2[i] -= midpoint


gfit, gerr = curve_fit(gaus, waveno1, counts1, p0 = [600, 0.05, 0.1])
lfit, lerr = curve_fit(lorentz, waveno1, counts1, p0 = [600, 0.05, 0.1])
vfit, verr = curve_fit(voigt, waveno1, counts1, p0 = [500, 0.05, 0.01, 0.1])
# afit, aerr = curve_fit(voigta, waveno1, counts1, p0 = [500, 0.05, 0.01, 0.1])

# print(waveno1)
#more format for the plots
#plt.yscale('log')
plt.plot(waveno1, counts1, label=in1.strip('gascell.csv').replace('_',' ').replace('1step', '1$^{st}$ step').replace('2step', '2^{nd} step'))
plt.plot(waveno1, gaus(waveno1, *gfit), '--', color= 'green', label = 'Gauss')
plt.plot(waveno1, lorentz(waveno1, *lfit), '--', color= 'red', label = 'Lorentz')

varname = 'Nμσγ'
for i in range(len(vfit)):
    print(varname[i] + '= ' + str(round(vfit[i],4)) + ' ± ' + str(round(sqrt(verr[i][i]),4)))

fg = 2*sqrt(2*log(2))*vfit[2]
errg = 2*sqrt(2*log(2))*verr[2][2]
fl = 2*vfit[3]
errl = 2*verr[3][3]
sq = sqrt(0.2166*fl**2 + fg**2)
fv = 0.5346*fl + sq
errf = sqrt((errg*fg/sq)**2 + (errl*(0.5346+(0.2166*fl/sq)))**2)
print('f= ' + str(round(fv,6)) + ' ± ' + str(round(errf,6)))

plt.plot(waveno1, voigt(waveno1, *vfit), '-', color= 'black', label = 'Voigt')
# plt.plot(waveno1, voigta(waveno1, *afit), '--', color= 'orange', label = 'Voigt Approx')

#plt.plot(waveno2, counts2, label=in2.strip('gascell.csv').replace('_',' ').replace('1step', '1$^{st}$ step').replace('2step', '2^{nd} step'))
plt.title('Reference = '+str(midpoint))
plt.grid(which='major', axis='both', linewidth=1)
plt.ylabel("Counts")
plt.xlabel("Wavenumber from reference [cm$^{-1}$]")
# plt.text(x= minimum, y= 1.1*counts2.max(), s = "Ref="+str(midpoint))
plt.legend(loc=2,fontsize= 'x-small')
# plt.yscale("log")

c = 29.9792458 #cm/ns
mean = (vfit[1] + midpoint)*c
meanerr = c*verr[1][1]
fwhm = (fv+midpoint)*c
fwhmerr = errf*c

f = open("results/stats.txt", "a")
f.write(in1[0]+in1[1]+in1[2] + '\t\t\t\t\t\t' + str(vfit[0]) + '\t' + str(sqrt(verr[0][0])) + '\t' + str(mean) + '\t' + str(meanerr) + '\t' + str(fwhm) + '\t' + str(fwhmerr) + '\t' + str(vfit[2])+ '\t' + str(verr[2][2]) + '\t' + str(vfit[3])+ '\t' + str(verr[3][3]) + '\n')
#save plot as a pdf (vector images are superior, change my mind) with transparency
plt.savefig("results/"+finalname+".pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
plt.savefig("results/"+finalname+".png", bbox_inches = 'tight', pad_inches = 0.1)
plt.close()
f.close()