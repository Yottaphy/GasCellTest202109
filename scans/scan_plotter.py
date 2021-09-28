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
    return n0*(1/sqrt(2*PI))*(1/abs(sigma))*np.exp(-(x-mu)**2/(2*sigma**2))
def lorentz(x, n0, mu, gamma):
    return n0*(PI*gamma*(1+((x-mu)/gamma)**2))**(-1)
def voigta(x, n, mu, sigma, gamma):
    fg = 2*sigma*sqrt(2*log(2))
    fl = 2*gamma
    f = (fg**5 + 2.69269*fl*(fg**4) + 2.42843*(fg**3)*(fl**2) + 4.47163*(fg**2)*(fl**3) + 0.07842*fg*(fl**4) + fl**5)**(1/5)
    eta = 1.36603*(fl/f) - 0.47719*(fl/f)**2 + 0.11116*(fl/f)**3
    return eta*lorentz(x, n, mu, f) + (1-eta)*gaus(x, n, mu, f)
def voigt(x, n, mu, sigma, gamma):
    return n*np.real(wofz((x-mu+1j*gamma)/sigma/sqrt(2)))/abs(sigma)/sqrt(2*PI)


#read the file and output the correct count array and plot limits    
def txtreader(filename):
    x,y,z = np.genfromtxt( filename, unpack= True, delimiter=',', skip_header=1)
    return x,y


#read 3 columns in order
in1       = sys.argv[1]
finalname = sys.argv[2]
stepno    = int(sys.argv[3])
waveno1, counts1 = txtreader(in1)
# waveno2, counts2 = txtreader(in2)

#define some nicer format for the plots
plt.rcParams['font.size'] = 18


#FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.

#plot the experimental data (from the text file) as dots with error bars. 
minimum = waveno1.min()
maximum = waveno1.max()

if stepno == 2:
    midpoint = 10991.185*2
    for i in range(len(waveno1)):
        waveno1[i] *= 2
        waveno1[i] -= midpoint
elif stepno == 1:
    midpoint = 13085.7509258*3
    for i in range(len(waveno1)):
        waveno1[i] *= 3
        waveno1[i] -= midpoint


# for i in range(len(waveno2)):
#     waveno2[i] -= midpoint

plt.plot(waveno1, counts1, label=finalname.strip('_fit').replace('_1step', ' 1$^{st}$ step').replace('_2step', ' 2$^{nd}$ step'))

#if stepno == 2:
#    plt.savefig("2step/results/"+finalname+".pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
#    plt.savefig("2step/results/"+finalname+".png", bbox_inches = 'tight', pad_inches = 0.1)
#else:
#    plt.savefig("1step/results/"+finalname+".pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
#    plt.savefig("1step/results/"+finalname+".png", bbox_inches = 'tight', pad_inches = 0.1)


gfit, gerr = curve_fit(gaus, waveno1, counts1, p0 = [600, 0.05, 1])
lfit, lerr = curve_fit(lorentz, waveno1, counts1, p0 = [600, 0.05, 1])
vfit, verr = curve_fit(voigt, waveno1, counts1, p0 = [500, 0.05, 1, 1])
# afit, aerr = curve_fit(voigta, waveno1, counts1, p0 = [500, 0.05, 0.01, 0.1])

# print(waveno1)
#more format for the plots
#plt.yscale('log')
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
plt.title('Reference = '+str(round(midpoint,2)))
plt.grid(which='major', axis='both', linewidth=1)
plt.ylabel("Counts")
plt.xlabel("Wavenumber from reference [cm$^{-1}$]")
# plt.text(x= minimum, y= 1.1*counts2.max(), s = "Ref="+str(midpoint))
plt.legend(loc=2,fontsize= 'x-small')
# plt.yscale("log")

c = 29.9792458 #cm/ns
mean = (vfit[1] + midpoint)*c
meanerr = c*verr[1][1]
fwhm = (fv)*c
fwhmerr = errf*c

if stepno==2:
    f = open("2step/stats2.txt", "a")
elif stepno == 1:
    f = open("1step/stats1.txt", "a")
    
f.write(finalname[0]+finalname[1]+finalname[2] + '\t\t\t\t\t\t' + str(vfit[0]) + '\t' + str(sqrt(verr[0][0])) + '\t' + str(mean) + '\t' + str(meanerr) + '\t' + str(fwhm) + '\t' + str(fwhmerr) + '\t' + str(vfit[2])+ '\t' + str(verr[2][2]) + '\t' + str(vfit[3])+ '\t' + str(verr[3][3]) + '\n')
#save plot as a pdf (vector images are superior, change my mind) with transparency
if stepno == 2:
    plt.savefig("2step/results/"+finalname+".pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("2step/results/"+finalname+".png", bbox_inches = 'tight', pad_inches = 0.1)
elif stepno == 1:
    plt.savefig("1step/results/"+finalname+".pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("1step/results/"+finalname+".png", bbox_inches = 'tight', pad_inches = 0.1)

plt.close()
f.close()
