from statistics import mode
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
from lmfit import Model
from lmfit.models import SkewedVoigtModel

PI = 3.14159265359
c = 29.9792458 #cm/ns

#read the file and output the correct count array and plot limits    
def txtreader(filename):
    x,y,z = np.genfromtxt( filename, unpack= True, delimiter=',', skip_header=1)
    return x,y
def txtreader2(filename):
    a,cts,b,c,wvno,d,e,f,g,h,i,j= np.genfromtxt( filename, unpack= True, delimiter=',', skip_header=1)
    return wvno, cts


#read 3 columns in order
in1       = sys.argv[1]
finalname = sys.argv[2]
stepno    = sys.argv[3]
# waveno, counts = txtreader(in1)
waveno, counts = txtreader2(in1)

#define some nicer format for the plots
plt.rcParams['font.size'] = 18


#FITTING FUNCTION: fit with expo, xvalues=timefit, yvalues=intenfit, error=errfit, p0 is the initial values for N_0 and tau, respectively.

#plot the experimental data (from the text file) as dots with error bars. 
minimum = waveno.min()
maximum = waveno.max()
freq =[]

if stepno == '2' or stepno == 'feb':
    midpoint = c*1E7/454.91
    for i in range(len(waveno)):
        freq.append(2*c*waveno[i])
        freq[i] -= midpoint
elif stepno == '1':
    midpoint = c*1E7/254.73
    for i in range(len(waveno)):
        freq.append(3*c*waveno[i])
        freq[i] -= midpoint

plt.plot(freq, counts, 'o', markersize=2,  label=finalname.strip('_fit').replace('_1step', ' 1$^{st}$ step').replace('_2step', ' 2$^{nd}$ step'))

svm = SkewedVoigtModel()
params = svm.make_params(amplitude=600, center=-10, sigma=150, gamma=130, skew=5)

result = svm.fit(counts, params, x= freq)

plt.plot(freq, result.best_fit, '-', color = 'red',  label="Asymmetric Voigt fit")

plt.grid(which='major', axis='both', linewidth=1)
plt.ylabel('Counts')
plt.xlabel('Frequency - '+str(round(midpoint))+' [GHz]')
# plt.text(x= minimum, y= 1.1*counts2.max(), s = "Ref="+str(midpoint))
plt.legend(loc=2,fontsize= 'x-small')
# plt.yscale("log")

if stepno=='2':
    f = open("2step/stats2_asym.txt", "a")
elif stepno == '1':
    f = open("1step/stats1_asym.txt", "a")
elif stepno == 'feb':
    f = open("feb/statsfeb_asym.txt", "a")

g    = result.params['gamma'].value
gerr = result.params['gamma'].stderr

s    = result.params['sigma'].value
serr = result.params['sigma'].stderr

F1   = 1.0692
F2   = 0.8664
F3   = 5.545083
fwhm = F1*g + sqrt(F2*g**2 + F3*s**2)
fwhmerr = sqrt((gerr*(F1+F2*g/(fwhm-F1*g)))**2 + (serr*s*F3/(fwhm-F1*g))**2)
    
mu = result.params['center'].value
muerr = result.params['center'].stderr
skew = result.params['skew'].value
skewerr = result.params['skew'].stderr

centre = freq[result.best_fit.argmax()] 
centreerr = muerr

# f.write(finalname[0]+finalname[1]+finalname[2] + '\t\t\t\t\t\t' + str(result) + '\t' + str(sqrt(verr[0][0])) + '\t' + str(mean) + '\t' + str(meanerr) + '\t' + str(fwhm) + '\t' + str(fwhmerr) + '\t' + str(vfit[2])+ '\t' + str(verr[2][2]) + '\t' + str(vfit[3])+ '\t' + str(verr[3][3]) + '\n')
f.write(finalname[0]+finalname[1]+finalname[2] + '\t\t\t\t\t\t' + str(centre) +'\t'+ str(centreerr) +'\t'+ str(fwhm)+'\t'+ str(fwhmerr)+'\n')


#plt.vlines(centre,0,500)
#plt.axvspan(freq[result.best_fit.argmax() -1], freq[result.best_fit.argmax()+2], color='red')


#save plot as a pdf (vector images are superior, change my mind) with transparency
if stepno == 'feb':
    plt.savefig("feb/results/"+finalname+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("feb/results/"+finalname+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)
if stepno == '2':
    plt.savefig("2step/results/"+finalname+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("2step/results/"+finalname+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)
elif stepno == '1':
    plt.savefig("1step/results/"+finalname+"_asym.pdf", bbox_inches = 'tight', pad_inches = 0.1, transparent=True)
    plt.savefig("1step/results/"+finalname+"_asym.png", bbox_inches = 'tight', pad_inches = 0.1)

plt.close()
f.close()
