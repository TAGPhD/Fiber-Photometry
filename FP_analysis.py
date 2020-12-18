# -*- coding: utf-8 -*-
"""
FP_analysis.py
Converting Matlab analysis program for Fiber Photometry into Python. 
Based on the analysis described in Martianova et al. 2019.
First attempt.
"""
# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from sklearn import linear_model

###############################################################################
### defining some important functions
def read_signal(path_name,file_name):
    """
    Reads in the data and returns it as pandas dataframe.
    """
    raw_signal = pd.read_csv(path_name+file_name)
    return raw_signal

def roll_mean(signal_col):
    """
    Takes in a cloumn signal (column of DataFrame) and produces the moving 
    window mean as another DataFrame column.
    
    Window = 21 before, example program from Martianova used 10. 
    """
    col_of_means = signal_col.rolling(window=10,center=True,min_periods=1).mean()
    return col_of_means

""" 
The following two functions (WhittakerSmooth and airPLS) are from a public 
github, see python file airPLS_Python.py (and below) for the legalese and 
further notes.

airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares
This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls
Reference:
Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive 
iteratively reweighted penalized least squares. Analyst 135 (5), 1138-1146 
(2010).
Description from the original documentation:
Baseline drift always blurs or even swamps signals and deteriorates analytical 
results, particularly in multivariate analysis.  It is necessary to correct 
baseline drift to perform further data analysis. Simple or modified polynomial 
fitting has been found to be effective in some extent. However, this method 
requires user intervention and prone to variability especially in low signal-
to-noise ratio environments. The proposed adaptive iteratively reweighted 
Penalized Least Squares (airPLS) algorithm doesn't require any user 
intervention and prior information, such as detected peaks. It iteratively 
changes weights of sum squares errors (SSE) between the fitted baseline and 
original signals, and the weights of SSE are obtained adaptively using between 
previously fitted baseline and original signals. This baseline estimator is 
general, fast and flexible in fitting baseline.

"""

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks 
           and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  
                 the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m) # Hmm this doesn't seem to be used at all? TG
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for 
                baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight 
                  # is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

def standardize_signal(signal_col):
    """ 
    Standardize the signal by subtracting the median and dividing by the 
    standard deviation. 
    """
    stdz_signal = (signal_col - np.median(signal_col)) / np.std(signal_col)
    #stdz_signal = (signal_col - np.mean(signal_col)) / np.std(signal_col)
    return stdz_signal

def linear_reg(z_ref,z_sig):
    """ 
    Performs linear regression on the reference signal (iso) to fit it to 
    a signal (RCaMP or GCaMP). Returns np array with fitted iso
    
    Partly copied from the Jupyter Notebook by 
    Martianova and associates (see lab OneNote)
    
    z_ref is raw_signal_iso['Stdz(Number,Color)']
    z_sig is raw_signal_(color)cmp['Stdz(Number,Color)']
    
    The Pandas data series need to be reshaped to work with lin.fit 
    (which perform a linear regression fit with the function Lasso)
    """
    lin = linear_model.Lasso(alpha=0.0001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
    
    nref = len(z_ref)
    nsig = len(z_sig)
    n = min(nref,nsig)
    
    ref = np.array(z_ref[0:n].values.reshape(n,1))
    sig = np.array(z_sig[0:n].values.reshape(n,1))
    
    lin.fit(ref,sig)
    z_ref_fitted = lin.predict(ref.reshape(n,1)).reshape(n,)
    return z_ref_fitted
    

###############################################################################
### READING IN RAW SIGNAL (already de-interleaved)
# Make sure the path name is correct to retrieve the data (can copy/paste 
# using windows explorer, just make sure all the slashes are /, and the path 
# name ends with /). Verify the names of the files (copy/paste works well) and 
# make sure the name ends with .csv.
# Note 410/415 is isosbestic, 470 is GCaMP, 560 is RCaMP
path_name = "C:/Users/HP/Desktop/Python Programs/Matlab conversion/Test Data/"
file_name_iso = "FST_C333_DatCreM2_410Raw_2020_8_13_(10.11.308)_2020-08-13T10_11_25.csv"
file_name_gcmp = "FST_C333_DatCreM2_470Raw_2020_8_13_(10.11.308)_2020-08-13T10_11_25.csv"
file_name_rcmp = "FST_C333_DatCreM2_560Raw_2020_8_13_(10.11.308)_2020-08-13T10_11_25.csv"
file_name_key = "FST_C333_DatCreM2_KeyDown_2020_8_13_(10.11.308)_2020-08-13T10_11_04.csv"

raw_signal_iso = read_signal(path_name,file_name_iso)
raw_signal_gcmp = read_signal(path_name,file_name_gcmp)
raw_signal_rcmp = read_signal(path_name,file_name_rcmp)
key_down = read_signal(path_name,file_name_key)

# Converting to relevant time (in seconds)
key_down["Timestamp"] -= raw_signal_iso["Timestamp"][0]
raw_signal_iso["Timestamp"] -= raw_signal_iso["Timestamp"][0]
raw_signal_gcmp["Timestamp"] -= raw_signal_gcmp["Timestamp"][0]
raw_signal_rcmp["Timestamp"] -= raw_signal_rcmp["Timestamp"][0]

### Step 1 - Moving window mean to smooth the signal
# This is the 'smoothed' signal, if ever refered to below. Also called the
# mean signal
raw_signal_iso['Mean0R'] = roll_mean(raw_signal_iso["Unmarked Fiber0R"])
raw_signal_iso['Mean1R'] = roll_mean(raw_signal_iso["Marked Fiber1R"])
raw_signal_iso['Mean2G'] = roll_mean(raw_signal_iso["Unmarked Fiber2G"])
raw_signal_iso['Mean3G'] = roll_mean(raw_signal_iso["Marked Fiber3G"])

raw_signal_gcmp['Mean2G'] = roll_mean(raw_signal_gcmp["Unmarked Fiber2G"])
raw_signal_gcmp['Mean3G'] = roll_mean(raw_signal_gcmp["Marked Fiber3G"])

raw_signal_rcmp['Mean0R'] = roll_mean(raw_signal_rcmp["Unmarked Fiber0R"])
raw_signal_rcmp['Mean1R'] = roll_mean(raw_signal_rcmp["Marked Fiber1R"])

# Plotting an example mean to see how things are progressing - looks good!
plt.figure()
plt.plot(raw_signal_iso["Timestamp"],raw_signal_iso["Unmarked Fiber2G"],'k',\
         raw_signal_iso["Timestamp"],raw_signal_iso["Mean2G"],'b',\
         raw_signal_gcmp["Timestamp"],raw_signal_gcmp["Mean2G"],'g')
plt.legend(("Raw Iso","Mean Iso","Mean GCaMP"))
plt.title("Unmarked Fiber, ROI 2G")
plt.savefig("Testing Means for 2G.pdf")

### Step 2 - is baseline correction with airPLS, from Zhang et al. 2010. 
# A python version of the functions is available on gibhub, just need to 
# understand how it takes in data and what it outputs!
lambda_ = 5e4  # SUPER IMPORTANT, controls flatness fo baseline.
               # Current best value known: 1e9 (from MATLAB version trials)
               # Martianova's exp program used lambd = 5e4
porder = 1
itermax = 50   # These values recommended by exp prog

raw_signal_iso['BLC 0R'] = airPLS(raw_signal_iso['Mean0R'],lambda_,porder,itermax)
raw_signal_iso['BLC 1R'] = airPLS(raw_signal_iso['Mean1R'],lambda_,porder,itermax)
raw_signal_iso['BLC 2G'] = airPLS(raw_signal_iso['Mean2G'],lambda_,porder,itermax)
raw_signal_iso['BLC 3G'] = airPLS(raw_signal_iso['Mean3G'],lambda_,porder,itermax)

raw_signal_gcmp['BLC 2G'] = airPLS(raw_signal_gcmp['Mean2G'],lambda_,porder,itermax)
raw_signal_gcmp['BLC 3G'] = airPLS(raw_signal_gcmp['Mean3G'],lambda_,porder,itermax)

raw_signal_rcmp['BLC 0R'] = airPLS(raw_signal_rcmp['Mean0R'],lambda_,porder,itermax)
raw_signal_rcmp['BLC 1R'] = airPLS(raw_signal_rcmp['Mean1R'],lambda_,porder,itermax)

# Plotting an example baseline correction to see how things are progressing
plt.figure()
plt.plot(raw_signal_iso["Timestamp"],raw_signal_iso["Mean0R"],'b',\
         raw_signal_iso["Timestamp"],raw_signal_iso["BLC 0R"],'purple')#,\
#         raw_signal_rcmp["Timestamp"],raw_signal_rcmp["Mean2G"],'r')
plt.legend(("Mean Iso","BLC Iso"))
plt.title("Unmarked Fiber, ROI 0R")
plt.savefig("Testing BLC for 0R.pdf")

# It came out REALLY flat. I think I might need some real data to test this on,
# to be sure things are coming out right. This fake data doesn't have any 
# changes to it, since it wasn't connected to a mouse and the fibers were not 
# manipulated during recording. 

### Step 2.5 - Subtract the BLC signal from the smoothed (mean) signal
# This step was not listed in the paper, but is in the exp program. So I'm 
# adding it here. It also was not included in the Matlab version of the 
# program (again, because it wasn't in the paper.)

raw_signal_iso['Sig0R'] = raw_signal_iso['Mean0R'] - raw_signal_iso['BLC 0R']
raw_signal_iso['Sig1R'] = raw_signal_iso['Mean1R'] - raw_signal_iso['BLC 1R']
raw_signal_iso['Sig2G'] = raw_signal_iso['Mean2G'] - raw_signal_iso['BLC 2G']
raw_signal_iso['Sig3G'] = raw_signal_iso['Mean3G'] - raw_signal_iso['BLC 3G']

raw_signal_gcmp['Sig2G'] = raw_signal_gcmp['Mean2G'] - raw_signal_gcmp['BLC 2G']
raw_signal_gcmp['Sig3G'] = raw_signal_gcmp['Mean3G'] - raw_signal_gcmp['BLC 3G']

raw_signal_rcmp['Sig0R'] = raw_signal_rcmp['Mean0R'] - raw_signal_rcmp['BLC 0R']
raw_signal_rcmp['Sig1R'] = raw_signal_rcmp['Mean1R'] - raw_signal_rcmp['BLC 1R']

# Plot this to be sure the signal was corrected properly
plt.figure()
plt.plot(raw_signal_gcmp['Timestamp'],raw_signal_gcmp['Sig3G'],'orange')
plt.title(('Corrected Smoothed Signal, Marked 3G'))

### Step 3 - standardize the waveform. 
# This appears to be the baseline corrected signal minus the median value, 
# then divide by the standard deviation. I thought it was the mean we are 
# supposed to subtract, but Martianova says "median(Int)", so median it is.

raw_signal_iso['Stdz0R'] = standardize_signal(raw_signal_iso['Sig0R'])
raw_signal_iso['Stdz1R'] = standardize_signal(raw_signal_iso['Sig1R'])
raw_signal_iso['Stdz2G'] = standardize_signal(raw_signal_iso['Sig2G'])
raw_signal_iso['Stdz3G'] = standardize_signal(raw_signal_iso['Sig3G'])

raw_signal_gcmp['Stdz2G'] = standardize_signal(raw_signal_gcmp['Sig2G'])
raw_signal_gcmp['Stdz3G'] = standardize_signal(raw_signal_gcmp['Sig3G'])

raw_signal_rcmp['Stdz0R'] = standardize_signal(raw_signal_rcmp['Sig0R'])
raw_signal_rcmp['Stdz1R'] = standardize_signal(raw_signal_rcmp['Sig1R'])

# Plotting an example standardized signal to see how things are progressing
plt.figure()
plt.plot(raw_signal_gcmp["Timestamp"],raw_signal_gcmp["Mean3G"],'b',\
         raw_signal_gcmp["Timestamp"],raw_signal_gcmp["BLC 3G"],'purple',\
         raw_signal_gcmp["Timestamp"],raw_signal_gcmp["Stdz3G"],'g')
plt.legend(("Mean Gcmp","BLC Gcmp","Stdrzd Gcmp"))
plt.title("Marked Fiber, ROI 3G")
plt.savefig("Testing Standardization for 3G.pdf")

# Still need real data, since this data doesn't seem to be producing any 
# recognizable results when analyzed, making me think something may be wrong 
# with the program. NEED REAL DATA!!!!

### Step 4 - apply non-negative robust linear regression.
# Basically, fit the Isobestic signal to the complimentary GCaMP (or RCaMP) 
# signal. Not all signals have same length (off by 1, usually). Need to trim 
# back to shortest signal

ni = len(raw_signal_iso["Timestamp"])
nr = len(raw_signal_rcmp["Timestamp"])
ng = len(raw_signal_gcmp["Timestamp"])
n = min(ni,ng,nr)

indx = list(range(n))

final_sig_GCaMP = pd.DataFrame(raw_signal_gcmp['Timestamp'][0:n])
final_sig_RCaMP = pd.DataFrame(raw_signal_rcmp['Timestamp'][0:n])

final_sig_RCaMP['FitIso0R'] = linear_reg(raw_signal_iso['Stdz0R'],raw_signal_rcmp['Stdz0R'])
final_sig_RCaMP['FitIso1R'] = linear_reg(raw_signal_iso['Stdz1R'],raw_signal_rcmp['Stdz1R'])
final_sig_GCaMP['FitIso2G'] = linear_reg(raw_signal_iso['Stdz2G'],raw_signal_gcmp['Stdz2G'])
final_sig_GCaMP['FitIso3G'] = linear_reg(raw_signal_iso['Stdz3G'],raw_signal_gcmp['Stdz3G'])


# plotting to see how the signal is changing
plt.figure()
plt.plot(raw_signal_gcmp["Timestamp"],raw_signal_gcmp["Stdz3G"],'g',\
         final_sig_GCaMP["Timestamp"],final_sig_GCaMP["FitIso3G"],'purple')
plt.legend(("Stdrzd Gcmp","LR of 3G iso"))
plt.title("Marked Fiber, ROI 3G")
plt.savefig("Testing Linear Regression for 3G.pdf")

### Step 5 - bringing it all together
# z(dF/F) = Stdz_sig - FitIso_sig

final_sig_RCaMP['zdFF 0R'] = raw_signal_rcmp['Stdz0R'][0:n] - final_sig_RCaMP['FitIso0R']
final_sig_RCaMP['zdFF 1R'] = raw_signal_rcmp['Stdz1R'][0:n] - final_sig_RCaMP['FitIso1R']
final_sig_GCaMP['zdFF 2G'] = raw_signal_gcmp['Stdz2G'][0:n] - final_sig_GCaMP['FitIso2G']
final_sig_GCaMP['zdFF 3G'] = raw_signal_gcmp['Stdz3G'][0:n] - final_sig_GCaMP['FitIso3G']

### plotting final signals and saving figures
x_key = list(key_down["Timestamp"])

plt.figure()
plt.plot(final_sig_RCaMP['Timestamp'], final_sig_RCaMP['zdFF 0R'],'red')
for keyline in x_key:
    plt.axvline(x=keyline,ls='--',color='black')
plt.legend(("Signal","Event"))
plt.title("Final Signal, RCaMP Unmarked Fiber")
plt.ylabel("z dF/F")
plt.xlabel("Time (sec)")
plt.savefig("Final Signal RCaMP Unmrk, M3 Cre Test.pdf")

plt.figure()
plt.plot(final_sig_RCaMP['Timestamp'], final_sig_RCaMP['zdFF 1R'],'red')
for keyline in x_key:
    plt.axvline(x=keyline,ls='--',color='black')
plt.legend(("Signal","Event"))
plt.title("Final Signal, RCaMP Marked Fiber")
plt.ylabel("z dF/F")
plt.xlabel("Time (sec)")
plt.savefig("Final Signal RCaMP Mrk, M3 Cre Test.pdf")

plt.figure()
plt.plot(final_sig_GCaMP['Timestamp'], final_sig_GCaMP['zdFF 2G'],'green')
for keyline in x_key:
    plt.axvline(x=keyline,ls='--',color='black')
plt.legend(("Signal","Event"))
plt.title("Final Signal, GCaMP Unmarked Fiber")
plt.ylabel("z dF/F")
plt.xlabel("Time (sec)")
plt.savefig("Final Signal GCaMP Unmrk, M3 Cre Test.pdf")

plt.figure()
plt.plot(final_sig_GCaMP['Timestamp'], final_sig_GCaMP['zdFF 3G'],'green')
for keyline in x_key:
    plt.axvline(x=keyline,ls='--',color='black')
plt.legend(("Signal","Event"))
plt.title("Final Signal, GCaMP Marked Fiber")
plt.ylabel("z dF/F")
plt.xlabel("Time (sec)")
plt.savefig("Final Signal GCaMP Mrk, M3 Cre Test.pdf")

# save the final signals into csv files
final_sig_RCaMP.to_csv('RCaMP_M3 Cre Test.csv',index=False)
final_sig_GCaMP.to_csv('GCaMP_M3 Cre Test.csv',index=False)
key_down.to_csv('keydown_M3 Cre Test.csv',index=False)

# See if can test with older, good data. Will have to insert titles for columns






