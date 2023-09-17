# Module with functions for A2

# load dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
# import helper function module from A1
from A1_helper_module import *

# Inputs:
fs = 44.1e3 # sampling rate
block_size = 1024 # block size
hop_size = 512 # hop size


# create spectral centroid function - takes blocked audio and sampling rate as input
def extract_spectral_centroid(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(flen) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_cent = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        X_fft[i,:] = (np.abs(np.fft.fft(xb[i,:]))[0:flen])/(0.5*b_size)
        # window fft
        #X_fft[i,:] = hw*X_fft[i,:]
        S_cent[i] = (np.dot(f,X_fft[i,:]))/np.sum(X_fft[i,:]) # spectral centroid
    return S_cent

# calculate RMS energy in dB
def extract_rms(xb):
    nblocks, b_size = np.shape(xb) # get size of input vector
    rms_dB = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        rms_dB[i] = np.maximum(20*np.log10(np.sqrt(np.sum(np.square(xb[i,:])/np.size(xb[i,:])))), -100)
    return rms_dB
        


# Create a function for extracting zero_crossings
def extract_zerocrossingrate(xb,fs):
    # loop over each block and find zero crossing rate by finding negative difference values
    nblocks, b_size = np.shape(xb) # get size of input vector
    zcr = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        zcr[i] = (1/(2*b_size))*np.sum( np.abs(np.diff(np.sign(xb[i,:]))))
    return zcr

# create a function to calculate spectral crest
def extract_spectral_crest(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(flen) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_crest = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        X_fft[i,:] = (np.abs(np.fft.fft(xb[i,:]))[0:flen])/(0.5*b_size)
        # window fft
        #X_fft[i,:] = hw*X_fft[i,:]
        S_crest[i] = (np.max(X_fft[i,:]))/np.sum(X_fft[i,:]) # spectral crest
    return S_crest
                                                  ## Plotting and Testing

# create a sample signal
t = np.arange(0,1,1/fs) # time vector
x = np.sin(2*np.pi*440*t) #+ np.sin(2*np.pi*880*t) + np.sin(2*np.pi*1320*t)  # signal vector


# First block an input signal
xb, timeInSec = block_audio(x,block_size, hop_size, fs)
NumOfBlocks = len(timeInSec)

# calculate spectral centroid
S_cent = extract_spectral_centroid(xb, fs)

# calculate rms_dB
rms_dB = extract_rms(xb)



# plot sample spectrum
# plt.figure()
# plt.plot(f, X_fft[4,:])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Sample Spectrum')
# plt.show(block=False)

# plot spectrogram
# plt.figure()
# plt.pcolormesh(X_fft.T, shading='flat')
# plt.colorbar()
# plt.xticks(np.arange(0,NumOfBlocks,10),np.round(timeInSec[0:NumOfBlocks:10],1))
# plt.yticks(np.arange(0,flen,100),np.round(f[0:flen:100]))
# plt.xlabel('Time (sec)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Spectrogram')
# plt.show(block=False)


# plot spectral centroid vector
plt.figure()
plt.plot(timeInSec, S_cent)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Spectral Centroid')
plt.show(block=False)

# plot RMS Energy
plt.figure()
plt.plot(timeInSec, rms_dB)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('RMS [dB]')
plt.show()