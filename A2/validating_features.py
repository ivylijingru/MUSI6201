# Module with functions for A2

# load dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa
# import helper function module from A1
from A1_helper_module import *

# Inputs:
fs = 44.1e3 # sampling rate
block_size = 1024 # block size
hop_size = 512 # hop size

# create a sample signal
t = np.arange(0,10,1/fs) # time vector
x = np.sin(2*np.pi*441*t) #+ np.sin(2*np.pi*880*t) + np.sin(2*np.pi*1320*t)  # signal vector

# First block an input signal
xb, timeInSec = block_audio(x,block_size, hop_size, fs)
NumOfBlocks = len(timeInSec)


# create spectral centroid function - takes blocked audio and sampling rate as input

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


# plot sample spectra, sample blocks as well
plt.figure()
plt.plot(f, X_fft[np.random.randint(0, NumOfBlocks),:])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Sample Spectrum')
plt.show(block=False)

plt.figure()
plt.plot(xb[np.random.randint(0, NumOfBlocks),:])
plt.xlabel('Time (samples)')
plt.ylabel('Magnitude')
plt.title('Sample block')
plt.show(block=False)

# calculate spectral centroid with librosa and compare
S_cent_lib = librosa.feature.spectral_centroid(S=np.transpose(X_fft))


# plot spectral centroid vector
plt.figure()
plt.plot(timeInSec, S_cent)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Spectral Centroid - local')
plt.show(block=False)

# plot spectral centroid vector
plt.figure()
plt.plot(timeInSec, np.squeeze(S_cent_lib))
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Spectral Centroid - librosa')
plt.show()