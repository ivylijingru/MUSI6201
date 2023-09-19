# Module with functions for A2

# load dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
# import helper function module from A1
from A1_helper_module import *
import os

# Inputs:
fs = 44.1e3 # sampling rate
block_size = 1024 # block size
hop_size = 512 # hop size

# load audio and corresponding text data files from directory
# set absolute path to directory containing all files
musicpath = r'/Users/ananyabhardwaj/Downloads/music_speech data/music_wav' # update as required for your system
speechpath = r'/Users/ananyabhardwaj/Downloads/music_speech data/speech_wav' # update as required for your system

# create spectral centroid function - takes blocked audio and sampling rate as input
def extract_spectral_centroid(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(b_size) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_cent = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        X_fft[i,:] = (np.abs(np.fft.fft(hw*xb[i,:]))[0:flen])/(0.5*b_size)
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
def extract_zerocrossingrate(xb):
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
    hw = np.hanning(b_size) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_crest = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        X_fft[i,:] = (np.abs(np.fft.fft(hw*xb[i,:]))[0:flen])/(0.5*b_size)
        S_crest[i] = (np.max(X_fft[i,:]))/np.sum(X_fft[i,:]) # spectral crest
    return S_crest

# create a function to calculate spectral flux
def extract_spectral_flux(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(b_size) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_flux = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks-1):
        X_fft[i,:] = (np.abs(np.fft.fft(hw*xb[i,:]))[0:flen])/(0.5*b_size)
        X_fft[i+1,:] = (np.abs(np.fft.fft(hw*xb[i+1,:]))[0:flen])/(0.5*b_size)
        S_flux[i] = np.sqrt(np.sum(np.square(X_fft[i+1,:] - X_fft[i,:])))/ (b_size+1) # spectral flux
    return S_flux








                                                  ## Plotting and Testing

# create a sample signal
t = np.arange(0,1,1/fs) # time vector
x = np.sin(2*np.pi*441*t) #+ np.sin(2*np.pi*880*t) + np.sin(2*np.pi*1320*t)  # signal vector
# add some filtered white noise to signal vector
noise = np.random.normal(0,1,len(x))
b, a = signal.butter(2, 15e3, 'low', fs = fs, analog=False) # to prevent aliasing, filter noise before adding to signal
noise = signal.filtfilt(b, a, noise)
#x = x + 0.1*noise


def extract_features(x, block_size, hop_size, fs):
    # First block an input signal
    xb, timeInSec = block_audio(x,block_size, hop_size, fs)
    NumOfBlocks = len(timeInSec)

    # create an empty feature array
    feature_array = np.zeros((NumOfBlocks, 5))

    # calculate spectral centroid
    S_cent = extract_spectral_centroid(xb, fs)
    feature_array[:,0] = S_cent

    # calculate rms_dB
    rms_dB = extract_rms(xb)
    feature_array[:,1] = rms_dB

    # calculate spectral crest
    S_crest = extract_spectral_crest(xb, fs)
    feature_array[:,2] = S_crest

    # calculate zero crossings
    zcr = extract_zerocrossingrate(xb)
    feature_array[:,3] = zcr

    # calculate spectral flux
    S_flux = extract_spectral_flux(xb, fs)
    feature_array[:,4] = S_flux

    return feature_array, timeInSec

def aggregate_feature_per_file(features):
    # aggregate features per file by taking the mean and standard deviation of each feature
    agg_features = np.zeros(10) # initialize empty array
    # loop over features to calculate mean and standard deviation
    c = 0
    for i in range(np.shape(features)[1]):
        agg_features[c],agg_features[c+1]  =   np.mean(features[:,i]), np.std(features[:,i])
        c = c+2
    return agg_features # remove first row of zeros

# loop over all files in the directory and extract features
# initialize empty array
features_music = np.zeros((len(os.listdir(musicpath)),10))
features_speech = np.zeros((len(os.listdir(speechpath)),10))


# plot input signal
plt.figure()
plt.plot(t,x)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Input Signal')
plt.show(block=False)

plt.figure()
plt.plot(t,noise)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Noise')
plt.show(block=False)

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

features, timeInSec = extract_features(x, block_size, hop_size, fs)
features_agg = aggregate_feature_per_file(features)

print(features_agg)

# plot spectral centroid vector
plt.figure()
plt.plot(timeInSec, features[:,0])
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Spectral Centroid')
plt.show(block=False)

# plot RMS Energy
plt.figure()
plt.plot(timeInSec, features[:,1])
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('RMS [dB]')
plt.show(block=False)

# plot zerocrossings
plt.figure()
plt.plot(timeInSec, features[:,2])
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Zero Crossings')
plt.show(block=False)

# plot spectral crest
plt.figure()
plt.plot(timeInSec, features[:,3])
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Spectral Crest')
plt.show(block=False)

# plot spectral flux
plt.figure()
plt.plot(timeInSec, features[:,4])
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Spectral Flux')
plt.show()