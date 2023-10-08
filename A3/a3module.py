## Main module for Assignment A3 - Includes all of the reuqisite functions

# import dependencies
import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os

# add path to A1 and A2 modules
import sys
path = os.getcwd()
sys.path.append(path + '/A1 Pitch Tracking')
sys.path.append(path + '/A2')
from A1_helper_module import *


## Part A
# A.1 - Create a spectrogram function compute_spectrogram(xb, fs) which computes the magbitude spectrogram of a given block of audio data xb sampled at a rate fs.
def create_spectrogram(xb, fs):
    # compute the spectrogram of a given block of audio data xb sampled at a rate fs
    # xb: block of audio data
    # fs: sampling rate
    # returns: 2D array of spectrogram data
    # create a hann window of the same length as the block of audio data
    hann_window = np.hanning(len(xb))
    # multiply the window by the block of audio data
    xb = xb * hann_window
    # compute the fft of the block of audio data 
    fft = np.fft.fft(xb)
    # compute the magnitude of the fft
    magnitude = np.abs(fft)
    # compute the spectrogram from the fft, rejecting the second half of the fft
    X = magnitude[:len(magnitude)//2]
    # create a frequency vector
    fInHz = np.arange(0, fs/2, fs/len(X))
    return X, fInHz

#A.2 - create a pitch tracker that estimates the pitch from the spectrogram by finding the blockwise peak of the spectrogram and returning the corresponding frequency
def track_pitch_fftmax(x, blockSize, hopSize, fs):
    # block input audio vector x
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    # calculate magnitude spectrogram
    spect, freq = create_spectrogram(xb, fs)
    # find blockwise peak of spectrogram
    maxIndex = np.argmax(spect, axis=0)
    # return corresponding frequency vector
    return freq[maxIndex]

# A.3 : Question: Frequency resolution of blocked audio pitch tracker and how it can be improved
    # Answer: The frequency resolution is a function of block length and sampling rate, and fft length = fmin = fs/N, where N is the block length.
    # Zero padding can be used to improve the frequency resolution, but it will not improve the accuracy of the pitch tracker.

# Part B