## Main module for Assignment A3 - Includes all of the reuqisite functions

# import dependencies
import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os

# add path to A1 and A2 modules
import sys

from a1solution import convert_freq2midi
from a2solution import block_audio


## Part A
# A.1 - Create a spectrogram function compute_spectrogram(xb, fs) which computes the magbitude spectrogram of a given block of audio data xb sampled at a rate fs.
def create_spectrogram(xb, fs):
    # compute the spectrogram of a given block of audio data xb sampled at a rate fs
    # xb: block of audio data
    # fs: sampling rate
    # returns: 2D array of spectrogram data
    # create a hann window of the same length as the block of audio data
    NumOfBlocks, blockSize = np.shape(xb)
    hann_window = np.hanning(np.shape(xb)[1])
    hann_window = np.resize(hann_window, (np.shape(xb)))
    # multiply the window by the block of audio data
    xb = xb * hann_window
    # compute the fft of the block of audio data
    fft = np.fft.fft(xb)
    # compute the magnitude of the fft
    magnitude = np.abs(fft) * (2 / blockSize)
    # create a frequency vector
    fInHz = np.arange(0, fs / 2 + 1, fs / blockSize)
    # compute the spectrogram from the fft, rejecting the second half of the fft
    # magnitude = np.transpose(magnitude)
    Y = magnitude[:, 0 : blockSize // 2 + 1]

    X = np.transpose(Y)
    return X, fInHz


# A.2 - create a pitch tracker that estimates the pitch from the spectrogram by finding the blockwise peak of the spectrogram and returning the corresponding frequency
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
# Harmonic Product Spectrum (HPS) Pitch Tracker - multiply each spectrogram block with its "harmonics" an order number of times and extract peak
# B.1 - get f0 from Hps function
def get_f0_from_Hps(X, fs, order):
    P = X
    blockSize, NumOfBlocks = np.shape(X)

    # loop over all the blocks and
    for i in range(np.shape(X)[-1]):
        for j in range(order - 1):
            X1 = X[np.arange(1, blockSize, j + 2), i]
            P[:, i] = P[:, i] * np.append(X1, np.zeros(len(P) - len(X1)))

    # calculate the peak value for each block and store it into a vector
    maxIndex = np.argmax(P, axis=0)
    freq = np.arange(0, fs / 2, fs / blockSize)
    f0 = freq[maxIndex]
    return f0


# B.2 track pitch with HPS function
def track_pitch_hps(x, blockSize, hopSize, fs):
    # variables
    order = 4
    # block input audio vector x
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    # calculate magnitude spectrogram
    spect, freq = create_spectrogram(xb, fs)
    # estimate fundamental frequency using the HPS method
    f0 = get_f0_from_Hps(spect, fs, order)
    return f0


# Part C - Voicing Detection
# create a voicing mask function
# C.1 and C.2
# xb = block_audio(x)
# rmsDb = extract_rms(xb)
def create_voicing_mask(rmsDb, thresholdDb):
    # loop over the vector block by block and apply mask based on threshold
    mask = np.zeros(np.shape(rmsDb))
    for i in range(len(rmsDb)):
        if rmsDb[i] > thresholdDb:
            mask[i] = 1


# C.3 - create a function that applies a voicing mask to the extracted f0 vector
def apply_voicing_mask(f0, mask):
    f0 = f0 * mask
    return f0


# Part D - Evaluation
# calculate how many false positive values (estimated fundamental frequencies where the mask is set to 0)
def eval_voiced_fp(estimation, annotation):
    # calculate zero values in annotation
    N = len(annotation)
    zero_inds = np.where(annotation == 0)
    fps = np.count_nonzero(estimation[zero_inds])
    pfp = fps / len(zero_inds)
    return pfp


# calculate how many false negatives values (estimated fundamental frequencies where the mask is nonzero but estimate is 0)
def eval_voiced_fn(estimation, annotation):
    # calculate zero values in annotation
    N = len(annotation)
    nonzero_inds = np.where(annotation != 0)
    fns = np.size(estimation[nonzero_inds] == 0)
    pfn = fns / len(nonzero_inds)
    return pfn


def eval_pitchtrack_v2(estimate_in_hz, groundtruth_in_hz):
    estimate_in_hz = np.array(estimate_in_hz)  # make sure inputs are numpy arrays
    groundtruth_in_hz = np.array(groundtruth_in_hz)  # make sure inputs are numpy arrays
    estimate_pitch_midi = convert_freq2midi(estimate_in_hz)  # convert estimate to MIDI
    groundtruth_pitch_midi = convert_freq2midi(
        groundtruth_in_hz
    )  # convert ground truth to MIDI

    p_err = estimate_pitch_midi - groundtruth_pitch_midi  # error in pitch  - MIDI
    err_cent = 100 * p_err
    errCentRms = np.sqrt(np.sum(np.square(err_cent)) / np.size(err_cent))

    pfp = eval_voiced_fp(estimate_in_hz, groundtruth_in_hz)
    pfn = eval_voiced_fn(estimate_in_hz, groundtruth_in_hz)

    return errCentRms, pfp, pfn


## Part E - Evaluation
# E.1
def executeassign3():
    # create test signal
    fs = 44.1e3
    t1 = np.arange(0, 1, 1 / fs)
    t2 = np.arange(1, 2, 1 / fs)
    t = np.append(t1, t2)
    f1 = 441
    f2 = 882
    x = np.append(np.sin(2 * np.pi * f1 * t1), np.sin(2 * np.pi * f2 * t2))
    # plot test signal
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (raw)")
    plt.title("Test Signal")
    plt.show(block=False)

    blockSize = 1024
    hopSize = 512
    f0_fft = track_pitch_fftmax(x, blockSize, hopSize, fs)
    f0_hps = track_pitch_hps(x, blockSize, hopSize, fs)
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)

    # plot returns
    plt.figure()
    plt.plot(f0_fft)
    plt.plot(f0_hps)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (raw)")
    plt.title("Estimated Pitch")
    plt.legend("FFT", "HPS")
    plt.show()

    # calculate absolute error per block
    annotation = np.append(
        np.ones(np.ceil(len(timeInSec) / 2).astype(int)) * f1,
        np.ones(np.floor(len(timeInSec) / 2).astype(int)) * f2,
    )
    err_fft = np.abs(f0_fft - annotation)
    err_hps = np.abs(f0_hps - annotation)

    # plot errors
    plt.figure()
    plt.plot(err_fft)
    plt.plot(err_hps)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (raw)")
    plt.title("Error")
    plt.legend("FFT", "HPS")
    plt.show()


if __name__ == "__main__":
    executeassign3()
