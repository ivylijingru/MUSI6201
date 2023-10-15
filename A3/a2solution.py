# -*- coding: utf-8 -*-
"""
comment
"""

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt

def  block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb,t)

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

def compute_spectrogram(xb):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(sp.fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) #let's be pedantic about normalization

    
    return X

def  extract_spectral_centroid(xb,fs):
    return (FeatureSpectralCentroid(compute_spectrogram(xb),fs))

def extract_spectral_crest(xb):
    return (FeatureSpectralCrestFactor(compute_spectrogram(xb)))

def extract_spectral_flux(xb):
    return (FeatureSpectralFlux(compute_spectrogram(xb)))

def extract_rms(xb):
    return (FeatureTimeRms(xb))

def extract_zerocrossingrate(xb):
    return (FeatureTimeZeroCrossingRate(xb))

def extract_features(afAudioData,blockSize=1024,hopSize=256,fs=44100):
    [xb,t] = block_audio(afAudioData, blockSize, hopSize, fs)

    V = np.zeros([5,xb.shape[0]])

    V[0,:] = extract_spectral_centroid(xb,fs)
    V[1,:] = extract_rms(xb)
    V[2,:] = extract_zerocrossingrate(xb)
    V[3,:] = extract_spectral_crest(xb)
    V[4,:] = extract_spectral_flux(xb)

    return V

def aggregate_feature_per_file(V):
    va = np.concatenate([np.mean(V,axis=1), np.std(V,axis=1)])
    return va

def get_feature_data(path, blockSize=1024, hopSize=256):
    import os
    from ToolReadAudio import ToolReadAudio

    # get number of files
    iNumOfFiles = 0
    for file in os.listdir(path):
        if file.endswith(".wav"):
            iNumOfFiles += 1
        else:
            continue

    if iNumOfFiles == 0:
        return -1
    else:
        Vf = np.zeros([10,iNumOfFiles])

    # for loop over files
    iNumOfFiles = 0
    for file in os.listdir(path):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(path + file)
        else:
            continue

        # extract features
        Vf[:,iNumOfFiles-1] = aggregate_feature_per_file(extract_features(afAudioData))


    return Vf

def  normalize_zscore(featureData):

    mu = np.mean(featureData,axis=1)
    std = np.std(featureData,axis=1)

    normFeatureData = ((featureData.transpose() - mu) / std).transpose()
    return normFeatureData

def visualize_features(cPath):
    music_path = cPath + 'music_wav/'
    speech_path = cPath + 'speech_wav/'

    blockSize = 1024
    hopSize = 256

    Vf_music = get_feature_data(music_path, blockSize, hopSize)
    Vf_speech = get_feature_data(speech_path, blockSize, hopSize)
    Vf = np.concatenate((Vf_music,Vf_speech),axis=1) 
    Vf_norm = normalize_zscore(Vf)

    showIdx = np.array([[0,3], [4,2], [1,6], [7,8], [5,9]])

    for n in range(Vf.shape[0]//2):
        plt.figure()
        plt.plot(Vf_norm[showIdx[0,n],:Vf_music.shape[1]],Vf_norm[showIdx[1,n],:Vf_music.shape[1]],'ro',Vf_norm[showIdx[0,n],Vf_music.shape[1]:],Vf_norm[showIdx[1,n],Vf_music.shape[1]:],'bo')
        plt.show()



def FeatureSpectralCentroid(X, f_s):

    # X = X**2 removed for consistency with book

    norm = X.sum(axis=0, keepdims=True)
    norm[norm == 0] = 1

    vsc = np.dot(np.arange(0, X.shape[0]), X) / norm

    # convert from index to Hz
    vsc = vsc / (X.shape[0] - 1) * f_s / 2

    return (vsc)

def FeatureTimeRms(xb):

    # number of results
    numBlocks = xb.shape[0]

    # allocate memory
    vrms = np.zeros(numBlocks)

    for n in range(0, numBlocks):
        # calculate the rms
        vrms[n] = np.sqrt(np.dot(xb[n,:], xb[n,:]) / xb.shape[1])

    # convert to dB
    epsilon = 1e-5  # -100dB

    vrms[vrms < epsilon] = epsilon
    vrms = 20 * np.log10(vrms)

    return (vrms)

def FeatureTimeZeroCrossingRate(xb):

    # number of results
    numBlocks = xb.shape[0]

    # allocate memory
    vzc = np.zeros(numBlocks)

    for n in range(0, numBlocks):
        # calculate the zero crossing rate
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign(xb[n,:]))))

    return (vzc)

def FeatureSpectralCrestFactor(X, f_s = 44100):

    norm = X.sum(axis=0)
    norm[norm == 0] = 1

    vtsc = X.max(axis=0) / norm

    return (vtsc)

def FeatureSpectralFlux(X, f_s = 44100):

    # difference spectrum (set first diff to zero)
    X = np.c_[X[:, 0], X]
    # X = np.concatenate(X[:,0],X, axis=1)
    afDeltaX = np.diff(X, 1, axis=1)

    # flux
    vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]

    return (vsf)
