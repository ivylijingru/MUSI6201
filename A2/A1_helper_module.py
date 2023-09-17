# Main script that takes as an input an audio file (x), sampling rate (fs), block size and hop size in samples and calculates fundamental 
# frequency using an autocorrelation approach

# import dependencies
import numpy as np
import matplotlib.pyplot as plt

# "main" function that calls other functions to calculate blockwise pitch for an input signal
def track_pitch_acf(x,block_size,hop_size,fs):
    # call blocking function
    xb, timeInSec = block_audio(x,block_size, hop_size, fs)
    NumOfBlocks = len(timeInSec)

    # vector to store f0 for each block
    f0_vec = np.zeros((NumOfBlocks,1))

    # call ACF function and get_f0 function in a loop and save output
    for i in range(NumOfBlocks):
        # bls_normalized = np.dot(xb[i,:],xb[i,:])
        # if bls_normalized == 0:
        #     bls_normalized = 1

        # calculate ACF for each block
        r = comp_acf(xb[i,:],is_normalized = True) # normaliztion turned on by default

        f0_vec[i] = get_f0_from_acf(r, fs)


    return f0_vec, timeInSec

# function to block audio signal
def block_audio(x,block_size, hop_size, fs):
    x = np.array(x) # convert to numpy array
    NumOfBlocks = int(np.floor(len(x)/hop_size)) # number of blocks - will pad the last block with zeros if necessary
    xb = np.zeros((NumOfBlocks, block_size)) # initialize 2D array for audio blocks
    timeInSec = np.zeros((NumOfBlocks,1)) # start time of each block in seconds
    i = 0 # iteration variable for xb
    j = 0 # iteration variable for timeInSec
    while i +  block_size <= len(x):
        xb[j,:] = x[i:i+block_size]
        timeInSec[j] = i/fs
        i += hop_size
        j += 1
    return xb[0:j], timeInSec[0:j]

# function to calculate ACF
def comp_acf(inputVector,is_normalized: bool):
    inputVector = np.array(inputVector) # convert to numpy array
    if is_normalized:
         bls_normalized = np.dot(inputVector,inputVector)
         if bls_normalized == 0: # to avoid divide by zero error
            bls_normalized = 1
    else:
        bls_normalized = 1 # setting it to 1 means we don't normalize the ACF

    noutput = len(inputVector) # length of output vector - since we only return half the ACF vector
    r = np.zeros((noutput,1)) # ACF output
    iv2 = inputVector # time shifted copy of vector for ACF calculation

    for i in range(noutput):
        r[i] = np.dot(inputVector, iv2)
        iv2 = np.roll(iv2,1) # right shift by one 
        iv2[0] = 0 # set rolled sample to zero
        
    
    r = np.divide(r,bls_normalized) # normalize with the provided normalization parameter
    return r

# function to calculate f0 from ACF
def get_f0_from_acf (r, fs):
    fmax = 2000
    # find the lags between the first and second peaks of the ACF and estimate the Time Period and Fundamental frequency
    i1 = np.argmax(r) # index of first maxima - which is the first index since it's at 0 lags
    min_lvl = 0.5 # an arbitrary acf value which it should fall below, set to not pick out main lobe value as second maxima
    ix = np.where(r[i1+1:] < min_lvl) # find index of acf crossing this minimum level 

    if not np.size(ix): # if ix is a null vector
        ix = np.ceil(np.divide(fs, fmax)).astype(int)
    else:
        ix = i1 + ix[0][0] + 1

    i2 = np.argmax(r[ix:])# index of second maxima
    f0 = np.divide(fs, i2 + ix)
    return f0 
