# This is a helper function for implementing blocking, or block processing of data (here, audio data).
# The function takes in a signal, and returns a list of blocks of the signal, each of length block_size.

# import necessary libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt


def block_audio(x,block_size, hop_size, fs):
    x = np.array(x) # convert to numpy array
    NumOfBlocks = int(np.ceil(len(x)/hop_size)) # number of blocks - will pad the last block with zeros
    xb = np.zeros((NumOfBlocks, block_size)) # initialize 2D array for audio blocks
    timeInSec = np.zeros((NumOfBlocks,1)) # start time of each block in seconds
    i = 0 # iteration variable for xb
    j = 0 # iteration variable for timeInSec
    while i +  block_size < len(x):
        xb[j,:] = x[i:i+block_size]
        timeInSec[j] = i/fs
        i += hop_size
        j += 1
    return xb, timeInSec
    

# Test the function
fs = 44100
f1 = 441 # frequency of first second of sine wave in Hz
f2 = 882 # frequency of second second of sine wave in Hz
block_size = 1024 # number of samples per block
hop_size = 512 # number of samples per hop
x1 = np.sin(2*np.pi*f1*np.arange(0,1,1/fs)) # 1 second of 441 Hz sine wave
x2 = np.sin(2*np.pi*f2*np.arange(0,1,1/fs)) # 1 second of 882 Hz sine wave
x = np.concatenate((x1,x2)) # combine into 2 second signal

xb, timeInSec = block_audio(x,block_size, hop_size, fs)

# Plot the signal
plt.figure(figsize=(10,4))
plt.plot(np.arange(0,2,1/fs),x)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.title('Original Signal')


# Plot a couple sample blocks
plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(np.arange(0,block_size/fs,1/fs),xb[0,:])
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.title('Block 0')
plt.subplot(2,1,2)
middleblock = int(np.ceil(len(xb)/2) + 1)
plt.plot(np.arange(0,block_size/fs,1/fs),xb[middleblock,:])
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.title('Block ' + str(middleblock))
plt.show()

    
# if __name__ == "__main__":
#     print("This is a module. Import this module to use the function block_audio(x,block_size, hop_size, fs)")

