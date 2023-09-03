# main script that loads audio and text files, performs blocking and pitch tracking, and evaluates pitch tracking performance - printing out cent error in MIDI pitch

# load dependencies
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from track_pitch_acf import *
from eval_pitchtrack import *
import os

# load audio and corresponding text data files from directory
# set absolute path to directory containing all files
mypath = r'/Users/ananyabhardwaj/Downloads/trainData' # update as required for your system

# set processing variables
block_size = 1024 # number of samples per block
hop_size = 512 # number of samples per hop


def run_evaluation(complete_path_to_data_folder):
    # loads pairs of audio and text files from the same directory
    audio_files = [] # initialize list of audio files
    text_files = [] # initialize list of text files
    for file in os.listdir(complete_path_to_data_folder): # iterate over all files in the directory
        if file.endswith('.wav'): # if the file is an audio file
            audio_files.append(os.path.join(complete_path_to_data_folder, file)) # add the file to the list of audio files
            text_files.append(os.path.join(complete_path_to_data_folder, file.replace('.wav','.f0.Corrected.txt'))) # add the corresponding text file to the list of text files


    # loop over audio and corresponding text files, perform cent error analyses on each file and report deviation from ground truth
    audio_data = [] # initialize list of audio data
    text_data = [] # initialize list of text data
    cent_error = [] # initialize list of cent errors
    for i in range(len(audio_files)): # iterate over all audio files
        print('Processing file ' + str(i+1) + ' of ' + str(len(audio_files)) + '...') # print progress
        fs, x = wav.read(audio_files[i]) # load audio file
        # # plot audio file
        # plt.figure(figsize=(10,4))
        # plt.plot(np.arange(0,len(x)/fs,1/fs),x)
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Amplitude')
        # plt.title('Original Signal')
        # plt.show(block=False)


        # read text file columns 1 and 3 into numpy array - columns 1 and 3 contain start time and f0 values
        text_data = np.loadtxt(text_files[i], usecols=(0,2)) # load text file
        # call pitch tracking function
        f0_vec, timeInSec = track_pitch_acf(x,block_size,hop_size,fs)
        # keep only same number of values in text file as in f0_vec
        text_data = text_data[0:len(f0_vec),:]

        # exclude all values where text file f0 is 0
        f0_vec = f0_vec[text_data[:,1] != 0]
        timeInSec = timeInSec[text_data[:,1] != 0]
        text_data = text_data[text_data[:,1] != 0,:]
        

        # plot calculated f0 vector and text file f0 vector for comparison
        # plt.figure(figsize=(10,4))
        # plt.plot(timeInSec, f0_vec, label='Calculated f0')
        # plt.plot(text_data[:,0], text_data[:,1], label='Text file f0')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Frequency (Hz)')
        # plt.title('Pitch Tracking')
        # plt.legend()
        # plt.show()


        # call cent error function
        cent_error = eval_pitchtrack(f0_vec, text_data[:,1])
        # print cent error
        print('Cent error for file ' + str(i+1) + ' is ' + str(cent_error) + ' cents.')

run_evaluation(mypath)