# Function to convert frequency in Hz to MIDI scale
import numpy as np
import matplotlib.pyplot as plt

def convert_freq2midi(freq_in_hz):
    freq_in_hz = np.array(freq_in_hz).astype(float)
    fa4 = 440 # reference frequency in hertz
    p = 69 + 12*np.log2(freq_in_hz/fa4)
    return p
