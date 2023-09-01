# function to evaluate pitch of sample dataset

# include dependencies
import numpy as np
from convert_freq2midi import *
from matplotlib import pyplot as plt

def eval_pitchtrack(estimate_in_hz, groundtruth_in_hz):
    estimate_in_hz = np.array(estimate_in_hz) # make sure inputs are numpy arrays
    groundtruth_in_hz = np.array(groundtruth_in_hz) # make sure inputs are numpy arrays
    estimate_pitch_midi = convert_freq2midi(estimate_in_hz) # convert estimate to MIDI
    groundtruth_pitch_midi = convert_freq2midi(groundtruth_in_hz) # convert ground truth to MIDI

    # plot estimate and ground truth midi pitch - comment out if not needed
    # plt.figure(figsize=(10,4))
    # plt.plot(estimate_pitch_midi)
    # plt.plot(groundtruth_pitch_midi)
    # plt.xlabel('Block Number')
    # plt.ylabel('MIDI Pitch')
    # plt.title('Pitch Tracking')
    # plt.legend(['Estimate','Ground Truth'])
    # # show without blocking
    # plt.show(block=False)

    p_err = estimate_pitch_midi - groundtruth_pitch_midi  # error in pitch  - MIDI
    err_cent = 100*p_err
    errCentRms = np.sqrt(np.sum(np.square(err_cent))/np.size(err_cent))
    return errCentRms
