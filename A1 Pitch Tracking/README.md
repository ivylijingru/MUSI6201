# This folder includes code for the first assignemt of MUSI6201 - Fall 2023
This assignment implements a Pitch Tracker using Autocorrelation Function

# The track_pitch_acf.py script file includes the following methods:
block_audio
comp_acf
get_f0_from_acf

This file also generates the 2 second test signal, computes ACF based pitch tracking of the signal and plots the result.

The convert_freq2midi and eval_pitchtrack functions perform conversion from Hz to MIDI pitch and compare calculated pitch to ground truth, respectively.
The run_evaluation.py script when run calls these two functions to calculate RMS error in pitch tracking and prints output.

The error in cents is ~ 500, 700, and 700 for the three files. The errors mainly arise at certain transitional blocks where the pitch tracking method picks out 
a really high pitch by error. It is mostly very accurate for 95% of all track durations - you can uncomment the plots in the eval_pitchtrack function file to see the plots

