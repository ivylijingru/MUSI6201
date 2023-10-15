# -*- coding: utf-8 -*-
"""
Assignment 1 reference solution - do not share, all rights reserved
"""

import numpy as np
import math


def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)), axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb, t)


def comp_acf(inputVector, bIsNormalized=True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1

    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size - 1, afCorr.size)]

    return afCorr


def get_f0_from_acf(r, fs):
    eta_min = 1
    afDeltaCorr = np.diff(r)
    eta_tmp = np.argmax(afDeltaCorr > 0)
    eta_min = np.max([eta_min, eta_tmp])

    f = np.argmax(r[np.arange(eta_min + 1, r.size)])
    f = fs / (f + eta_min + 1)

    return f


def track_pitch_acf(x, blockSize, hopSize, fs):
    # get blocks
    [xb, t] = block_audio(x, blockSize, hopSize, fs)

    # init result
    f0 = np.zeros(xb.shape[0])
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n, :])
        f0[n] = get_f0_from_acf(r, fs)

    return (f0, t)


def convert_freq2midi(fInHz, fA4InHz=440):
    def convert_freq2midi_scalar(f, fA4InHz):
        if f <= 0:
            return 0
        else:
            return 69 + 12 * np.log2(f / fA4InHz)

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
        return convert_freq2midi_scalar(fInHz, fA4InHz)

    midi = np.zeros(fInHz.shape)
    for k, f in enumerate(fInHz):
        midi[k] = convert_freq2midi_scalar(f, fA4InHz)

    return midi


def eval_pitchtrack(estimateInHz, groundtruthInHz):
    if np.abs(groundtruthInHz).sum() <= 0:
        return 0

    # truncate longer vector
    if groundtruthInHz.size < estimateInHz.size:
        estimateInHz = estimateInHz[np.arange(0, groundtruthInHz.size)]
    elif estimateInHz.size < groundtruthInHz.size:
        groundtruthInHz = groundtruthInHz[np.arange(0, estimateInHz.size)]

    diffInCent = 100 * (
        convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz)
    )

    rms = np.sqrt(np.mean(diffInCent[groundtruthInHz != 0] ** 2))
    return rms


def run_evaluation(complete_path_to_data_folder):
    import os
    from ToolReadAudio import ToolReadAudio

    # init
    rmsAvg = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)

            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(
                complete_path_to_data_folder
                + os.path.splitext(file)[0]
                + ".f0.Corrected.txt"
            )
        else:
            continue

        # extract pitch
        [f0, t] = track_pitch_acf(afAudioData, 1024, 512, fs)

        # compute rms and accumulate
        rmsAvg += eval_pitchtrack(f0, refdata[:, 2])

    if iNumOfFiles == 0:
        return -1

    return rmsAvg / iNumOfFiles


def track_pitch_acfmod(x, blockSize, hopSize, fs):
    # get blocks
    [xb, t] = block_audio(x, blockSize, hopSize, fs)

    # init result
    f0 = np.zeros(xb.shape[0])
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n, :])
        f0[n] = get_f0_from_acf(r, fs)

    return (f0, t)
