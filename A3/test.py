import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os

from a1solution import convert_freq2midi, track_pitch_acfmod
from a2solution import block_audio

# from A2.a2solution import extract_rms


def extract_rms(xb):
    nblocks, b_size = np.shape(xb)  # get size of input blocked signal
    rms_dB = np.zeros(nblocks)  # spectral centroid
    for i in range(nblocks):
        rms_of_block = np.sqrt(np.sum(np.square(xb[i, :]) / np.size(xb[i, :])))
        rms_dB[i] = np.maximum(
            20 * np.log10(rms_of_block / np.power(2, 15)), -100
        )  # Dividing by 2^15 (for 16-bit depth) to get dbFS
    return rms_dB


def run_evaluation(complete_path_to_data_folder, method, voicingThres=None):
    import os

    # init
    errCentRmsAvg = 0
    pfpAvg = 0
    pfnAvg = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = wav.read(
                os.path.join(complete_path_to_data_folder, file)
            )

            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(
                os.path.join(
                    complete_path_to_data_folder,
                    os.path.splitext(file)[0] + ".f0.Corrected.txt",
                )
            )
        else:
            continue

        blockSize = 1024
        hopSize = 512
        xb, timeInSec = block_audio(afAudioData, blockSize, hopSize, fs)
        rms_db = extract_rms(xb)
        for i in range(0, rms_db.shape[0]):
            print(rms_db[i])
            if i > 10:
                break
        # mask = create_voicing_mask(rms_db, voicingThres)
        # f0_adj = apply_voicing_mask(f0, mask)


if __name__ == "__main__":
    run_evaluation(
        os.path.join("../trainData"),
        track_pitch_acfmod,
        voicingThres=-30,
    )
