## E. Evaluation
### 1. Why does the HPS method fail with this signal?
F0 curve for HPS & FFT, block size 1024

![f0_curve](../figures/Estimated%20Pitch%20-%20Block%201024.png)

Error for HPS & FFT

![error](../figures/Error%20-%20Block%201024.png)

The signal has its fundamental frequency only, without harmonics. As a result, when multiplied with the downsampled spectrogram, the detected frequency will become smaller thus will fail.

### 2. Do you see any improvement in performance?
F0 curve for HPS & FFT, block size 2048

![f0_curve](../figures/Estimated%20Pitch%20-%20Block%202048.png)

Error for HPS & FFT

![error](../figures/Error%20-%20Block%202048.png)

Yes, this is because the frequency resolution (fs / blocksize) become finer when blocksize is two times of the original.

### 3. Report the average performance metrics across the development set for track_pitch_fftmax()

### 4. Report the average performance metrics across the development set for track_pitch_hps()

### 5.  report the results with two values of threshold (threshold = -40, -20)

For the above 3 questions:

|           | Error Cent RMS | False Positive Percentage | False Negative Percentage |
|-----------|----------------|---------------------------|---------------------------|
| FFT Results       | 1694.3987780830735 | 0.5625 | 0.0015120967741935483 |
| HPS Results       | 394.9651522198153  | 0.14166666666666666 | 0.018145161290322582 |
| FFT Results -20   | 1632.6219028948951 | 0.004166666666666667 | 0.42086693548387094 |
| FFT Results -40   | 1695.0905054489017 | 0.10833333333333334 | 0.003528225806451613 |
| HPS Results -20   | 375.7397382122289  | 0.004166666666666667 | 0.4213709677419355 |
| HPS Results -40   | 392.0980423933154  | 0.09166666666666666 | 0.01965725806451613 |
| ACF Results -20   | 80.43031306904199  | 0.004166666666666667 | 0.42086693548387094 |
| ACF Results -40   | 200.91268776740975 | 0.12083333333333333 | 0.0025201612903225806 |
