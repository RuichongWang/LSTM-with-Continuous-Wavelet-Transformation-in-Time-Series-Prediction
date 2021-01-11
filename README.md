# LSTM-with-Continuous-Wavelet-Transformation-in-Time-Series-Prediction
Inspired by the sucess of Continuous Wavelet Transformation in signal processing, this project applies CWT to transform the 1-d time series data into 2-d time-frequency data to extract a more explicit long-short term pattern. This method reduces MSE by 17.5%, averaged on three datasets.

Get & Merge Data.py downloads stock information of all the stocks in the Chinese Stock Market and split the whole sample into two train and test data.

Comparison of FFT STFT & CWT.ipynb construct two signals and compared the Fourier Transformation, Short-time Fourier Transformation and Continuous Wavelet Transformation.

Training.py Construct a simple LSTM Neural Network and compared the performance of CWT signal and the original signal.

![Original Signal and FTT](https://github.com/RuichongWang/LSTM-with-Continuous-Wavelet-Transformation-in-Time-Series-Prediction/blob/main/Original%20Signal%20and%20FTT.png?raw=true)
![STFT & CWT](https://github.com/RuichongWang/LSTM-with-Continuous-Wavelet-Transformation-in-Time-Series-Prediction/blob/main/STFT%20%26%20CWT.png?raw=true)
