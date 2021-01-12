# LSTM with Continuous Wavelet Transformation in Time Series Prediction
Inspired by the sucess of Continuous Wavelet Transformation in signal processing, this project applies CWT to transform the 1-d time series data into 2-d time-frequency data to extract a more explicit long-short term pattern. This method reduces MSE by 17.5%, averaged on three datasets.
## Repository contents
* Get_and_Merge_Data.py == Data downloading and X_y Split
* Comparison_of_FFT_STFT_and_CWT.ipynb == Comparison of common signal decomposition methods
* Training.py == simple LSTM model based on CWT signal
## Details
### Get_and_Merge_Data.py
This file downloads information of all the stocks in the Chinese Stock Market and split the whole sample into two train and test data.
### Comparison_of_FFT_STFT_and_CWT.ipynb
This file construct two signals and compared the Fourier Transformation, Short-time Fourier Transformation and Continuous Wavelet Transformation.
![Original Signal and FTT](https://github.com/RuichongWang/LSTM-with-Continuous-Wavelet-Transformation-in-Time-Series-Prediction/blob/main/img/Original%20Signal%20and%20FTT.png?raw=true)
![STFT & CWT](https://github.com/RuichongWang/LSTM-with-Continuous-Wavelet-Transformation-in-Time-Series-Prediction/blob/main/img/STFT%20%26%20CWT.png?raw=true)
### Training.py
This file construct a simple LSTM Neural Network and compared the performance of CWT signal and the original signal.
