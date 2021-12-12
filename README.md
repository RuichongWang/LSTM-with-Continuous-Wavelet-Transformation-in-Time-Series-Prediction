# LSTM with Continuous Wavelet Transformation in Time Series Prediction
## ***Ruichong (Alex) Wang***
## This is my Undergraduate Final Thesis
## Thanks Prof. Ming Yi @ *Huazhong University of Science and Technology* for support and encouragement.
Inspired by the sucess of Continuous Wavelet Transformation in signal processing, this project applies CWT to transform the 1-d time series data into 2-d time-frequency data to extract a more explicit long-short term pattern. This method reduces MSE by 17.5%, averaged on three datasets.
## Repository contents
* Get_and_Merge_Data.py == Data downloading and X_y Split
* Comparison_of_FFT_STFT_and_CWT.ipynb == Comparison of common signal decomposition methods
* Training.py == simple LSTM model based on CWT signal
## Repository Details
### Get_and_Merge_Data.py
This file downloads information of all the stocks in the Chinese Stock Market and split the whole sample into two train and test data.
### Comparison_of_FFT_STFT_and_CWT.ipynb
This file construct two signals and compared the Fourier Transformation, Short-time Fourier Transformation and Continuous Wavelet Transformation.
<p align="middle">
  <img src="img/Original Signal and FTT.png" height="250"/>
  <img src="img/STFT & CWT.png" height=250"/>
</p>

### Training.py
This file construct a simple LSTM Neural Network and compared the performance of CWT signal and the original signal.
