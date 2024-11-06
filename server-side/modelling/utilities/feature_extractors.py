import math
import pandas as pd
import numpy as np
# import tensorflow as tf
import pywt
from concurrent.futures import ThreadPoolExecutor
import datetime
import pandas as pd
from scipy.signal import butter, lfilter, hilbert, find_peaks
from scipy.stats import entropy, kurtosis, skew
from statsmodels.tsa.ar_model import AutoReg

def _compute_morphological_feats(data: pd.DataFrame | np.ndarray, col_to_use: str='raw_signal'):
    """
    compute morphological related features of signals

    args:
        data - 
        col_to_use - 
    """
    signal = data[col_to_use]
    skewness = skew(signal)
    kurt = kurtosis(signal)
    return skewness, kurt

def interpolate_signals(data: pd.DataFrame, sample_rate: int=128, start_time: datetime.datetime | str='01/01/1970', target_hz: int=8):
    """
    interpolates signals to a certain frequency i.e. if an
    eda signal has been recorded at 128hz it can be interpolated
    to 8hz, 16hz, etc. via downsampling or to 256hz via upsampling
    """
    
    # get number of rows of dataframe
    n_rows = data.shape[0]
    if sample_rate < 8:
        if sample_rate == 2:
            # we modify the index of the dataframe to be time values
            data.index = pd.date_range(start=start_time, periods=n_rows, freq='500ms')
        elif sample_rate == 4:
            data.index = pd.date_range(start=start_time, periods=n_rows, freq='250ms')

        # if our sample rate is 1, 3, 5, 6, or 7 which are all below 8 then
        # we convert our timestamps to that that increments to 125 milliseconds 
        # now
        data = data.resample(f'{1000 / target_hz}ms').mean()
    
    # this is if sample rate is 8 or greater than 8
    else:
        if sample_rate > 8:
            # we create a list of indeces to use as index to access parts 
            # of the dataframe which are at this point still integers
            indices = list(range(0, n_rows))
            indices_sampled = indices[0::int(sample_rate / target_hz)]
            data = data.iloc[indices_sampled]
        
        # get number of rows of dataframe which were newly sampled
        sampled_n_rows = data.shape[0]

        # set the index to be our target frequency (in hz) i.e. if our new target 
        # frequency of  the signal is 8hz then 1000 / 8 would result in 125ms, if
        # we want it to be 16 then 1000 / 16 would result in 62.5ms
        freq = f'{1000 / target_hz}ms'
        # print(freq)
        data.index = pd.date_range(start=start_time, periods=sampled_n_rows, freq=freq)

    # since data resampling and then aggregation (i.e. mean, sum, etc.) 
    # from 500ms to 125ms for instance will likely generate empty or 
    # nan values we need to also interpolate these empty values
    data = _interpolate_empty_values(data)

    return data



def _interpolate_empty_values(data: pd.DataFrame):
    """
    # since data resampling and then aggregation (i.e. mean, sum, etc.) 
    # from 500ms to 125ms for instance will likely generate empty or 
    # nan values we need to also interpolate these empty values
    """

    # instead of DataframeIndex object extract 
    # its original values
    cols = data.columns.values

    for col in cols:
        # loop through each column/feature to use to access
        # series of values in each column to interpolate thems
        data.loc[:, col] = data[col].interpolate()

    return data



def _butter_lowpass(cutoff, samp_freq, order):
    """
    defines filter characteristics of the signal to be filtered
    which will be used by butter_lowpass_filter, by utilizing
    this functions return values which are the filter coefficients
    a and b
    """

    # calculate nyquist frequency which is half the given
    # sampling frequency 
    nyq = samp_freq / 2
    normed_cutoff = cutoff / nyq

    # calculate filter coefficients
    b, a = butter(order, normed_cutoff, btype='low', analog=False)

    return b, a



def butter_lowpass_filter(data, cutoff, samp_freq, order=5):
    """
    applies a low-pass filter to the eda signals using filter
    coefficients calculated by butter_lowpass, which removes high 
    frequency noise components from signals
    """
    # call butter_lowpass to obtain filter coefficients
    b, a = _butter_lowpass(cutoff, samp_freq, order)

    # apply coefficients to signal
    filt_signal = lfilter(b, a, data)

    return filt_signal



def rejoin_data(features_per_hour_1: list[tuple], features_per_hour_2: list[tuple]):
    """
    args:
        features_per_hour_1 - is a list containing tuples containing an hour long 
        feature dataframe and its respective labels e.g.

        [(                         raw_128hz_amp  raw_128hz_1d_max  raw_128hz_1d_min  \
            1970-01-01 00:00:00.000       0.000118          0.000111          0.000000   
            1970-01-01 00:00:00.500       0.000222          0.000111          0.000111   
            1970-01-01 00:00:01.000       0.000222          0.000111          0.000111   
            ...                                ...               ...               ...   
            1970-01-01 00:59:59.500       0.000000          0.000000          0.000000   
            
                                    filt_128hz_amp  ...  first_32thofa_sec_max  \
            1970-01-01 00:00:00.000    2.197134e-07  ...               0.000000   
            1970-01-01 00:00:00.500    6.022223e-05  ...               0.000000   
            1970-01-01 00:00:01.000    2.305891e-04  ...               0.000000   
            ...                                 ...  ...                    ...   
            1970-01-01 00:59:59.500    0.000000e+00  ...               0.000000   
            
                                    second_32thofa_sec_n_coeffs_above_zero  
            1970-01-01 00:00:00.000                                     1.0  
            1970-01-01 00:00:00.500                                     0.0  
            1970-01-01 00:00:01.000                                     0.0  
            ...                                                         ...  
            1970-01-01 00:59:59.500                                     0.0  
            
            [7200 rows x 43 columns],
            0       0.0
            1       0.0
            2       0.0
                    ... 
            7199    0.0
            Length: 7200, dtype: float64), ...
        ] of the uninterpolated raw eda data

        features_per_hour_2 - same as features_per_hour_2 but is the interpolated version
        of the raw eda data i.e. 16hz version of the raw 128hz eda data which always is
        of the same length as the features_per_hour of the raw 128hz eda data
    """

    # initially create empty dataframes to populate later
    # when dataframes are merged using concatenation
    merged_features_1 = pd.DataFrame()
    merged_features_2 = pd.DataFrame()
    eda_labels = pd.Series()
    for i in range(len(features_per_hour_1)):
        merged_features_1 = pd.concat([merged_features_1, features_per_hour_1[i][0]], axis=0, ignore_index=True)
        merged_features_2 = pd.concat([merged_features_2, features_per_hour_2[i][0]], axis=0, ignore_index=True)
        eda_labels = pd.concat([eda_labels, features_per_hour_1[i][1]], ignore_index=True)

    # concatenate the final two feature dataframes into 1 
    # feature dataframe including now all features from the 
    # uninterpolated and interpolated signals i.e. 128hz and 16hz
    eda_feature_df = pd.concat([merged_features_1, merged_features_2], axis=1)

    return eda_feature_df, eda_labels



def get_time_frequency(sample_rate):
    return f'{1000 / sample_rate}ms'



def _differentiate(data):
    """
    computes the 1st and 2nd order derivative values 
    of the eda signal
    """
    
    F1_prime = (data[1:-1] + data[2:]) / 2 - data[1:-1] + data[:-2] / 2
    F2_prime = data[2:] - (2 * data[1:-1]) + data[:-2]

    return F1_prime, F2_prime



def _shannon_entropy(data):
    """
    computes the shannon entropy value of the
    given segment of an eda signal
    """

    # compute probability distribution
    probs = np.power(data, 2) / np.sum(np.power(data, 2))

    # calculate entropy by multiplying probability distribution
    # to logarithm of probability distribution of base 2
    entropy = probs * np.log2(probs)
    shannon_entropy = -1 * np.sum(entropy)

    return shannon_entropy

def _standardize_signals(data):
    """
    standardizes the given signal either using min-max 
    scaling or by z-score
    """

def _compute_stat_feats(data: pd.Series | np.ndarray, col_to_use: str):
    """
    computes the statistical features of both the 
    raw/unfiltered signal and the filtered signal
    """

    raw_signal = data[col_to_use]

    raw_1d_signal, raw_2d_signal = _differentiate(raw_signal)

    raw_max = np.max(raw_signal, axis=0)
    raw_min = np.min(raw_signal, axis=0)
    raw_amp = np.mean(raw_signal, axis=0)
    raw_median = np.median(raw_signal, axis=0)
    raw_std = np.std(raw_signal, axis=0)
    raw_range = np.max(raw_signal, axis=0) - np.min(raw_signal, axis=0)
    raw_shannon_entropy = entropy(raw_signal.value_counts())

    raw_1d_max = np.max(raw_1d_signal, axis=0)
    raw_1d_min = np.min(raw_1d_signal, axis=0)
    raw_1d_amp = np.mean(raw_1d_signal, axis=0)
    raw_1d_median = np.median(raw_1d_signal, axis=0)
    raw_1d_std = np.std(raw_1d_signal, axis=0)
    raw_1d_range = np.max(raw_1d_signal, axis=0) - np.min(raw_1d_signal, axis=0)
    raw_1d_shannon_entropy = entropy(raw_1d_signal.value_counts())
    raw_1d_max_abs = np.max(np.absolute(raw_1d_signal), axis=0)
    raw_1d_avg_abs = np.mean(np.absolute(raw_1d_signal), axis=0)

    raw_2d_max = np.max(raw_2d_signal, axis=0)
    raw_2d_min = np.min(raw_2d_signal, axis=0)
    raw_2d_amp = np.mean(raw_2d_signal, axis=0)
    raw_2d_median = np.median(raw_2d_signal, axis=0)
    raw_2d_std = np.std(raw_2d_signal, axis=0)
    raw_2d_range = np.max(raw_2d_signal, axis=0) - np.min(raw_2d_signal, axis=0)
    raw_2d_shannon_entropy = entropy(raw_2d_signal.value_counts())
    raw_2d_max_abs = np.max(np.absolute(raw_2d_signal), axis=0)
    raw_2d_avg_abs = np.mean(np.absolute(raw_1d_signal), axis=0)

    return (raw_max, raw_min, raw_amp, raw_median, raw_std, raw_range, raw_shannon_entropy,
    raw_1d_max, raw_1d_min, raw_1d_amp, raw_1d_median, raw_1d_std, raw_1d_range, raw_1d_shannon_entropy, raw_1d_max_abs, raw_1d_avg_abs, 
    raw_2d_max, raw_2d_min, raw_2d_amp, raw_2d_median, raw_2d_std, raw_2d_range, raw_2d_shannon_entropy, raw_2d_max_abs, raw_2d_avg_abs)



def _compute_ar_feats(data: pd.DataFrame | np.ndarray, col_to_use):
    """
    computes autoregressive features by training AutoReg
    from statsmodels.tsa.ar_model then obtaining sigma2
    and param attributes containing the error variance and
    all the optimized coefficients excluding the intercept

    args:
        data - is a 0.5s segment/window/epoch of a subjects
        128hz signals
    """
    
    # raw_signal = data['raw_signal']

    # train_size = 0.8
    # partition_index = int(train_size * data.shape[0])
    # train_data = data.iloc[:partition_index]
    # test_data = data.iloc[partition_index:]

    ar_model = AutoReg(data[col_to_use], lags=2)
    ar_results = ar_model.fit()

    # get all autoregressive model's optimized coefficients
    # except for the intercept or bias coefficient. here we
    # would have 2 coefficients since our lag value was 2
    ar_coeffs = ar_results.params.iloc[1:].tolist()

    # get error variance attribute from trained 
    # autoregressive model
    ar_error_var = ar_results.sigma2

    # combine ar coeffs and ar error variance
    ar_features = ar_coeffs + [ar_error_var]

    return ar_features



def _get_amp_phase(z):
    """
    Calculates amplitude and phase from a complex number.

    args:
        z (complex): A complex number representing a sample of the Hilbert-transformed signal.

    Returns:
        tuple: A tuple containing the amplitude and phase.
    """
    # recall that a complex number consists of a 
    # real number and an imaginary number
    z_imag = z.imag

    # note to avoid using potential zero as a divisor
    # we must add 1e-100 to make sure \
    
    z_real = z.real + 1e-100 if z.real == 0 else z.real

    # calculate amplitude and phase
    amp = np.abs(z_imag ** 2 + z_real ** 2)
    phase = np.arctan(z_imag / z_real)
    return amp, phase



def _compute_vfcdm_feats(data: pd.DataFrame | np.ndarray, col_to_use: str, hertz: int):
    """
    computes time-frequency based features based on variable
    frequency complex demodulation (VFCDM), using cutoff
    frequencies of 64, 48, 32, and 16 hertz on the 128hz
    signal

    args:
        data - is a 0.5s segment/window/epoch of a subjects
        128hz signals
    """

    # these cutoffs namely 0.5, 0.375, 0.25, and 0.125 when
    # multiplied to 128 results in 64, 48, 32, and 16. 
    # This will basically just cutoff the signal by 50%, 37.5%,
    # 25%, and 12.5%
    cutoffs = [4 / 8, 3 / 8, 2 / 8, 1 / 8]
    decomp_signals = []

    for cutoff in cutoffs:
        filtered_signal = butter_lowpass_filter(data[col_to_use], cutoff=cutoff, samp_freq=hertz)

        # hilbert signal would have same shape as filtered signal 
        # i.e. (64,) or (8,) if signal is 16hz 
        hilbert_signal = hilbert(filtered_signal)

        decomp_signal = []
        for s_t in hilbert_signal:
            amp, phase = _get_amp_phase(s_t)

            # 2 * 3.14... * 0.5 * 128 + phase
            point = amp * np.cos(2 * np.pi * cutoff * 128 + phase)
            decomp_signal.append(point)

        decomp_signals.append(decomp_signal)

    # shape of decomposed signals will be (len(cutoffs), samples_per_win_size)
    # which in this case is (4, 64)
    decomp_signals = np.array(decomp_signals)
    # print(decomp_signals.shape)

    # compute statistical features from resulting
    # decomposed signal
    vfcdm_signals_mean = np.mean(decomp_signals, axis=1).tolist()
    vfcdm_signals_std = np.std(decomp_signals, axis=1).tolist()

    return vfcdm_signals_mean, vfcdm_signals_std



def load_wavelet_data(data: pd.DataFrame | np.ndarray, col_to_use: str, hertz: int, samples_per_win_size: int):
    """
    function to create whole and half wavelet dataframes
    """
    # determine frequency
    # 128 / 8 is 16, then we use 16 as a divisor to 1
    # 16 / 8 is 2, then we use 2 as a divisor to 1
    # 8 / 8 is 1, then we use 1 as a divisor to 1
    whole_freq = int(hertz / 8)
    half_freq = int((hertz / 8) * 2)
    # print(whole_freq)
    # print(half_freq)

    # create timestamps
    timestamp_list = pd.to_datetime(data['time'].iloc[0::samples_per_win_size], unit='s')
    # print(timestamp_list)

    # 128hz with 0.5s window is to 1/16 of a second yielding timestamps of 
    # 8 hours total and 1/32 of a second yielding timestamps of 4 hours total 
    # 16hz with 0.5s window is to 1/2 of a second yielding timestamps of 
    # 8 hours total and 1/4 of a second yielding timestamps of 4 hours total 
    # and 8hz with 5s window is to 1/1 and 1/2
    whole_inc_ts = pd.date_range(start=timestamp_list[0], periods=data.shape[0], freq=f'{((1 / whole_freq) * 1000)}ms')
    half_inc_ts = pd.date_range(start=timestamp_list[0], periods=data.shape[0], freq=f'{((1 / half_freq) * 1000)}ms')

    # obtain wavelet coefficients
    coeffs = restructure_wavelets(pywt.wavedec(data[col_to_use], wavelet='haar', level=3))
    cA_1, cD_3, cD_2, cD_1 = coeffs
    n_rows_wavelet = cD_3.shape[0]

    # reshape, calculate absolute and max value of wavelet coefficients
    # such that all shapes of wavelet coefficients adhere to that of the
    # third level coefficients shape
    whole_coeff_1 = np.max(np.absolute(np.reshape(cD_1[:n_rows_wavelet * 4], newshape=(n_rows_wavelet, 4))), axis=1)
    whole_coeff_2 = np.max(np.absolute(np.reshape(cD_2[:n_rows_wavelet * 2], newshape=(n_rows_wavelet, 2))), axis=1)
    whole_coeff_3 = np.absolute(cD_3[:n_rows_wavelet])

    # create whole wave features dataframe
    whole_wave_features = pd.DataFrame({
        f'first_{whole_freq}thofa_sec_feat': whole_coeff_1,
        f'second_{whole_freq}thofa_sec_feat': whole_coeff_2,
        f'third_{whole_freq}thofa_sec_feat': whole_coeff_3,
    })

    # use whole increment timestamps as indices for the whole wave features dataframe
    whole_wave_features.set_index(whole_inc_ts[:whole_wave_features.shape[0]], inplace=True)

    # recall that cD_1 has shape of 9688 which is second_data_n_segments_half * 2
    half_coeff_1 = np.max(np.absolute(np.reshape(cD_1[:n_rows_wavelet * 4], newshape=(n_rows_wavelet * 2, 2))), axis=1)
    half_coeff_2 = np.absolute(cD_2[:n_rows_wavelet * 2])

    half_wave_features = pd.DataFrame({
        f'first_{half_freq}thofa_sec_feat': half_coeff_1,
        f'second_{half_freq}thofa_sec_feat': half_coeff_2,
    })

    half_wave_features.set_index(half_inc_ts[:half_wave_features.shape[0]], inplace=True)

    return whole_wave_features, half_wave_features
    


def restructure_wavelets(wavelet_coeffs):
    """
    takes in the calculated wavelet coefficients and
    appends zeroes if needed to match the 3rd level
    wavelet coefficients shape so it can later be reshaped
    e.g. signal data of 310004 rows yields wavelet coefficients
    from a level 3 discrete haar wavelet transform of shapes
    155007, 77504, and 38752, as we can see 155007 is odd so we
    append a zero at the end so that it has now 155008 elements
    and so it can be reshaped to 
    """

    cA_1, cD_3, cD_2, cD_1 = wavelet_coeffs

    third_lvl_shape = cD_3.shape[0]
    second_lvl_shape = cD_2.shape[0]
    first_lvl_shape = cD_1.shape[0]
    # print(third_lvl_shape)
    # print(second_lvl_shape)
    # print(first_lvl_shape)

    if (((third_lvl_shape * 2) - second_lvl_shape) > 0) or (((third_lvl_shape * 4) - first_lvl_shape) > 0) or (third_lvl_shape % 2 != 0):
        # calculate amount of zeros to append to 1st and 2nd level coefficients
        n_zeros_second_lvl = (third_lvl_shape * 2) - second_lvl_shape
        n_zeros_first_lvl = (third_lvl_shape * 4) - first_lvl_shape
        # print(n_zeros_first_lvl, n_zeros_second_lvl)

        # append the amount of calculated zeros to the 1st and 2nd level coefficients
        second_lvl_zeros = np.zeros(shape=(n_zeros_second_lvl, ))
        first_lvl_zeros = np.zeros(shape=(n_zeros_first_lvl, ))
        cD_2 = np.hstack((cD_2, second_lvl_zeros))
        cD_1 = np.hstack((cD_1, first_lvl_zeros))

    return cA_1, cD_3, cD_2, cD_1



def _compute_wave_feats(wave: pd.DataFrame | np.ndarray):
    """
    computes the maximum, mean, standard deviation, median, range
    and no. of coefficients above zero values of the given wavelet 
    dataframe
    """

    wavelet_feats_max = wave.max(axis=0).tolist()
    wavelet_feats_mean = wave.mean(axis=0).tolist()
    wavelet_feats_std = wave.std(axis=0).tolist()
    wavelet_feats_median = wave.median(axis=0).tolist()
    wavelet_feats_range = (wave.max(axis=0) - wave.min(axis=0)).tolist()
    wavelet_feats_n_coeffs_above_zero = (wave > 0).astype('int').sum(axis=0)

    return wavelet_feats_max, wavelet_feats_mean, wavelet_feats_std, wavelet_feats_median, wavelet_feats_range, wavelet_feats_n_coeffs_above_zero



def compute_features(data: pd.DataFrame | np.ndarray, whole_wave: pd.DataFrame | np.ndarray, half_wave: pd.DataFrame | np.ndarray, samples_per_sec: int):
    """
    computes the ff. features given the 0.5s segment/window dataframe or numpy matrix 'data'

    raw_max - the maximum value of the raw/unfiltered eda signal
    raw_min - the maximum value of the raw/unfiltered eda signal
    raw_amp - the average/mean value of the raw/unfiltered eda signal
    raw_median - the median value of the raw/unfiltered eda signal
    raw_std - the standard deviation value of the raw/unfiltered eda signal
    raw_range - the range value of the raw/unfiltered eda signal
    raw_shannon_entropy - the shannon entropy value of the raw/unfiltered eda signal

    raw_1d_max - the maximum value of the first order derivative of the unfiltered eda signal
    raw_1d_min - the minimum value of the first order derivative of the unfiltered eda signal
    raw_1d_amp - the average/mean value of the first order derivative of the unfiltered eda signal
    raw_1d_median - the median value of the first order derivative of the unfiltered eda signal
    raw_1d_std - the standard deviation value of the first order derivative of the unfiltered eda signal
    raw_1d_range - the range value of the first order derivative of the unfiltered eda signal
    raw_1d_shannon_entropy - the shannon entropy value of the first order derivative of the unfiltered eda signal
    raw_1d_max_abs - the maximum value out of the absolute values of the first order derivative of the unfiltered eda signal
    raw_1d_avg_abs - the average/mean value out of the absolute values of the first order derivative of the unfiltered eda signal

    raw_2d_max - the maximum value of the second order derivative of the unfiltered eda signal
    raw_2d_min - the minimum value of the second order derivative of the unfiltered eda signal
    raw_2d_amp - the average/mean value of the second order derivative of the unfiltered eda signal
    raw_2d_median - the median value of the second order derivative of the unfiltered eda signal
    raw_2d_std - the standard deviation value of the second order derivative of the unfiltered eda signal
    raw_2d_range - the range value of the second order derivative of the unfiltered eda signal
    raw_2d_shannon_entropy - the shannon entropy value of the second order derivative of the unfiltered eda signal
    raw_2d_max_abs - the maximum value out of the absolute values of the second order derivative of the unfiltered eda signal
    raw_2d_avg_abs - the average/mean value out of the absolute values of the second order derivative of the unfiltered eda signal
    
    filt_max - the maximum value of the low-pass filtered eda signal
    filt_min - the maximum value of the low-pass filtered eda signal
    filt_amp - the average/mean value of the low-pass filtered eda signal
    filt_median - the median value of the low-pass filtered eda signal
    filt_std - the standard deviation value of the low-pass filtered eda signal
    filt_range - the range value of the low-pass filtered eda signal
    filt_shannon_entropy - the shannon entropy value of the low-pass filtered eda signal

    filt_1d_max - the maximum value of the first order derivative of the low-pass filtered eda signal
    filt_1d_min - the minimum value of the first order derivative of the low-pass filtered eda signal
    filt_1d_amp - the average/mean value of the first order derivative of the low-pass filtered eda signal
    filt_1d_median - the median value of the first order derivative of the low-pass filtered eda signal
    filt_1d_std - the standard deviation value of the first order derivative of the low-pass filtered eda signal
    filt_1d_range - the range value of the first order derivative of the low-pass filtered eda signal
    filt_1d_shannon_entropy - the shannon entropy value of the first order derivative of the low-pass filtered eda signal
    filt_1d_max_abs - the maximum value out of the absolute values of the first order derivative of the low-pass filtered eda signal
    filt_1d_avg_abs - the average/mean value out of the absolute values of the first order derivative of the low-pass filtered eda signal

    filt_2d_max - the maximum value of the second order derivative of the low-pass filtered eda signal
    filt_2d_min - the minimum value of the second order derivative of the low-pass filtered eda signal
    filt_2d_amp - the average/mean value of the second order derivative of the low-pass filtered eda signal
    filt_2d_median - the median value of the second order derivative of the low-pass filtered eda signal
    filt_2d_std - the standard deviation value of the second order derivative of the low-pass filtered eda signal
    filt_2d_range - the range value of the second order derivative of the low-pass filtered eda signal
    filt_2d_shannon_entropy - the shannon entropy value of the second order derivative of the low-pass filtered eda signal
    filt_2d_max_abs - the maximum value out of the absolute values of the second order derivative of the low-pass filtered eda signal
    filt_2d_avg_abs - the average/mean value out of the absolute values of the second order derivative of the low-pass filtered eda signal
    
    ar_coeffs - coefficients excluding bias/intercept coefficient of the trained autoregressive model
    ar_err_var - error variance value of the trained autoregressive model

    vfcdm_4/8_cut_mean - 
    vfcdm_3/8_cut_mean - 
    vfcdm_2/8_cut_mean - 
    vfcdm_1/8_cut_mean - 
    vfcdm_4/8_cut_std - 
    vfcdm_3/8_cut_std - 
    vfcdm_2/8_cut_std - 
    vfcdm_1/8_cut_std - 

    first_whole_max - the maximum value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_whole_max - the maximum value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_whole_max - the maximum value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_whole_mean - the average/mean value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_whole_mean - the average/mean value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_whole_mean  - the average/mean value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_whole_std - the standard deviation of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_whole_std - the standard deviation of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_whole_std - the standard deviation of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_whole_median - the median value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_whole_median - the median value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_whole_median - the median value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_whole_range - the median value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_whole_range - the range value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_whole_range - the range value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_whole_n_coeffs_above_zero - the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_whole_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_whole_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe

    first_half_max - the maximum value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half wavelet dataframe
    second_half_max - the maximum value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half wavelet dataframe
    first_half_mean - the average/mean value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half wavelet dataframe
    second_half_mean - the average/mean value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half wavelet dataframe
    first_half_std - the standard deviation of the 0.5s segment/window/epoch from the 1st wavelet feature of the half wavelet dataframe
    second_half_std - the standard deviation of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half wavelet dataframe
    first_half_median - the median value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half wavelet dataframe
    second_half_median - the median value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half wavelet dataframe
    first_half_range - the range value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half wavelet dataframe
    second_half_range - the range value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half wavelet dataframe
    first_half_n_coeffs_above_zero - the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 1st wavelet feature of the half wavelet dataframe
    second_half_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half wavelet dataframe

    args:
        data - data slice/segment/window of 0.5s 
        whole_wave - a dataframe containing whole of the wavelet coefficients 
        half_wave - a dataframe containing half the time of the whole wavelet coefficients
    """

    # compute statistical features
    raw_max, raw_min, raw_amp, raw_median, raw_std, raw_range, raw_shannon_entropy, \
    raw_1d_max, raw_1d_min, raw_1d_amp, raw_1d_median, raw_1d_std, raw_1d_range, raw_1d_shannon_entropy, raw_1d_max_abs, raw_1d_avg_abs, \
    raw_2d_max, raw_2d_min, raw_2d_amp, raw_2d_median, raw_2d_std, raw_2d_range, raw_2d_shannon_entropy, raw_2d_max_abs, raw_2d_avg_abs, = _compute_stat_feats(data, col_to_use="raw_signal")
    filt_max, filt_min, filt_amp, filt_median, filt_std, filt_range, filt_shannon_entropy, \
    filt_1d_max, filt_1d_min, filt_1d_amp, filt_1d_median, filt_1d_std, filt_1d_range, filt_1d_shannon_entropy, filt_1d_max_abs, filt_1d_avg_abs, \
    filt_2d_max, filt_2d_min, filt_2d_amp, filt_2d_median, filt_2d_std, filt_2d_range, filt_2d_shannon_entropy, filt_2d_max_abs, filt_2d_avg_abs = _compute_stat_feats(data, col_to_use="filtered_signal")

    # compute autoregressive features
    *ar_coeffs, ar_err_var = _compute_ar_feats(data, col_to_use="raw_signal")
    # ar_feats = _compute_ar_feats(data)
    # ar_coeffs, ar_err_var = ar_feats[:-1], ar_feats[-1]

    # compute VFCDM features
    vfcdm_signals_mean, vfcdm_signals_std = _compute_vfcdm_feats(data, col_to_use="raw_signal", hertz=samples_per_sec)

    # compute morphological features
    raw_skewness, raw_kurt = _compute_morphological_feats(data, col_to_use='raw_signal')
    filt_skewness, filt_kurt = _compute_morphological_feats(data, col_to_use='filtered_signal')

    # compute wavelet features
    wavelet_feats_whole_max, wavelet_feats_whole_mean, wavelet_feats_whole_std, wavelet_feats_whole_median, wavelet_feats_whole_range, wavelet_feats_whole_n_coeffs_above_zero = _compute_wave_feats(whole_wave)
    wavelet_feats_half_max, wavelet_feats_half_mean, wavelet_feats_half_std, wavelet_feats_half_median, wavelet_feats_half_range, wavelet_feats_half_n_coeffs_above_zero = _compute_wave_feats(half_wave)

    

    features = np.hstack([
        # statistical features
        raw_max, raw_min, raw_amp, raw_median, raw_std, raw_range, raw_shannon_entropy, 
        raw_1d_max, raw_1d_min, raw_1d_amp, raw_1d_median, raw_1d_std, raw_1d_range, raw_1d_shannon_entropy, raw_1d_max_abs, raw_1d_avg_abs, 
        raw_2d_max, raw_2d_min, raw_2d_amp, raw_2d_median, raw_2d_std, raw_2d_range, raw_2d_shannon_entropy, raw_2d_max_abs, raw_2d_avg_abs, 
        filt_max, filt_min, filt_amp, filt_median, filt_std, filt_range, filt_shannon_entropy, 
        filt_1d_max, filt_1d_min, filt_1d_amp, filt_1d_median, filt_1d_std, filt_1d_range, filt_1d_shannon_entropy, filt_1d_max_abs, filt_1d_avg_abs, 
        filt_2d_max, filt_2d_min, filt_2d_amp, filt_2d_median, filt_2d_std, filt_2d_range, filt_2d_shannon_entropy, filt_2d_max_abs, filt_2d_avg_abs,

        # autoregressive coefficients excluding bias/intercept and error variance
        ar_coeffs, ar_err_var,

        # vfcdm signals means and standard deviations at cutoffs of 50%, 37.5%, 25%, 12.5%
        vfcdm_signals_mean, vfcdm_signals_std,

        # morphological features
        raw_skewness, raw_kurt,
        filt_skewness, filt_kurt,

        # each wavelet is a list of 3 elements since it was calculated from a dataframe of 3 columns
        wavelet_feats_whole_max, wavelet_feats_whole_mean, wavelet_feats_whole_std, wavelet_feats_whole_median, wavelet_feats_whole_range, wavelet_feats_whole_n_coeffs_above_zero, 

        # each wavelet is a list of 2 elements since it was calculated from a dataframe of 2 columns
        wavelet_feats_half_max, wavelet_feats_half_mean, wavelet_feats_half_std, wavelet_feats_half_median, wavelet_feats_half_range, wavelet_feats_half_n_coeffs_above_zero,
        ])
    
    return features

def get_features(data: pd.DataFrame, data_slice: pd.DataFrame | np.ndarray, whole_wave: pd.DataFrame | np.ndarray, half_wave: pd.DataFrame | np.ndarray, samples_per_sec: int, samples_per_win_size: int):
    """
    creates and returns a dataframe containing all 
    the features for a data slice of at least 1 hour
    """

    whole_freq = int(samples_per_sec / 8)
    half_freq = int((samples_per_sec / 8) * 2)

    feature_names = [
        # statistical features names
        f"raw_{samples_per_sec}hz_max", f"raw_{samples_per_sec}hz_min", 
        f"raw_{samples_per_sec}hz_amp", f"raw_{samples_per_sec}hz_median", 
        f"raw_{samples_per_sec}hz_std", f"raw_{samples_per_sec}hz_range", 
        f"raw_{samples_per_sec}hz_shannon_entropy",
        
        f"raw_{samples_per_sec}hz_1d_max", f"raw_{samples_per_sec}hz_1d_min", 
        f"raw_{samples_per_sec}hz_1d_amp", f"raw_{samples_per_sec}hz_1d_median", 
        f"raw_{samples_per_sec}hz_1d_std", f"raw_{samples_per_sec}hz_1d_range", 
        f"raw_{samples_per_sec}hz_1d_shannon_entropy",
        f"raw_{samples_per_sec}hz_1d_max_abs", f"raw_{samples_per_sec}hz_1d_avg_abs", 

        f"raw_{samples_per_sec}hz_2d_max", f"raw_{samples_per_sec}hz_2d_min", 
        f"raw_{samples_per_sec}hz_2d_amp", f"raw_{samples_per_sec}hz_2d_median", 
        f"raw_{samples_per_sec}hz_2d_std", f"raw_{samples_per_sec}hz_2d_range", 
        f"raw_{samples_per_sec}hz_2d_shannon_entropy",
        f"raw_{samples_per_sec}hz_2d_max_abs", f"raw_{samples_per_sec}hz_2d_avg_abs", 

        f"filt_{samples_per_sec}hz_max", f"filt_{samples_per_sec}hz_min", 
        f"filt_{samples_per_sec}hz_amp", f"filt_{samples_per_sec}hz_median", 
        f"filt_{samples_per_sec}hz_std", f"filt_{samples_per_sec}hz_range", 
        f"filt_{samples_per_sec}hz_shannon_entropy",
        
        f"filt_{samples_per_sec}hz_1d_max", f"filt_{samples_per_sec}hz_1d_min", 
        f"filt_{samples_per_sec}hz_1d_amp", f"filt_{samples_per_sec}hz_1d_median", 
        f"filt_{samples_per_sec}hz_1d_std", f"filt_{samples_per_sec}hz_1d_range", 
        f"filt_{samples_per_sec}hz_1d_shannon_entropy",
        f"filt_{samples_per_sec}hz_1d_max_abs", f"filt_{samples_per_sec}hz_1d_avg_abs", 

        f"filt_{samples_per_sec}hz_2d_max", f"filt_{samples_per_sec}hz_2d_min", 
        f"filt_{samples_per_sec}hz_2d_amp", f"filt_{samples_per_sec}hz_2d_median", 
        f"filt_{samples_per_sec}hz_2d_std", f"filt_{samples_per_sec}hz_2d_range", 
        f"filt_{samples_per_sec}hz_2d_shannon_entropy",
        f"filt_{samples_per_sec}hz_2d_max_abs", f"filt_{samples_per_sec}hz_2d_avg_abs", 

        # autoregressive features names
        f"ar_coeff_1_{samples_per_sec}hz", f"ar_coeff_2_{samples_per_sec}hz", f"ar_err_var_{samples_per_sec}hz",

        # vfcdm features names
        f"vfcdm_4/8_{samples_per_sec}hz_mean", f"vfcdm_3/8_cut_{samples_per_sec}hz_mean", f"vfcdm_2/8_cut_{samples_per_sec}hz_mean", f"vfcdm_1/8_cut_{samples_per_sec}hz_mean", 
        f"vfcdm_4/8_{samples_per_sec}hz_std", f"vfcdm_3/8_cut_{samples_per_sec}hz_std", f"vfcdm_2/8_cut_{samples_per_sec}hz_std", f"vfcdm_1/8_cut_{samples_per_sec}hz_std",

        f"raw_{samples_per_sec}hz_skewness", f"raw_{samples_per_sec}hz_kurt",
        f"filt_{samples_per_sec}hz_skewness", f"filt_{samples_per_sec}hz_kurt",

        # wavelet features names
        f"first_{whole_freq}thofa_sec_max", f"second_{whole_freq}thofa_sec_max", f"third_{whole_freq}thofa_sec_max", 
        f"first_{whole_freq}thofa_sec_mean", f"second_{whole_freq}thofa_sec_mean", f"third_{whole_freq}thofa_sec_mean", 
        f"first_{whole_freq}thofa_sec_std", f"second_{whole_freq}thofa_sec_std", f"third_{whole_freq}thofa_sec_std", 
        f"first_{whole_freq}thofa_sec_median", f"second_{whole_freq}thofa_sec_median", f"third_{whole_freq}thofa_sec_median",
        f"first_{whole_freq}thofa_sec_range", f"second_{whole_freq}thofa_sec_range", f"third_{whole_freq}thofa_sec_range",
        f"first_{whole_freq}thofa_sec_n_coeffs_above_zero", f"second_{whole_freq}thofa_sec_n_coeffs_above_zero", f"third_{whole_freq}thofa_sec_n_coeffs_above_zero",

        f"first_{half_freq}thofa_sec_max", f"second_{half_freq}thofa_sec_max", 
        f"first_{half_freq}thofa_sec_mean", f"second_{half_freq}thofa_sec_mean", 
        f"first_{half_freq}thofa_sec_std", f"second_{half_freq}thofa_sec_std", 
        f"first_{half_freq}thofa_sec_median", f"second_{half_freq}thofa_sec_median", 
        f"first_{half_freq}thofa_sec_range", f"second_{half_freq}thofa_sec_range", 
        f"first_{half_freq}thofa_sec_n_coeffs_above_zero", f"second_{half_freq}thofa_sec_n_coeffs_above_zero"
    ]

    feature_names_len = len(feature_names)

    # create a list of timestamps from our data that 
    # has step size samples_per_win_size i.e. if our 
    # 128hz data increments by 7.8125s and takes 128 
    # samples to get to 1 full second, then a 64 step
    # size will get only a row of data every 64 rows
    # thus effectively resulting in a timestamp list that
    # increments by 0.5s
    timestamp_list = data_slice.index.tolist()[::samples_per_win_size]
    # timestamp list is goes from 0 to 63, 64 to 127, and so on
    print(f'last timestamp: {timestamp_list[-1]}')
    
    # this includes one last timestamp:
    # the exclusion timestamp index
    exc_ts_index = data.index.get_loc(timestamp_list[-1])
    exc_ts = data.index[exc_ts_index:exc_ts_index + samples_per_win_size]
    # exc goes from '1970-01-01 00:59:59.500000' to '1970-01-01 00:59:59.992187500'
    print(f'exclusion timestamp list: {exc_ts} {exc_ts.shape}')

    timestamp_list_len = len(timestamp_list)
    print(f'hour long timestamp list length: {timestamp_list_len}')

    # initially create empty feature_segments dataframe of zeros
    # this will be of length 7200 because of the length of timestamps
    feature_segments = pd.DataFrame(np.zeros(shape=(timestamp_list_len, feature_names_len)), columns=feature_names, index=timestamp_list)
    labels = pd.Series(np.zeros(shape=(timestamp_list_len)))
    for i in range(timestamp_list_len - 1):
    # for i in range(timestamp_list_len): # """SAFETY TEST"""
        # get start time, end time, and both its respective indeces in the
        # dataframe to use for artifact correction later as these mappings
        # from the timestamp to the created feature will be of paramount
        # importance 

        # because there is a i + 1 we are forced to cut short the loop
        # only from [0] until [7198] since [7199 + 1] would be out of bounds
        # and [7198 + 1] is still permitted
        start_time = timestamp_list[i]
        end_time = timestamp_list[i + 1]

        data_segment = data_slice[start_time:end_time].iloc[:-1]
        whole_wave_segment = whole_wave[start_time:end_time].iloc[:-1]
        half_wave_segment = half_wave[start_time:end_time].iloc[:-1]

        start_time_index = data.index.get_loc(start_time)
        end_time_index = data.index.get_loc(data_segment.index.tolist()[-1])

        if i == 0 or i == 1 or i == (timestamp_list_len - 3) or i == (timestamp_list_len - 2):
            print(f'calculating features from {start_time} to {end_time} exclusively for index {i}')
            print(f'calculating features from {start_time_index} to {end_time_index} inclusively for index {i}')
            print(f'data segment: \n{data_segment}')

        # compute the features for each 0.5s segment and assign to
        # its respective index in the empty dataframe
        feature_segment = compute_features(data_segment, whole_wave_segment, half_wave_segment, samples_per_sec)
        feature_segments.iloc[i] = feature_segment

        # returns the mean of a list or matrix of values given an axis ignoring 
        # any nan values. Here according to Llanes-Jurado et al. (2023)'s paper 
        # if more than 50% of the segment was labeled as an artifact, such a
        # segment of 0.5 s was labeled indeed as an artifact
        labels[i] = 1 if np.nanmean(data_segment['label']) > 0.5 else 0

    # in theory we would have to still extract the features at timestamp_list[7199]
    # to timestamp_list[7200] as timestamp_list[7200] would be a timestamp that will
    # only be used as an exclusive time value and not used

    # fill any potential null or nan value with zeros
    feature_segments.fillna(0, inplace=True)

    return feature_segments, labels



def extract_features_per_hour(
    data: pd.DataFrame | np.ndarray, 
    hertz: int=128, 
    window_time: float | int=1, 
    verbose: bool=False) -> list[tuple]:
    """
    partitions the given dataframe of eda data into at least 1 hour 
    and extracts wavelet and statistical features in each of these
    hours

    args:
        data - dataframe consisting of the raw unfiltered signal, labels, and filtered signal
        hertz - sampling rate of the data
        window_size - amount of seconds for each segmented signal
    """

    # note this samples per sec is not arbitrary and a 
    # user defined value but derived from our frequency 
    # value entirely i.e. because we've recorded our
    # raw data at 128hz then that means that the 
    # samples of data we have per second would be 128
    samples_per_sec = hertz
    secs_per_min = 60
    min_per_hour = 60

    # here we would be calculating how many samples we would have 
    # per hour given we have 128 samples per second
    samples_per_hour = samples_per_sec * secs_per_min * min_per_hour

    # we also need to specify how large our windows/epochs/segments
    # would be in order to create the rows for our dataset and subsequently
    # each feature of that window or row
    samples_per_win_size = int(samples_per_sec * window_time)

    # get number of rows of 128hz timestamps and signals
    n_rows = data.shape[0]

    # dividing the number of rows by the number of samples per 0.5 seconds 
    # will allow us to get a sense how many segments or rows of 0.5 seconds
    # can we get from this time series dataframe of our signals
    num_labels = math.ceil(n_rows / samples_per_win_size)
    print(f'num_labels: {num_labels}')
    
    hours = math.ceil(n_rows / samples_per_hour)

    features_per_hour = []
    for hour in range(hours):
        start = hour * samples_per_hour
        end = min((hour + 1) * samples_per_hour, n_rows)
        curr_data = data.iloc[start:end]

        # if 128 hertz data samples per sec will be 128 and samples per window size
        # will be as specified which in this case must be 64
        # this will create wavelet dataframe for the current data slice of 1 hour 
        whole_wave, half_wave = load_wavelet_data(curr_data, col_to_use="raw_signal", hertz=samples_per_sec, samples_per_win_size=samples_per_win_size)

        # samples per sec 128 samples per win size 64
        # samples per sec 16 samples per win size 8
        features_per_hour.append(get_features(data, curr_data, whole_wave, half_wave, samples_per_sec, samples_per_win_size))

        if verbose == True:
            print(f'processed hour {hour} - start: {start} | end: {end}')
            print("whole wave: ")
            print(whole_wave)
            print("half wave: ")
            print(half_wave)

    return features_per_hour

def load_wavelet_data_hybrid(data: pd.DataFrame | np.ndarray, col_to_use: str, hertz: int):
    """
    function to create whole and half wavelet dataframes
    """

    # determine frequency
    # 128 / 8 is 16, then we use 16 as a divisor to 1
    # 16 / 8 is 2, then we use 2 as a divisor to 1
    # 8 / 8 is 1, then we use 1 as a divisor to 1
    whole_freq = int(hertz / 8)
    half_freq = int((hertz / 8) * 2)

    # obtain wavelet coefficients
    coeffs = restructure_wavelets(pywt.wavedec(data[col_to_use], wavelet='haar', level=3))
    cA_1, cD_3, cD_2, cD_1 = coeffs
    n_rows_wavelet = cD_3.shape[0]

    # reshape, calculate absolute and max value of wavelet coefficients
    # such that all shapes of wavelet coefficients adhere to that of the
    # third level coefficients shape
    whole_coeff_1 = np.max(np.absolute(np.reshape(cD_1[:n_rows_wavelet * 4], newshape=(n_rows_wavelet, 4))), axis=1)
    whole_coeff_2 = np.max(np.absolute(np.reshape(cD_2[:n_rows_wavelet * 2], newshape=(n_rows_wavelet, 2))), axis=1)
    whole_coeff_3 = np.absolute(cD_3[:n_rows_wavelet])

    # create whole wave features dataframe
    whole_wave_features = pd.DataFrame({
        f'first_{whole_freq}thofa_sec_feat': whole_coeff_1,
        f'second_{whole_freq}thofa_sec_feat': whole_coeff_2,
        f'third_{whole_freq}thofa_sec_feat': whole_coeff_3,
    })

    # recall that cD_1 has shape of 9688 which is second_data_n_segments_half * 2
    half_coeff_1 = np.max(np.absolute(np.reshape(cD_1[:n_rows_wavelet * 4], newshape=(n_rows_wavelet * 2, 2))), axis=1)
    half_coeff_2 = np.absolute(cD_2[:n_rows_wavelet * 2])

    half_wave_features = pd.DataFrame({
        f'first_{half_freq}thofa_sec_feat': half_coeff_1,
        f'second_{half_freq}thofa_sec_feat': half_coeff_2,
    })

    return whole_wave_features, half_wave_features

    

def extract_features_hybrid(data: pd.DataFrame | np.ndarray, 
    hertz: int=128, 
    window_time: float | int=5, 
    x_col="raw_signal", 
    target_time=0.5, 
    y_col=None, 
    scale=False,
    verbose: bool=False) -> list[tuple[pd.DataFrame, pd.Series]] | list[tuple[pd.DataFrame]]:
    """
    modified version of charge_raw_data() where each window of the signal is used in order to
    calculate ML based/lower order features from the 5 second window of the signal. Like
    charge_raw_data() where in case a target is introduced i.e. y_col != None, the target is 
    cut the last 0.5 seconds of the binary target, becoming the target of the corresponding 5
    seconds segement.
    """

    whole_freq = int(hertz / 8)
    half_freq = int((hertz / 8) * 2)

    feature_names = [
        # statistical features names
        f"raw_{hertz}hz_max", f"raw_{hertz}hz_min", 
        f"raw_{hertz}hz_amp", f"raw_{hertz}hz_median", 
        f"raw_{hertz}hz_std", f"raw_{hertz}hz_range", 
        f"raw_{hertz}hz_shannon_entropy",
        
        f"raw_{hertz}hz_1d_max", f"raw_{hertz}hz_1d_min", 
        f"raw_{hertz}hz_1d_amp", f"raw_{hertz}hz_1d_median", 
        f"raw_{hertz}hz_1d_std", f"raw_{hertz}hz_1d_range", 
        f"raw_{hertz}hz_1d_shannon_entropy",
        f"raw_{hertz}hz_1d_max_abs", f"raw_{hertz}hz_1d_avg_abs", 

        f"raw_{hertz}hz_2d_max", f"raw_{hertz}hz_2d_min", 
        f"raw_{hertz}hz_2d_amp", f"raw_{hertz}hz_2d_median", 
        f"raw_{hertz}hz_2d_std", f"raw_{hertz}hz_2d_range", 
        f"raw_{hertz}hz_2d_shannon_entropy",
        f"raw_{hertz}hz_2d_max_abs", f"raw_{hertz}hz_2d_avg_abs", 

        f"filt_{hertz}hz_max", f"filt_{hertz}hz_min", 
        f"filt_{hertz}hz_amp", f"filt_{hertz}hz_median", 
        f"filt_{hertz}hz_std", f"filt_{hertz}hz_range", 
        f"filt_{hertz}hz_shannon_entropy",
        
        f"filt_{hertz}hz_1d_max", f"filt_{hertz}hz_1d_min", 
        f"filt_{hertz}hz_1d_amp", f"filt_{hertz}hz_1d_median", 
        f"filt_{hertz}hz_1d_std", f"filt_{hertz}hz_1d_range", 
        f"filt_{hertz}hz_1d_shannon_entropy",
        f"filt_{hertz}hz_1d_max_abs", f"filt_{hertz}hz_1d_avg_abs", 

        f"filt_{hertz}hz_2d_max", f"filt_{hertz}hz_2d_min", 
        f"filt_{hertz}hz_2d_amp", f"filt_{hertz}hz_2d_median", 
        f"filt_{hertz}hz_2d_std", f"filt_{hertz}hz_2d_range", 
        f"filt_{hertz}hz_2d_shannon_entropy",
        f"filt_{hertz}hz_2d_max_abs", f"filt_{hertz}hz_2d_avg_abs", 

        # autoregressive features names
        f"ar_coeff_1_{hertz}hz", f"ar_coeff_2_{hertz}hz", f"ar_err_var_{hertz}hz",

        # vfcdm features names
        f"vfcdm_4/8_{hertz}hz_mean", f"vfcdm_3/8_cut_{hertz}hz_mean", f"vfcdm_2/8_cut_{hertz}hz_mean", f"vfcdm_1/8_cut_{hertz}hz_mean", 
        f"vfcdm_4/8_{hertz}hz_std", f"vfcdm_3/8_cut_{hertz}hz_std", f"vfcdm_2/8_cut_{hertz}hz_std", f"vfcdm_1/8_cut_{hertz}hz_std",

        # morphological features names
        f"raw_{hertz}hz_skewness", f"raw_{hertz}hz_kurt",
        f"filt_{hertz}hz_skewness", f"filt_{hertz}hz_kurt",

        # wavelet features names
        f"first_{whole_freq}thofa_sec_max", f"second_{whole_freq}thofa_sec_max", f"third_{whole_freq}thofa_sec_max", 
        f"first_{whole_freq}thofa_sec_mean", f"second_{whole_freq}thofa_sec_mean", f"third_{whole_freq}thofa_sec_mean", 
        f"first_{whole_freq}thofa_sec_std", f"second_{whole_freq}thofa_sec_std", f"third_{whole_freq}thofa_sec_std", 
        f"first_{whole_freq}thofa_sec_median", f"second_{whole_freq}thofa_sec_median", f"third_{whole_freq}thofa_sec_median",
        f"first_{whole_freq}thofa_sec_range", f"second_{whole_freq}thofa_sec_range", f"third_{whole_freq}thofa_sec_range",
        f"first_{whole_freq}thofa_sec_n_coeffs_above_zero", f"second_{whole_freq}thofa_sec_n_coeffs_above_zero", f"third_{whole_freq}thofa_sec_n_coeffs_above_zero",

        f"first_{half_freq}thofa_sec_max", f"second_{half_freq}thofa_sec_max", 
        f"first_{half_freq}thofa_sec_mean", f"second_{half_freq}thofa_sec_mean", 
        f"first_{half_freq}thofa_sec_std", f"second_{half_freq}thofa_sec_std", 
        f"first_{half_freq}thofa_sec_median", f"second_{half_freq}thofa_sec_median", 
        f"first_{half_freq}thofa_sec_range", f"second_{half_freq}thofa_sec_range", 
        f"first_{half_freq}thofa_sec_n_coeffs_above_zero", f"second_{half_freq}thofa_sec_n_coeffs_above_zero"
    ]

    # create empty list  that will be populated during
    # computation of the features of the window of the specific
    # slice of the signal
    feature_segments_list = []

    # we access the SCR values via raw data column
    x_signals = data[x_col].values

    # here if we would want to create windows of the raw data including the target label
    # we must specify which target label we want to include since there are multiple columns
    # that pertain to the label I vbelieve which are: binary_target, predicted artifacts
    # and post processed artifacts
    if y_col is not None:
        y_signals = data[y_col].values

    window_size = int(window_time * hertz)
    target_size = int(target_time * hertz)

    x_window_list, y_window_list = [], []

    ctr = 0
    # so if we have a length of 765045 rows for the raw eda data
    # and in each row we'd have to multiply 128 to get specific seconds e.g.
    # to get 0th second we multiply 128 by 0 and use it as index
    # raw_eda_df['time'].iloc[:128 * 0], to get 1st second mark we'd have
    # to multiply 128 by 1 and use it as index raw_eda_df['time'].iloc[:128 * 1]

    # but what is the point of subtracting 765045 by window size of 640 (5 * 128)?
    print(f'length of x_signals: {len(x_signals)}')
    print(f'window size: {window_size}')

    # create empty dataframe to be populated later on
    feature_segments = pd.DataFrame(columns=feature_names)
    signals_len = data.shape[0]
    for i in range(window_size, signals_len, target_size):
        # print(f'start x: {i - window_size} - end x: {i}')
        # iteration pattern is the following
        # 0 <= 765045 - 640 (764405)
        # 64 <= 765045 - 640
        # 128 <= 765045 - 640
        # 192 <= 765045 - 640
        # 256 <= 765045 - 640
        # 320 <= 765045 - 640
        # ...
        # 764288 <= 765045 - 640
        # 764352 <= 765045 - 640 (764405) we only go until here as 764352 + 64
        # (or another 0.5s segment would result in 764416 which is greater than 764405)  

        # maybe what this truly does is we get 5 seconds of a signal and since there are 128 signals per second
        # we would in total get 640 rows for 5 seconds of our signals

        # oh so this is the denominator part of the min max scaling formula
        # and as stated by llanes-jurado et al. they used min max scaling to scale the raw signals
        # mroeover nanmax and min is used in case of nan values in the windows which returns
        # minimum of an array or minimum along an axis, ignoring any NaNs
        # 0:0 + 640 = 0:640
        # 64:64 + 640 = 64:704
        # 128:128 + 640 = 128:768
        # 192:192 + 640 = 192:832
        # ...
        # 764352:764352 + 640 = 764352:764992
        # if we exceed 764352 by adding 64 then we have 764416
        # 764416:764416 + 640 = 764416:765056 and 765056 exceeds the index and rows of 765045 

        # if scale is true then min max scaling is applied
        x_signal = x_signals[(i - window_size):i]
        if scale == True:
            
            denominator_norm = (np.nanmax(x_signal) - np.nanmin(x_signal))
            denominator_norm = denominator_norm + 1e-100 if denominator_norm == 0 else denominator_norm

            # this is full min max scaling formula with the denominator using
            # the difference of the min and max of a window
            # to address also potential zero division concerns
            x_window = (x_signal - np.nanmin(x_signal)) / denominator_norm
        else:
            # this would be appropriate if there was a larger ram
            x_window = x_signal

        # put windowed signal to current dataframe slice
        curr_data = data.iloc[(i - window_size):i]
        curr_data['scaled_signal'] = x_window

        # calculate current dataframe slices whole wavelets and half wavelets
        whole_wave, half_wave = load_wavelet_data_hybrid(curr_data, col_to_use="scaled_signal", hertz=hertz)
        feature_segment = compute_features(curr_data, whole_wave, half_wave, samples_per_sec=hertz)
        row_to_add = pd.DataFrame(feature_segment.reshape(1, -1), columns=feature_names)
        feature_segments  = pd.concat([feature_segments, row_to_add], axis=0, ignore_index=True)

        # returns the mean of a list or matrix of values given an
        # axis ignoring any nan values. Based on Llanes-Jurado et al. (2023)
        # the threshold for a 0.5s segment of a signal to be accepted as an
        # artifact must be 0.5 or 50% if it is less than this then the label
        # of such a segment of the signal will be not an artifact
        # 0 + 640 - 64:0 + 640 = 576:640
        # 64 + 640 - 64:64 + 640 = 640:704
        # 128 + 640 - 64:128 + 640 = 704:768
        # ...
        # 764288 + 640 - 64:764288 + 640 = 764864:764928
        # 764352 + 640 - 64:764352 + 640 = 764928:764992
        # this iteration pattern now I know just gets the last 0.5s segment of a 5s segment and 
        
        if y_col is not None:
            y_signal = y_signals[(i - target_size):i]
            cond = np.nanmean(y_signal) > 0.5
            y_window_list.append(1 if cond else 0)
        
        # this will increment our i by the size of our target frames which in this 
        # case is 0.5s or 64 rows since 1 second is 128 rows or 128hz
        ctr += 1
    print(f'number of rows created: {ctr}')

    # because x_window_list and y_window_list when converted to a numpy array will
    # be of dimensions (m, 640) and (m,) respectively we need to first and foremost
    # reshpae x_window_list into a 3D matrix such that it is able to be taken in
    # by an LSTM layer, m being the number of examples, 640 being the number of time steps
    # and 1 being the number of features which will be just our raw eda signals.
    subject_signals = feature_segments.fillna(0)

    # and because y_window_list is merely of dimension (m, ) we will have to
    # expand its dimensions such that it can be accepted by our tensorflow model
    # resulting shape of subject_labels will now be (m, 1)
    if y_col is not None:
        subject_labels = pd.Series(y_window_list)

        return [(subject_signals, subject_labels)]
    
    else:
        return subject_signals


def extract_features(df: pd.DataFrame | np.ndarray, extractor_fn):
    """
    takes in dataframe of 128hz downsamples it to 16hz and uses
    both frequencies of dataframes to extract low level features
    from

    args:
        df -
        extractor_fn - calllback function that will be used by extract_features()
        as feature extractor. This callback can either be a feature extractor
        function that extracts features from 5 second windows to be used for hybrid
        architectures like LSTM-SVM or 0.5 second windows to be used for traditional
        ML architectures like SVM, RF, LR, GBT, etc.
    """

    eda_df_128hz = df.copy()
    eda_df_128hz.columns = ['time', 'raw_signal', 'clean_signal', 'label', 'auto_signal', 'pred_art', 'post_proc_pred_art']
    # print(eda_df_128hz)

    # set index first of uninterpolated eda data
    start_time = eda_df_128hz.iloc[0]['time']
    eda_df_128hz.set_index(pd.date_range(start=start_time, periods=eda_df_128hz.shape[0], freq=get_time_frequency(128)), inplace=True)
    
    # interpolate data to 16hz by downsampling
    eda_df_16hz = interpolate_signals(eda_df_128hz, sample_rate=128, start_time=start_time, target_hz=16)

    # once downsampled low-pass filter both uninterpolated and
    # interpolated data. Cutoff of 1 means we retain the 128hz signal
    # at 128hz, because if we put in 0.5 for instance then we "cut off"
    # the 128hz signal by 0.5 or 50% which results in 64hz
    eda_df_128hz['filtered_signal'] = butter_lowpass_filter(eda_df_128hz['raw_signal'], cutoff=1.0, samp_freq=128, order=6)
    eda_df_16hz['filtered_signal'] = butter_lowpass_filter(eda_df_16hz['raw_signal'], cutoff=1.0, samp_freq=16, order=6)

    # define arguments for both 128 hertz and 16 hertz signal dataframes
    args_128hz = {
        'data': eda_df_128hz,
        'hertz': 128,
        'window_time': 0.5,
        'verbose': False
    } if extractor_fn.__name__ == "extract_features_per_hour" else {
        'data': eda_df_128hz, 
        'hertz': 128, 
        'window_time': 5, 
        'x_col': "raw_signal",
        'target_time': 0.5,
        'y_col': 'label',
        'scale': True,
        'verbose': False
    }

    args_16hz = {
        'data': eda_df_16hz,
        'hertz': 16,
        'window_time': 0.5,
        'verbose': False
    } if extractor_fn.__name__ == "extract_features_per_hour" else {
        'data': eda_df_16hz, 
        'hertz': 16, 
        'window_time': 5, 
        'x_col': "raw_signal",
        'target_time': 0.5,
        'y_col': 'label',
        'scale': True,
        'verbose': False
    }

    # process the dfs and extract its features
    data_128hz = extractor_fn(**args_128hz)
    data_16hz = extractor_fn(**args_16hz)

    # rejoin 128hz and 16hz features and labels
    eda_feature_df, eda_labels = rejoin_data(data_128hz, data_16hz)

    return eda_feature_df, eda_labels


def concur_extract_features_from_all(dir: str, files: list[str], arch: str="hybrid"):
    def helper(file: str):
        subject_name = file.strip(".csv")
        eda_df_128hz = pd.read_csv(f'{dir}{file}', sep=';')
        extractor_fn = extract_features_per_hour if arch.lower() == "ml" else extract_features_hybrid
        eda_feature_df, eda_labels = extract_features(eda_df_128hz, extractor_fn=extractor_fn)

        return (subject_name, (eda_feature_df, eda_labels))

    with ThreadPoolExecutor() as exe:
        eda_data = list(exe.map(helper, files))

    return eda_data