from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import pywt
from scipy.signal import butter, lfilter

def which_element(object_list):
    """
    This function only defines and returns the position 
    in an array or list where it is a True value.
    e.g. [True, False, true, true, False] -> [0, 2, 3]

    Dependency of find_begin_end()
    """
    result = [i_obj for i_obj, obj in enumerate(object_list) if obj]
    return result

def find_begin_end(x_p):
    """
    This function defines where it is found the first and the last 1 of a sequence of ones (1s)
    in a binary vector where. For example: X = [0, 0, 1, 1, 1, 0, 0, 1, 1]. This function will 
    have this as a result: find_begin_end(X) = [2, 7], [4, 8]. It means that at positions 2 and 7
    of X it begins a sequence of 1s. On the other hand, 4 and 8 are the positions where these 
    sequences ends. This function is necessary to localize the position of the 
    artifacts in the raw binary signal of the target or the prediction.    
    """
    
    # this takes in a boolean numpy array or a dataframe
    # not a list since [1, 0, 1, 0, 0] == 1 for instance 
    # would just return false
    # For instance if we had now signals that are labeled as
    # artifacts or non artifacts...
    # >>> signal = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1])
    # >>> signal == 1
    # array([False, False, True, True, True, False, False, True, True])
    # when this is passed through which_element() this will
    # return [2, 3, 4, 7, 8]
    pos_artf_true = which_element(x_p == 1)

    # get the very starting index where an artifact is indeed an artifact
    # which in this case for instance will be 2, start_pos_artf_true is [2]
    start_pos_artf_true = [pos_artf_true[0]]

    # this block finds the begin indices of artifacts in signal
    # pos_artf_true is [3, 4, 7, 8], i is initially 0, p_art_t is initially 3
    for i, p_art_t in enumerate(pos_artf_true[1:]):
        # (3 - 2) -> (4 - 3) -> (7 - 4) -> (8 - 7)
        if (p_art_t - pos_artf_true[i]) > 1:
            start_pos_artf_true.append(p_art_t)

    end_pos_artf_true = [p_art_t for i, p_art_t in enumerate(pos_artf_true[:-1]) if (pos_artf_true[i + 1] - p_art_t) > 1]

    end_pos_artf_true.append(pos_artf_true[-1])
    
    return start_pos_artf_true, end_pos_artf_true



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
        print(freq)
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



def load_wavelet_data(data: pd.DataFrame | np.ndarray):
    """
    function to create whole and half wavelet dataframes
    """

    # create timestamps
    timestamp_list = pd.to_datetime(data['time'].iloc[0::64], unit='s')

    # 128hz with 0.5s window is to 1/16 and 1/32 and 8hz with 5s window is to 1/1 and 1/2
    whole_inc_ts = pd.date_range(start=timestamp_list[0], periods=data.shape[0], freq='62.5ms')
    half_inc_ts = pd.date_range(start=timestamp_list[0], periods=data.shape[0], freq='31.25ms')

    # obtain wavelet coefficients
    coeffs = pywt.wavedec(data['raw_signal'], wavelet='haar', level=3)
    cA_3, cD_3, cD_2, cD_1 = coeffs
    n_rows_wavelet = cD_3.shape[0]

    # reshape, calculate absolute and max value of wavelet coefficients
    # such that all shapes of wavelet coefficients adhere to that of the
    # third level coefficients shape
    whole_coeff_1 = np.max(np.absolute(np.reshape(cD_1[:n_rows_wavelet * 4], newshape=(n_rows_wavelet, 4))), axis=1)
    whole_coeff_2 = np.max(np.absolute(np.reshape(cD_2[:n_rows_wavelet * 2], newshape=(n_rows_wavelet, 2))), axis=1)
    whole_coeff_3 = np.absolute(cD_3[:n_rows_wavelet])

    # create whole wave features dataframe
    whole_wave_features = pd.DataFrame({
        'first_16thofa_sec_feat': whole_coeff_1,
        'second_16thofa_sec_feat': whole_coeff_2,
        'third_16thofa_sec_feat': whole_coeff_3,
    })

    # use whole increment timestamps as indices for the whole wave features dataframe
    whole_wave_features.set_index(whole_inc_ts[:whole_wave_features.shape[0]], inplace=True)

    # recall that cD_1 has shape of 9688 which is second_data_n_segments_half * 2
    half_coeff_1 = np.max(np.absolute(np.reshape(cD_1[:n_rows_wavelet * 4], newshape=(n_rows_wavelet * 2, 2))), axis=1)
    half_coeff_2 = np.absolute(cD_2[:n_rows_wavelet * 2])

    half_wave_features = pd.DataFrame({
        'first_32thofa_sec_feat': half_coeff_1,
        'second_32thofa_sec_feat': half_coeff_2,
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
    print(third_lvl_shape)
    print(second_lvl_shape)
    print(first_lvl_shape)

    if (((third_lvl_shape * 2) - second_lvl_shape) > 0) or (((third_lvl_shape * 4) - first_lvl_shape) > 0) or (third_lvl_shape % 2 != 0):
        # calculate amount of zeros to append to 1st and 2nd level coefficients
        n_zeros_second_lvl = (third_lvl_shape * 2) - second_lvl_shape
        n_zeros_first_lvl = (third_lvl_shape * 4) - first_lvl_shape
        print(n_zeros_first_lvl, n_zeros_second_lvl)

        # append the amount of calculated zeros to the 1st and 2nd level coefficients
        second_lvl_zeros = np.zeros(shape=(n_zeros_second_lvl, ))
        first_lvl_zeros = np.zeros(shape=(n_zeros_first_lvl, ))
        cD_2 = np.hstack((cD_2, second_lvl_zeros))
        cD_1 = np.hstack((cD_1, first_lvl_zeros))

    return cA_1, cD_3, cD_2, cD_1



def _differentiate(data):
    """
    computes the 1st and 2nd order derivative values 
    of the eda signal
    """
    
    F1_prime = (data[1:-1] + data[2:]) / 2 - data[1:-1] + data[:-2] / 2
    F2_prime = data[2:] - (2 * data[1:-1]) + data[:-2]

    return F1_prime, F2_prime



def _compute_stat_feats(data):
    """
    computes the statistical features of both the 
    raw/unfiltered signal and the filtered signal
    """

    raw_signal = data['raw_signal']
    filt_signal = data['filtered_signal']

    raw_1d_signal, raw_2d_signal = _differentiate(raw_signal)
    filt_1d_signal, filt_2d_signal = _differentiate(filt_signal)

    raw_amp = np.mean(raw_signal)
    raw_1d_max = np.max(raw_1d_signal)
    raw_1d_min = np.min(raw_1d_signal)
    raw_1d_max_abs = np.max(np.absolute(raw_1d_signal))
    raw_1d_avg_abs = np.mean(np.absolute(raw_1d_signal))
    raw_2d_max = np.max(raw_2d_signal)
    raw_2d_min = np.min(raw_2d_signal)
    raw_2d_max_abs = np.max(np.absolute(raw_2d_signal))
    raw_2d_avg_abs = np.mean(np.absolute(raw_1d_signal))

    filt_amp = np.mean(filt_signal)
    filt_1d_max = np.max(filt_1d_signal)
    filt_1d_min = np.min(filt_1d_signal)
    filt_1d_max_abs = np.max(np.absolute(filt_1d_signal))
    filt_1d_avg_abs = np.mean(np.absolute(filt_1d_signal))
    filt_2d_max = np.max(filt_2d_signal)
    filt_2d_min = np.min(filt_2d_signal)
    filt_2d_max_abs = np.max(np.absolute(filt_2d_signal))
    filt_2d_avg_abs = np.mean(np.absolute(filt_1d_signal))

    return raw_amp, raw_1d_max, raw_1d_min, raw_1d_max_abs, raw_1d_avg_abs, raw_2d_max, raw_2d_min, raw_2d_max_abs, raw_2d_avg_abs, filt_amp, filt_1d_max, filt_1d_min, filt_1d_max_abs, filt_1d_avg_abs, filt_2d_max, filt_2d_min, filt_2d_max_abs, filt_2d_avg_abs



def _compute_wave_feats(wave: pd.DataFrame | np.ndarray):
    """
    computes the maximum, mean, standard deviation, median, and 
    no. of coefficients above zero values of the given wavelet 
    dataframe
    """

    wavelet_feats_max = wave.max(axis=0).tolist()
    wavelet_feats_mean = wave.mean(axis=0).tolist()
    wavelet_feats_std = wave.std(axis=0).tolist()
    wavelet_feats_median = wave.median(axis=0).tolist()
    wavelet_feats_n_coeffs_above_zero = (wave > 0).astype('int').sum(axis=0)

    return wavelet_feats_max, wavelet_feats_mean, wavelet_feats_std, wavelet_feats_median, wavelet_feats_n_coeffs_above_zero



def compute_features(data: pd.DataFrame | np.ndarray, whole_wave: pd.DataFrame | np.ndarray, half_wave: pd.DataFrame | np.ndarray):
    """
    computes the ff. features given the 0.5s segment/window dataframe or numpy matrix 'data'

    raw_amp - the amplitude of the raw/unfiltered 128hz eda signal
    raw_1d_max - the maximum value of the first order derivative of the unfiltered 128hz eda signal
    raw_1d_min - the minimum value of the first order derivative of the unfiltered 128hz eda signal
    raw_1d_max_abs - the maximum value out of the absolute values of the first order derivative of the unfiltered 128hz eda signal
    raw_1d_avg_abs - the average/mean value out of the absolute values of the first order derivative of the unfiltered 128hz eda signal
    raw_2d_max - the maximum value of the second order derivative of the unfiltered 128hz eda signal
    raw_2d_min - the minimum value of the second order derivative of the unfiltered 128hz eda signal
    raw_2d_max_abs - the maximum value out of the absolute values of the second order derivative of the unfiltered 128hz eda signal
    raw_2d_avg_abs - the average/mean value out of the absolute values of the second order derivative of the unfiltered 128hz eda signal
    
    filt_amp - the amplitude of the low-pass filtered 16hz eda signal
    filt_1d_max - the maximum value of the first order derivative of the low-pass filtered 16hz eda signal
    filt_1d_min - the minimum value of the first order derivative of the low-pass filtered 16hz eda signal
    filt_1d_max_abs - the maximum value out of the absolute values of the first order derivative of the low-pass filtered 16hz eda signal
    filt_1d_avg_abs - the average/mean value out of the absolute values of the first order derivative of the low-pass filtered 16hz eda signal
    filt_2d_max - the maximum value of the second order derivative of the low-pass filtered 16hz eda signal
    filt_2d_min - the minimum value of the second order derivative of the low-pass filtered 16hz eda signal
    filt_2d_max_abs - the maximum value out of the absolute values of the second order derivative of the low-pass filtered 16hz eda signal
    filt_2d_avg_abs - the average/mean value out of the absolute values of the second order derivative of the low-pass filtered 16hz eda signal
    
    first_16thoas_max - the maximum value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole/16th of a second wavelet dataframe
    second_16thoas_max - the maximum value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_16thoas_max - the maximum value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_16thoas_mean - the average/mean value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_16thoas_mean - the average/mean value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_16thoas_mean  - the average/mean value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_16thoas_std - the standard deviation of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_16thoas_std - the standard deviation of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_16thoas_std - the standard deviation of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_16thoas_median - the median value of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_16thoas_median - the median value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_16thoas_median - the median value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe
    first_16thoas_n_coeffs_above_zero - the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 1st wavelet feature of the whole wavelet dataframe
    second_16thoas_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 2nd wavelet feature of the whole wavelet dataframe
    third_16thoas_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 3rd wavelet feature of the whole wavelet dataframe

    first_32thoas_max - the maximum value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half/32th of a second wavelet dataframe
    second_32thoas_max - the maximum value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half/32th of a second wavelet dataframe
    third_32thoas_max - the maximum value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the half/32th of a second wavelet dataframe
    first_32thoas_mean - the average/mean value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half/32th of a second wavelet dataframe
    second_32thoas_mean - the average/mean value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half/32th of a second wavelet dataframe
    third_32thoas_mean  - the average/mean value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the half/32th of a second wavelet dataframe
    first_32thoas_std - the standard deviation of the 0.5s segment/window/epoch from the 1st wavelet feature of the half/32th of a second wavelet dataframe
    second_32thoas_std - the standard deviation of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half/32th of a second wavelet dataframe
    third_32thoas_std - the standard deviation of the 0.5s segment/window/epoch from the 3rd wavelet feature of the half/32th of a second wavelet dataframe
    first_32thoas_median - the median value of the 0.5s segment/window/epoch from the 1st wavelet feature of the half/32th of a second wavelet dataframe
    second_32thoas_median - the median value of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half/32th of a second wavelet dataframe
    third_32thoas_median - the median value of the 0.5s segment/window/epoch from the 3rd wavelet feature of the half/32th of a second wavelet dataframe
    first_32thoas_n_coeffs_above_zero - the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 1st wavelet feature of the half/32th of a second wavelet dataframe
    second_32thoas_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 2nd wavelet feature of the half/32th of a second wavelet dataframe
    third_32thoas_n_coeffs_above_zero - the the number/count of wavelet coefficients above zero of the 0.5s segment/window/epoch from the 3rd wavelet feature of the half/32th of a second wavelet dataframe

    args:
        data - data slice/segment/window of 0.5s 
        whole_wave - 16th of a second wavelet coefficients 
        half_wave - 32th of a second wavelet coefficients
    """

    # compute statistical features
    raw_amp, raw_1d_max, raw_1d_min, raw_1d_max_abs, raw_1d_avg_abs, raw_2d_max, raw_2d_min, raw_2d_max_abs, raw_2d_avg_abs, filt_amp, filt_1d_max, filt_1d_min, filt_1d_max_abs, filt_1d_avg_abs, filt_2d_max, filt_2d_min, filt_2d_max_abs, filt_2d_avg_abs = _compute_stat_feats(data)

    # compute wavelet features
    wavelet_feats_16thofas_max, wavelet_feats_16thofas_mean, wavelet_feats_16thofas_std, wavelet_feats_16thofas_median, wavelet_feats_16thofas_n_coeffs_above_zero = _compute_wave_feats(whole_wave)
    wavelet_feats_32thofas_max, wavelet_feats_32thofas_mean, wavelet_feats_32thofas_std, wavelet_feats_32thofas_median, wavelet_feats_32thofas_n_coeffs_above_zero = _compute_wave_feats(half_wave)

    features = np.hstack([
        raw_amp, raw_1d_max, raw_1d_min, raw_1d_max_abs, raw_1d_avg_abs, raw_2d_max, raw_2d_min, raw_2d_max_abs, raw_2d_avg_abs, filt_amp, filt_1d_max, filt_1d_min, filt_1d_max_abs, filt_1d_avg_abs, filt_2d_max, filt_2d_min, filt_2d_max_abs, filt_2d_avg_abs,
        wavelet_feats_16thofas_max, wavelet_feats_16thofas_mean, wavelet_feats_16thofas_std, wavelet_feats_16thofas_median, wavelet_feats_16thofas_n_coeffs_above_zero,
        wavelet_feats_32thofas_max, wavelet_feats_32thofas_mean, wavelet_feats_32thofas_std, wavelet_feats_32thofas_median, wavelet_feats_32thofas_n_coeffs_above_zero,
        ])
    
    return features
