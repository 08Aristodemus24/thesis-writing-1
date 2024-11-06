import math
import pandas as pd
import numpy as np
# import tensorflow as tf
import pywt
from concurrent.futures import ThreadPoolExecutor
import datetime
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from scipy.signal import butter, sosfiltfilt
from statsmodels.tsa.ar_model import AutoReg

def moving_average(array: np.ndarray, window_size: int | float):
    # calculates moving average of the corrected segments
    final_list = array[:int(window_size / 2)].tolist()
    start = int(window_size / 2)
    end = len(array) - int(window_size / 2)
    for i in range(start, end):
        start_i = i - int(window_size / 2)
        end_i = i + int(window_size / 2)

        # calculate mean of window
        value = np.mean(array[start_i:end_i])
        final_list += [value]
        
    final_list += array[-int(window_size / 2):].tolist()
    
    return np.array(final_list)

def decompose_signal(raw_signal, samp_freq: int=128, method: str="highpass"):
    """
    decomposes raw eda signal into phasic and tonic components
    either using highpass or median method
    """

    if method == "highpass":
        # calculate tonic component
        lowpass_sos = butter(2, 0.05, btype="lowpass", output="sos", fs=samp_freq)
        tonic_component = sosfiltfilt(lowpass_sos, raw_signal)

        # calculate phasic component
        highpass_sos = butter(2, 0.05, btype="highpass", output="sos", fs=samp_freq)
        phasic_component = sosfiltfilt(highpass_sos, raw_signal)

        return tonic_component, phasic_component

    elif method == "median":
        tonic_component = moving_average(raw_signal, 4 * samp_freq)
        phasic_component = raw_signal - tonic_component

        return tonic_component, phasic_component

def _differentiate(data):
    """
    computes the 1st and 2nd order derivative values 
    of the eda signal
    """
    
    F1_prime = (data[1:-1] + data[2:]) / 2 - data[1:-1] + data[:-2] / 2
    F2_prime = data[2:] - (2 * data[1:-1]) + data[:-2]

    return F1_prime, F2_prime

def _compute_stat_feats(data, col_to_use='raw_signal'):
    """
    computes the statistical features of both the 
    raw/unfiltered signal and the filtered signal
    """

    signal = data[col_to_use]

    _1d_signal, _2d_signal = _differentiate(signal)

    max = np.max(signal, axis=0)
    min = np.min(signal, axis=0)
    amp = np.mean(signal, axis=0)
    median = np.median(signal, axis=0)
    std = np.std(signal, axis=0)
    range = np.max(signal, axis=0) - np.min(signal, axis=0)
    shannon_entropy = entropy(signal.value_counts())

    _1d_max = np.max(_1d_signal, axis=0)
    _1d_min = np.min(_1d_signal, axis=0)
    _1d_amp = np.mean(_1d_signal, axis=0)
    _1d_median = np.median(_1d_signal, axis=0)
    _1d_std = np.std(_1d_signal, axis=0)
    _1d_range = np.max(_1d_signal, axis=0) - np.min(_1d_signal, axis=0)
    _1d_shannon_entropy = entropy(_1d_signal.value_counts())
    _1d_max_abs = np.max(np.absolute(_1d_signal), axis=0)
    _1d_avg_abs = np.mean(np.absolute(_1d_signal), axis=0)

    _2d_max = np.max(_2d_signal, axis=0)
    _2d_min = np.min(_2d_signal, axis=0)
    _2d_amp = np.mean(_2d_signal, axis=0)
    _2d_median = np.median(_2d_signal, axis=0)
    _2d_std = np.std(_2d_signal, axis=0)
    _2d_range = np.max(_2d_signal, axis=0) - np.min(_2d_signal, axis=0)
    _2d_shannon_entropy = entropy(_2d_signal.value_counts())
    _2d_max_abs = np.max(np.absolute(_2d_signal), axis=0)
    _2d_avg_abs = np.mean(np.absolute(_1d_signal), axis=0)

    return (max, min, amp, median, std, range, shannon_entropy,
    _1d_max, _1d_min, _1d_amp, _1d_median, _1d_std, _1d_range, _1d_shannon_entropy, _1d_max_abs, _1d_avg_abs, 
    _2d_max, _2d_min, _2d_amp, _2d_median, _2d_std, _2d_range, _2d_shannon_entropy, _2d_max_abs, _2d_avg_abs)

def _compute_ar_feats(data: pd.DataFrame | np.ndarray, col_to_use='raw_signal'):
    """
    computes autoregressive features by training AutoReg
    from statsmodels.tsa.ar_model then obtaining sigma2
    and param attributes containing the error variance and
    all the optimized coefficients excluding the intercept

    args:
        data - is a 0.5s segment/window/epoch of a subjects
        128hz signals
    """

    ar_results = None
    try:
        data = data.reset_index(drop=True)
        input = data[col_to_use]
        ar_model = AutoReg(input, lags=2)
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
        print(f'{ar_features}\n')

        return ar_features
        
    except ValueError:
        input = data[col_to_use]
        input = pd.concat([input, pd.Series([input.mean() for _ in range(10)])], ignore_index=True)
        ar_model = AutoReg(input, lags=2)
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
        print(f'{ar_features}\n')

        return ar_features


def _compute_morphological_feats(data: pd.DataFrame | np.ndarray, col_to_use: str='raw_signal'):
    skewness = skew(data[col_to_use])
    kurt = kurtosis(data[col_to_use])
    return skewness, kurt

def compute_features(data: pd.DataFrame | np.ndarray):
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
    
    phasic_max - the maximum value of the low-pass filtered eda signal
    phasic_min - the maximum value of the low-pass filtered eda signal
    phasic_amp - the average/mean value of the low-pass filtered eda signal
    phasic_median - the median value of the low-pass filtered eda signal
    phasic_std - the standard deviation value of the low-pass filtered eda signal
    phasic_range - the range value of the low-pass filtered eda signal
    phasic_shannon_entropy - the shannon entropy value of the low-pass filtered eda signal

    phasic_1d_max - the maximum value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_min - the minimum value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_amp - the average/mean value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_median - the median value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_std - the standard deviation value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_range - the range value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_shannon_entropy - the shannon entropy value of the first order derivative of the low-pass filtered eda signal
    phasic_1d_max_abs - the maximum value out of the absolute values of the first order derivative of the low-pass filtered eda signal
    phasic_1d_avg_abs - the average/mean value out of the absolute values of the first order derivative of the low-pass filtered eda signal

    phasic_2d_max - the maximum value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_min - the minimum value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_amp - the average/mean value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_median - the median value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_std - the standard deviation value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_range - the range value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_shannon_entropy - the shannon entropy value of the second order derivative of the low-pass filtered eda signal
    phasic_2d_max_abs - the maximum value out of the absolute values of the second order derivative of the low-pass filtered eda signal
    phasic_2d_avg_abs - the average/mean value out of the absolute values of the second order derivative of the low-pass filtered eda signal
    
    ar_coeffs - coefficients excluding bias/intercept coefficient of the trained autoregressive model
    ar_err_var - error variance value of the trained autoregressive model

    args:
        data - data slice/segment/window of 0.5s 
        whole_wave - a dataframe containing whole of the wavelet coefficients 
        half_wave - a dataframe containing half the time of the whole wavelet coefficients
    """

    # compute statistical features
    raw_max, raw_min, raw_amp, raw_median, raw_std, raw_range, raw_shannon_entropy, \
    raw_1d_max, raw_1d_min, raw_1d_amp, raw_1d_median, raw_1d_std, raw_1d_range, raw_1d_shannon_entropy, raw_1d_max_abs, raw_1d_avg_abs, \
    raw_2d_max, raw_2d_min, raw_2d_amp, raw_2d_median, raw_2d_std, raw_2d_range, raw_2d_shannon_entropy, raw_2d_max_abs, raw_2d_avg_abs =  _compute_stat_feats(data, col_to_use='raw_signal')

    phasic_max, phasic_min, phasic_amp, phasic_median, phasic_std, phasic_range, phasic_shannon_entropy, \
    phasic_1d_max, phasic_1d_min, phasic_1d_amp, phasic_1d_median, phasic_1d_std, phasic_1d_range, phasic_1d_shannon_entropy, phasic_1d_max_abs, phasic_1d_avg_abs, \
    phasic_2d_max, phasic_2d_min, phasic_2d_amp, phasic_2d_median, phasic_2d_std, phasic_2d_range, phasic_2d_shannon_entropy, phasic_2d_max_abs, phasic_2d_avg_abs = _compute_stat_feats(data, col_to_use='phasic')

    # compute autoregressive features
    # raw_ar_coeff_1, raw_ar_coeff_2, raw_ar_err_var = _compute_ar_feats(data, col_to_use='raw_signal')
    # phasic_ar_coeff_1, phasic_ar_coeff_2, phasic_ar_err_var = _compute_ar_feats(data, col_to_use='phasic')
    # ar_feats = _compute_ar_feats(data)
    # ar_coeffs, ar_err_var = ar_feats[:-1], ar_feats[-1]

    # compute morphological features
    raw_skewness, raw_kurt = _compute_morphological_feats(data, col_to_use='raw_signal')
    phasic_skewness, phasic_kurt = _compute_morphological_feats(data, col_to_use='phasic')

    features = np.hstack([
        # statistical features
        raw_max, raw_min, raw_amp, raw_median, raw_std, raw_range, raw_shannon_entropy, 
        raw_1d_max, raw_1d_min, raw_1d_amp, raw_1d_median, raw_1d_std, raw_1d_range, raw_1d_shannon_entropy, raw_1d_max_abs, raw_1d_avg_abs, 
        raw_2d_max, raw_2d_min, raw_2d_amp, raw_2d_median, raw_2d_std, raw_2d_range, raw_2d_shannon_entropy, raw_2d_max_abs, raw_2d_avg_abs, 
        phasic_max, phasic_min, phasic_amp, phasic_median, phasic_std, phasic_range, phasic_shannon_entropy, 
        phasic_1d_max, phasic_1d_min, phasic_1d_amp, phasic_1d_median, phasic_1d_std, phasic_1d_range, phasic_1d_shannon_entropy, phasic_1d_max_abs, phasic_1d_avg_abs, 
        phasic_2d_max, phasic_2d_min, phasic_2d_amp, phasic_2d_median, phasic_2d_std, phasic_2d_range, phasic_2d_shannon_entropy, phasic_2d_max_abs, phasic_2d_avg_abs,

        # autoregressive coefficients excluding bias/intercept and error variance
        # raw_ar_coeff_1, raw_ar_coeff_2, raw_ar_err_var,
        # phasic_ar_coeff_1, phasic_ar_coeff_2, phasic_ar_err_var,

        # morphological features of raw and phasic components of signal
        raw_skewness, raw_kurt,
        phasic_skewness, phasic_kurt
        ])
    
    return features

def get_features(subjects, hertz, window_size):
    # note this samples per sec is not arbitrary and a 
    # user defined value but derived from our frequency 
    # value entirely i.e. because we've recorded our
    # raw data at 128hz then that means that the 
    # samples of data we have per second would be 128
    samples_per_sec = hertz

    # define feature names in dataframe
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

        f"phasic_{samples_per_sec}hz_max", f"phasic_{samples_per_sec}hz_min", 
        f"phasic_{samples_per_sec}hz_amp", f"phasic_{samples_per_sec}hz_median", 
        f"phasic_{samples_per_sec}hz_std", f"phasic_{samples_per_sec}hz_range", 
        f"phasic_{samples_per_sec}hz_shannon_entropy",
        
        f"phasic_{samples_per_sec}hz_1d_max", f"phasic_{samples_per_sec}hz_1d_min", 
        f"phasic_{samples_per_sec}hz_1d_amp", f"phasic_{samples_per_sec}hz_1d_median", 
        f"phasic_{samples_per_sec}hz_1d_std", f"phasic_{samples_per_sec}hz_1d_range", 
        f"phasic_{samples_per_sec}hz_1d_shannon_entropy",
        f"phasic_{samples_per_sec}hz_1d_max_abs", f"phasic_{samples_per_sec}hz_1d_avg_abs", 

        f"phasic_{samples_per_sec}hz_2d_max", f"phasic_{samples_per_sec}hz_2d_min", 
        f"phasic_{samples_per_sec}hz_2d_amp", f"phasic_{samples_per_sec}hz_2d_median", 
        f"phasic_{samples_per_sec}hz_2d_std", f"phasic_{samples_per_sec}hz_2d_range", 
        f"phasic_{samples_per_sec}hz_2d_shannon_entropy",
        f"phasic_{samples_per_sec}hz_2d_max_abs", f"phasic_{samples_per_sec}hz_2d_avg_abs", 

        # autoregressive features names
        # f"raw_ar_coeff_1_{samples_per_sec}hz", f"raw_ar_coeff_2_{samples_per_sec}hz", f"raw_ar_var_{samples_per_sec}hz",
        # f"phasic_ar_coeff_1_{samples_per_sec}hz", f"phasic_ar_coeff_2_{samples_per_sec}hz", f"phasic_ar_var_{samples_per_sec}hz",

        # morphological features names
        f"raw_{samples_per_sec}hz_skewness", f"raw_{samples_per_sec}hz_kurt",
        f"phasic_{samples_per_sec}hz_skewness", f"phasic_{samples_per_sec}hz_kurt",

        ]

    feature_names_len = len(feature_names)

    subjects_features_and_labels = {}

    # concatenate all dataframes each subject has
    for subject, records in subjects.items():
        
        # reset after each subject
        subject_features_and_labels = []

        # perform feature extraction here for each subject
        for record in records:

            _, df = record
            print(f'new subject: {subject}')
            print(f'n records: {df.shape[0]}')

            # # here we would be calculating how many samples we would have 
            # # per hour given we have 128 samples per second
            # samples_per_hour = samples_per_sec * secs_per_min * min_per_hour

            # we also need to specify how large our windows/epochs/segments
            # would be in order to create the rows for our dataset and subsequently
            # each feature of that window or row
            samples_per_win_size = int(samples_per_sec * window_size)

            # get number of rows of 128hz timestamps and signals
            n_rows = df.shape[0]

            # dividing the number of rows by the number of samples per 0.5 seconds 
            # will allow us to get a sense how many segments or rows of 0.5 seconds
            # can we get from this time series dataframe of our signals
            num_labels = math.ceil(n_rows / samples_per_win_size)
            print(f'num_labels: {num_labels}')

            features_and_labels = pd.DataFrame(np.zeros(shape=(num_labels, feature_names_len)), columns=feature_names)

            for i in range(num_labels):
                start = i * samples_per_win_size
                end = min((i + 1) * samples_per_win_size, n_rows)
                # print(f'start: {start} - end: {end}')

                curr_data = df.iloc[start:end]

                # pass curr data to feature extraction function
                feature_segment = compute_features(curr_data)
                features_and_labels.iloc[i] = feature_segment

            # assign labels to features_and_labels dataframe
            features_and_labels['label'] = df['label'][0]

            # append and collect features and labels
            subject_features_and_labels.append(features_and_labels)
        
        # assign subject name key to collected subject features and labelss 
        subjects_features_and_labels[subject] = subject_features_and_labels
    return subjects_features_and_labels
    # window will be 5 seconds of non overlapping segments 