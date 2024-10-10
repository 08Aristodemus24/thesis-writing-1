from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import datetime
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, splrep, UnivariateSpline, spline
from scipy.signal import butter, filtfilt, lfilter, firwin, hilbert, sosfiltfilt

import tensorflow as tf

def down_sample(signals: pd.DataFrame | np.ndarray | list, target_freq = 16):
    """
    takes in raw signals x and converts it to another frequency
    via downsampling e.g. 128hz to 16hz
    """
    result = []
    n_rows = signals.shape[0]
    for i in np.arange(n_rows / target_freq):
        result += [np.nanmean(signals[int(i * target_freq):int((i + 1) * target_freq)])]

    return result

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
        # (3 - 2) > 1 will render false 
        # (4 - 3) > 1 will render false
        # (7 - 4) > 1 will render true
        # (8 - 7) > 1 will render false
        if (p_art_t - pos_artf_true[i]) > 1:
            start_pos_artf_true.append(p_art_t)

    # this will be the block that find the end indices of artifacts in a signal
    end_pos_artf_true = [p_art_t for i, p_art_t in enumerate(pos_artf_true[:-1]) if (pos_artf_true[i + 1] - p_art_t) > 1]
    end_pos_artf_true.append(pos_artf_true[-1])

    return start_pos_artf_true, end_pos_artf_true

def moving_average(data: np.ndarray | pd.DataFrame | pd.Series, window_size: int | float):
    """
    args:
        data - vector which will be modified
        window_size - window size is in seconds
    """

    pad_width = (window_size // 2, window_size // 2 - 1)

    first_val = data[0]
    last_val = data[-1]
    const_values = (first_val, last_val)

    data = np.pad(data, pad_width=pad_width, mode="constant", constant_values=const_values)
    ma = np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    return ma

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

def correct_signals(y_pred, df, freq_signal=128, th_t_postprocess=2.5, eda_signal="raw_signal", time_column="time"):
    """
    args:
        y_pred - is the numpy array or pandas series containing
        the predictions of a model whether it be lstm-cnn, lstm-svm,
        lr, gbt, rf, or svm models, these models nevertheless will
        output values of 1s and 0s, 

        moreover because different models have different feature
        extraction methods the selector_config and the estimator_name
        must be supplied  
        
        df - is the dataframe containing the raw signals

        model - the trained artifact detection model

        freq_signal - sampling frequency of the raw signals

        th_t_postprocess -

        eda_signal - column in the dataframe in which to correct
        
    """
    res_df = df.copy()

    target_size_fr = 64

    # the raw_signals that are less than 0 which will be likely be negative values 
    # will be rendered true and those raw signal rows with true booleans will just be
    # transformed to 0
    res_df.loc[res_df[eda_signal] < 0, eda_signal] = 0
    
    res_df["new_auto_signal"] = res_df[eda_signal].iloc[:]
    rawdata_spline_correct = res_df[eda_signal].iloc[:]

    

    ######################################
    ### AUTOMATIC ARTIFACT RECOGNITION ###
    ######################################
    
    
    
     # this initializes an array of zeros based on length of raw signals
    # to be predicted the artifacts and then corrected
    # future labels auto will have length the same as number of rows in eda df
    n_rows = df.shape[0]
    future_labels_auto = np.zeros(n_rows)
    for label_i, label in enumerate(y_pred):
        # below is the iteration pattern
        # 64 * 0:64 * (0 + 1) = 0:64
        # 64 * 1:64 * (1 + 1) = 64:128
        # 64 * 2:64 * (2 + 1) = 128:192
        # why this is, is because we know that we want segments of 0.5s
        # [0:64] = say our label for the first 0.5s segment is 1 then rows 0 to 64 will be labeled 1 
        # [64:128] = and say our label for the second 0.5s segment is 0 then rows 64 to 128 will be labeled 0
        # [128:192] = and the third segment 0 then rows 128 to 192 will be labeled 0

        # say our total calculated feature set length was 15315 rows
        # from raw 128hz signals of 980118 rows with a window size of 0.5s
        # when we finally feed our model this feature set of course the
        # corresponing predictions will have also 15315 rows and each row
        # represents a feature calculated from a segment from the raw data
        # i.e. rows [0] - [63] are the signals used in calculating the first
        # row in the feature data set, signals in rows [64] - [127] are the 
        # signals used in calculating the second row in the feature data set,
        # signals in rows [9800096] - [980117] although not entirely 64 rows
        # indicating a 0.5s segment are still used in calculating the [15314]
        # row in the feature data set 
        start = target_size_fr * label_i
        end = min(target_size_fr * (label_i + 1), n_rows)
        print(f"index {label_i}: start {start} - end {end}")
        future_labels_auto[start:end] += label
    
    # # ##############################
    # # ### TARGET POST-PROCESSING ###
    # # ##############################

    # """this"""
    # pred_target_array = res_df[eda_signal].iloc[:].copy()
    
    # # future labels will contain 1s and 0s and those greater than 0 will always be 1
    # # and those rows must be set to 1
    # future_labels_auto[future_labels_auto > 0] = 1

    # """and this seems redundant"""
    # pred_target_array = future_labels_auto

    # res_df["pred_art"] = pred_target_array

    # # pred_target_array overall is an array of 1s and 0s
    # # where it is passed to find_begin_end() which uses a function which_element() that in a boolean numpy array or a dataframe
    # # not a list since [1, 0, 1, 0, 0] == 1 for instance
    # # would just return false
    # # For instance if we had now signals that are labeled as
    # # artifacts or non artifacts...
    # # >>> signal = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1])
    # # >>> signal == 1
    # # array([False, False, True, True, True, False, False, True, True])
    # # when this is passed through which_element() this will
    # # return [2, 3, 4, 7, 8]
    # # pos_artf_true = which_element(x_p == 1)

    # # overall find_begin_end() returns to lists, the first representing the 
    # # beginning indices of all artifacts in a subjects signals 
    # # e.g. if a subjects signals that have been predicted for each 0.5s segment to
    # # have [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1] which total to 
    # # 2 seconds all in all of this subjects signals, the starting indices of these subjects
    # # signals artifacts will be collected as well as the end indices
    # # in this example start_artf_pred will be [3, 9, 13, 17, 21]
    # # and end_artf_pred will be [5, 11, 14, 18, 22]
    # start_artf_pred, end_artf_pred = find_begin_end(pred_target_array)
    
    # # length of start_artf_pred is 5 for instance
    # # 5 - 1 is 4 so therefore we would only be looping from 0 to 3 and give
    # # allowance for the last index 4 as we will be using an index of i + 1
    # # which takes into account a potential out of range error
    # for i in range(len(start_artf_pred) - 1):

    #     if np.abs(start_artf_pred[i + 1] - end_artf_pred[i]) / freq_signal <= th_t_postprocess:
    #         pred_target_array[end_artf_pred[i]:start_artf_pred[i + 1]] = 1
    
    # start_artf_pred, end_artf_pred = find_begin_end(pred_target_array)
    
    # res_df["post_proc_pred_art"] = pred_target_array
    
    # # #################################
    # # ### Compute artifacts metrics ###
    # # #################################
    
    # dict_metrics = {}
    
    # # Time until first artifact.
    # dict_metrics["time_first_artifact"] = start_artf_pred[0] / freq_signal
    
    # # Mean time between two artifacts.
    # t_btw_artf = (np.array(start_artf_pred)[1:] - np.array(end_artf_pred)[:-1] ) / freq_signal
    # dict_metrics["time_between_artifact"] = np.mean(t_btw_artf)
    
    # # Mean duration of the detected artifacts.
    # dur_time_artf_subj_train = (np.array(end_artf_pred) - np.array(start_artf_pred)) / freq_signal
    # dict_metrics["mean_artifact_duration"] = np.mean(dur_time_artf_subj_train)
    
    # # Minimum duration time of an artifact.
    # dict_metrics["minimum_artifact_duration"] = np.min(dur_time_artf_subj_train)
    
    # # Percentage of artifacts in the signal.
    # perc_of_artf = 100 * np.sum( pred_target_array ) / res_df.shape[0]
    # dict_metrics["percentage_of_artifacts"] = perc_of_artf
    
    # # Total number of artifacts in the signal.
    # n_artf_obtain = len(start_artf_pred)
    # dict_metrics["number_of_artifacts"] = n_artf_obtain

    # # print( "Number of artefacts predicted post-processed", n_artf_obtain )
    
    # # print("Beginning of the interpolation")
    
    # ############################
    # ### AUTOMATIC CORRECTION ### 
    # ############################
    
    
    # start_artf, end_artf = find_begin_end(pred_target_array)
    
    # begin_bad_elements = start_artf
    # end_bad_elements = end_artf
    
    # for ctr_it in range(len(end_bad_elements)):
    #     # 128 / 4 = 32
    #     begin_index = begin_bad_elements[ctr_it] - int(freq_signal / 4)
            
    #     if begin_index < 0:
    #         begin_index = 0
            
    #     end_index = end_bad_elements[ctr_it] + int(freq_signal / 4)
                      
    #     to_clean_segment = res_df[time_column].iloc[begin_index:end_index]
        
    #     to_plot = to_clean_segment
    #     to_clean = res_df[eda_signal].iloc[to_clean_segment.index.values]
        
    #     th_init_space = 0 if begin_bad_elements[ctr_it] == 0 else int(freq_signal/4)-1
        
    #     th_end_space = int(freq_signal/4)
        
    #     initl_pnt = to_clean.iloc[th_init_space]
    #     final_pnt = to_clean.iloc[-th_end_space]

    #     x_all_int = to_clean.index.values
    #     x_int = to_clean[th_init_space:-th_end_space].index.values
    #     y_int = to_clean[th_init_space:-th_end_space].values
        
    #     #########################
    #     ### SPLINE CORRECTION ###
    #     #########################

    #     # returns a callback
    #     f = interp1d([x_int[0], x_int[-1]], [y_int[0], y_int[-1]], kind="linear")
        
    #     intermediam_correct_lineal = f(x_int)
    #     init_correct = to_clean.iloc[:th_init_space] 
    #     final_correct = to_clean.iloc[-th_end_space:]
            
    #     x_to_spline = [x_int[0]] + down_sample(x_int, f = x_int.shape[0] / 8) + [x_int[-1]]
    #     y_to_spline = [y_int[0]] + down_sample(y_int, f = y_int.shape[0] / 8) + [y_int[-1]]
        
        
    #     y_output = spline(x_to_spline, y_to_spline, x_int)
    #     # spl = UnivariateSpline(x_to_spline, y_to_spline, k=x_int)
    #     # y_output = spl(x_to_spline)
    #     # """
    #     # Interpolate a curve at new points using a spline fit
    #     # args:	
    #     #     xk, yk - array_like which is the x and y values that define the curve.
    #     #     xnew - array_like which is the x values where spline should estimate the y values.
    #     #     order - int, Default is 3.
    #     #     kind - string i.e {‘smoothest’}
    #     #     conds - Don’t know
    #     # returns:	
    #     #     spline which is an array of y values; the spline evaluated at the positions xnew.
    #     # """
    #     mix_curve = np.mean([intermediam_correct_lineal, y_output], axis=0)
        
    #     tuple_concat = (mix_curve, final_correct) if init_correct.shape[0] < 2 else (init_correct, mix_curve, final_correct)
    #     correct_linear = np.concatenate(tuple_concat, axis=0)
        
    #     res_df["new_auto_signal"].iloc[to_clean_segment.index.values] = moving_average(correct_linear, freq_signal / 8)

    # return res_df, dict_metrics