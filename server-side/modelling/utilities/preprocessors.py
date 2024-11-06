import datetime
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, splrep, UnivariateSpline, BSpline, splev
from scipy.signal import butter, filtfilt, lfilter, firwin, hilbert, sosfiltfilt

import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

def down_sample(signals: pd.DataFrame | np.ndarray | list, target_freq=16):
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

def moving_average(array, window_size=7):
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

def correct_signals(y_pred, df, selector_config, estimator_name, target_size_freq=64, freq_signal=128, th_t_postprocess=2.5, signal_column="raw_signal", time_column="time"):
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

        selector_config  - 

        estimator_name - 

        target_size_freq - sampling frequency or how many rows per seconds
        must be taken into account for the label

        freq_signal - sampling frequency of the raw signals

        th_t_postprocess - represents the maximum time used to link two
        artifacts which temporal distance is below or equal that time

        signal_column - column in the dataframe in which to correct
        
    """
    # copy contents of dataframe to ensure no modifications
    # are made in original dataframe
    res_df = df.copy()

    # the raw_signals that are less than 0 which will be likely be negative values 
    # will be rendered true and those raw signal rows with true booleans will just be
    # transformed to 0
    res_df.loc[res_df[signal_column] < 0, signal_column] = 0
    
    res_df["new_signal"] = res_df[signal_column].iloc[:]
    
    rawdata_spline_correct = res_df[signal_column].iloc[:]

    

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
        start = target_size_freq * label_i
        end = min(target_size_freq * (label_i + 1), n_rows) 

        if label_i == 0 or (label_i == y_pred.shape[0] - 1):    
            print(f"index {label_i}: start {start} - end {end}")
        
        future_labels_auto[start:end] += label
    
    # ##############################
    # ### TARGET POST-PROCESSING ###
    # ##############################

    """this seems to have no use as pred_target_array
    is just assigned to another variable again"""
    pred_target_array = res_df[signal_column].iloc[:].copy()
    
    # future labels will contain 1s and 0s and those greater than 0 will always be 1
    # and those rows must be set to 1
    future_labels_auto[future_labels_auto > 0] = 1

    """and this seems redundant"""
    pred_target_array = future_labels_auto

    res_df["pred_art"] = pred_target_array

    # pred_target_array overall is an array of 1s and 0s
    # where it is passed to find_begin_end() which uses a function which_element() that in a boolean numpy array or a dataframe
    # not a list since [1, 0, 1, 0, 0] == 1 for instance
    # would just return false
    # For instance if we had now signals that are labeled as
    # artifacts or non artifacts...
    # >>> signal = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1])
    # >>> signal == 1
    # array([False, False, True, True, True, False, False, True, True])
    # when this is passed through which_element() this will
    # return [2, 3, 4, 7, 8]
    # pos_artf_true = which_element(x_p == 1)

    # overall find_begin_end() returns to lists, the first representing the 
    # beginning indices of all artifacts in a subjects signals 
    # e.g. if a subjects signals that have been predicted for each 0.5s segment to
    # have [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1] which total to 
    # 2 seconds all in all of this subjects signals, the starting indices of these subjects
    # signals artifacts will be collected as well as the end indices
    # in this example start_artf_pred will be [3, 9, 13, 17, 21]
    # and end_artf_pred will be [5, 11, 14, 18, 22]
    start_artf_pred, end_artf_pred = find_begin_end(pred_target_array)
    # print('unmerged predicted artifacts: ')
    # print(f'start indeces of unmerged predicted artifacts: {start_artf_pred} \n length: {len(start_artf_pred)}')
    # print(f'end indeces of unmerged predicted artifacts: {end_artf_pred} \n length: {len(end_artf_pred)}\n')
    
    # length of start_artf_pred is 5 for instance
    # 5 - 1 is 4 so therefore we would only be looping from 0 to 3 and give
    # allowance for the last index 4 as we will be using an index of i + 1
    # which takes into account a potential out of range error
    for i in range(len(start_artf_pred) - 1):
        # links and merges artifacts in the signals that are close together
        if np.abs(start_artf_pred[i + 1] - end_artf_pred[i]) / freq_signal <= th_t_postprocess:
            pred_target_array[end_artf_pred[i]:start_artf_pred[i + 1]] = 1
    
    # once merged we find again the begin indeces and end indeces of these artifacts
    start_artf_pred, end_artf_pred = find_begin_end(pred_target_array)
    # print('merged predicted artifacts: ')
    # print(f'start indeces of merged predicted artifacts: {start_artf_pred} \n length: {len(start_artf_pred)}')
    # print(f'end indeces of merged predicted artifacts: {end_artf_pred} \n length: {len(end_artf_pred)}\n')
    
    res_df["post_proc_pred_art"] = pred_target_array
    
    # #################################
    # ### Compute artifacts metrics ###
    # #################################
    
    dict_metrics = {}
    
    # Time until first artifact.
    dict_metrics["time_first_artifact"] = start_artf_pred[0] / freq_signal
    
    # Mean time between two artifacts.
    t_btw_artf = (np.array(start_artf_pred)[1:] - np.array(end_artf_pred)[:-1]) / freq_signal
    dict_metrics["time_between_artifact"] = np.mean(t_btw_artf)
    
    # Mean duration of the detected artifacts.
    dur_time_artf_subj_train = (np.array(end_artf_pred) - np.array(start_artf_pred)) / freq_signal
    dict_metrics["mean_artifact_duration"] = np.mean(dur_time_artf_subj_train)
    
    # Minimum duration time of an artifact.
    dict_metrics["minimum_artifact_duration"] = np.min(dur_time_artf_subj_train)
    
    # Percentage of artifacts in the signal.
    perc_of_artf = 100 * np.sum(pred_target_array) / res_df.shape[0]
    dict_metrics["percentage_of_artifacts"] = perc_of_artf
    
    # Total number of artifacts in the signal.
    n_artf_obtain = len(start_artf_pred)
    dict_metrics["number_of_artifacts"] = n_artf_obtain

    # print("Number of artefacts predicted post-processed", n_artf_obtain)
    
    
    
    # ############################
    # ### AUTOMATIC CORRECTION ### 
    # ############################
    
    print("commencing interpolation...")

    # find begin and end indeces of post processed predicted artifacts
    # again
    start_artf_pred, end_artf_pred = find_begin_end(pred_target_array)
    begin_bad_elements, end_bad_elements = start_artf_pred, end_artf_pred

    # loop thorugh the begin adn end indeces
    for ctr_it in range(len(end_bad_elements)):
        # # create two figures
        # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # axes = axes.flat
        # fig.tight_layout(pad=1)
        

        # 128 / 4 = 32
        # start indeces of post processed/merged artifacts for pqbqpr subject is 0, 8768, 10496, ..., 902592
        # start indeces of post processed/merged artifacts for pqbqpr subject is 63, 8831, 12351, ..., 902719
        # this means that we have an artifact at signals [0] to [63], [8768] to [8831], and [902592] to [902719]
        # we subtract these end indeces of the artifacts by the frequency signal of 128 divided by 4 which is 32
        # but we will like with a number 0 potentially inevatibly subtract by 32 which will result in a negative value
        # so if this is the case 0 - 32 = -32 then -32 is then just turned to 0 again
        # but in this case the pattern during iteration for the end indeces would be:
        # 0 - 32 = -32
        # 8768 - 32 = 8736
        # 10496 - 32 = 10464
        # ...
        # 902592 - 32 = 902560
        # but my question is why do we even subtract by 32 these end indeces?
        begin_index = begin_bad_elements[ctr_it] - int(freq_signal / 4)
        # series of begin indeces would now be 0, 8736, 10464, ..., 902560
        
        if begin_index < 0:
            begin_index = 0
        

        # 63 + 32 = 95
        # 8831 + 32 = 8863
        # 12351 + 32 = 12383
        # ...
        # 902719 + 32 = 902751
        # and now another question is why even add 32 to these 
        end_index = end_bad_elements[ctr_it] + int(freq_signal / 4)
        # series of end indeces would now be 95, 8863, 12383, 902751

        # all in all after the index positions of the artifacts in the signals 
        # will now be at [0] to [95], 
        # [8736] to [8863], 
        # [10464] to [12383], 
        # ...
        # [902560] to [902751]
        
        # we now get the time values based on these new begin and end indeces
        to_clean_segment = res_df[time_column].iloc[begin_index:end_index]
        # print(f'time to clean: {to_clean_segment.to_list()}')
        # print(f'segment to clean index values: {to_clean_segment.index.values}')
        # segment to clean index values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
        # 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
        # 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94]
        
        # unused variable
        to_plot = to_clean_segment

        to_clean = res_df[signal_column].iloc[to_clean_segment.index.values]
        # print(f'signals to clean: {to_clean.to_list()}')
        # [0.0, 0.0, 0.0, ..., 0.0002220002220002]
        
        # 0 if the beginning of an artifact has index 0 and if not 128 / 4 -> 32 -1 -> 31 is used
        # th initial space is what this variable is called
        # 0 == 0 is true so 0 is returned
        # 8736 == 0 is false so 31 is returned
        # 10464 == 0 is false so 31 is returned
        # ...
        # 902560 == 0 is false so 31 is returned
        th_init_space = 0 if begin_bad_elements[ctr_it] == 0 else int(freq_signal / 4) - 1
        
        # 128 / 4 = 32
        th_end_space = int(freq_signal / 4)
        
        # # we pick out initial point using either 0 or 31
        # # which for some reason is a variable unused
        # initl_pnt = to_clean.iloc[th_init_space]

        # # final point which for some reason is unused
        # # will always be -32 or depending on the frequency signals will be -(freq_signal / 4)
        # final_pnt = to_clean.iloc[-th_end_space]

        # # also unused
        # x_all_int = to_clean.index.values

        # we ought to use as x values are the indeces or the time value
        # but in this case we use just the index values
        # [0 or 31:-32]
        x_int = to_clean[th_init_space:-th_end_space].index.values
        # print(f'x_int: {x_int} - length: {x_int.shape[0]}')

        # as we know we ought to use as y values are the signals themselves
        y_int = to_clean[th_init_space:-th_end_space].values
        # print(f'y_int: {y_int} - length: {y_int.shape[0]}')
        
        #########################
        ### SPLINE CORRECTION ###
        #########################

        # returns a callback
        # x and y are arrays of values used to approximate some 
        # function f: y = f(x). This class returns a function whose
        # call method uses interpolation to find the value of new points.
        f = interp1d([x_int[0], x_int[-1]], [y_int[0], y_int[-1]], kind="linear")

        # there seems to be the x and y points that exist
        # where some or one of its points is used as input to an
        # interpolation function like the one above in order to make new
        # points in between these points, such that these new points are
        # smoother 
        # here x_int and y_int are the old points or the points that
        # currently exist which need to be smoothed
        intermediam_correct_lineal = f(x_int)
         
        init_correct = to_clean.iloc[:th_init_space]
        final_correct = to_clean.iloc[-th_end_space:]
        
        # x_to_spline and y_to_spline seem to be also points that
        # server as replacement to the previous points x_int, and y_int
        # my understanding with this is that why x_int and y_int are created 
        # is that as these points are from the 128hz signals they can be down
        # sampled further to create a shorter list of values i.e. 
        # x_int = [89023 89024 89025 ... 89086], y_int = [2.58095238 2.57216117 2.57655678 ... 2.79194139]
        # to x_to_spline = [89023 89026.5 89034.5 89042.5 89050.5 89058.5 89066.5 89074.5 89082.5 89086]
        # of length 10 and y_to_spline = [2.58095238 2.577289377289378 2.5860805860805858 ... 2.79194139]
        # also of length 10
        x_to_spline = [x_int[0]] + down_sample(x_int, target_freq=x_int.shape[0] / 8) + [x_int[-1]]
        y_to_spline = [y_int[0]] + down_sample(y_int, target_freq=y_int.shape[0] / 8) + [y_int[-1]]
        print(f'x_to_spline: {x_to_spline} - length: {len(x_to_spline)}')
        print(f'y_to_spline: {y_to_spline} - length: {len(y_to_spline)}')

        # axes[0].scatter(x_int, y_int, c="#eb34b7", marker='o')
        # axes[0].plot(x_int, y_int, c="#7c31de", linestyle="-")
        # axes[1].scatter(x_to_spline, y_to_spline, c="#eb34b7", marker='o')
        # axes[1].plot(x_to_spline, y_to_spline, c="#7c31de", linestyle="-")
        
        # scipy.interpolate.spline(*args, **kwds)
        # spline is deprecated! spline is deprecated in scipy 0.19.0, use Bspline class instead.
        # Interpolate a curve at new points using a spline fit
        # Parameters:	
        # xk, yk: array_like - The x and y values that define the curve.
        # xnew: array_like - The x values where spline should estimate the y values.
        # order: int - Default is 3.
        # kind: string - One of {‘smoothest’}

        # Returns:	
        # spline: ndarray - An array of y values; the spline evaluated at the positions xnew.
        # y_output = spline(x_to_spline, y_to_spline, x_int)

        # class BSpline(t, c, k, extrapolate=True, axis=0)[source]
        # Univariate spline in the B-spline basis.
        # where are B-spline basis functions of degree k and knots t.
        # Parameters:
        # t: ndarray, shape (n+k+1,) - knots
        # c: ndarray, shape (>=n, …) - spline coefficients
        # k: int - B-spline degree
        # spline = BSpline(x_to_spline, y_to_spline, k=3)
        tck = splrep(x_to_spline, y_to_spline)
        

        """
        # # Create a B-spline object
        # t = np.linspace(x_int[0], x_int[-1], num=4)  # Assuming 4 knots for a cubic spline
        # c = [y_int[0], y_int[1], y_int[-2], y_int[-1]]  # Assuming cubic spline
        # k = 3  # Cubic spline
        """
        # y_output = spline(x_int)
        y_output = splev(x_int, tck, der=0)
        # axes[2].scatter(x_int, y_output, c="#236cb0", marker="^")
        # axes[2].plot(x_int, y_output, c="#3de343", linestyle="--")
        
        # mixes the average values between the line that uses a linear graph
        # and a line that uses a cubic graph or a degree of 3 as used above
        mix_curve = np.mean([y_output], axis=0)
        
        tuple_concat = (mix_curve, final_correct) if init_correct.shape[0] < 2 else (init_correct, mix_curve, final_correct)
        correct_linear = np.concatenate(tuple_concat, axis=0)
        # print(f'corrected linear: {correct_linear}')
        # print(f'corrected linear shape: {correct_linear.shape}')
        
        
        # 128 / 8 as the window size is 16 or 0.25 seconds
        corrected_segment = moving_average(correct_linear, freq_signal / 8)
        res_df["new_signal"].iloc[to_clean_segment.index.values] = corrected_segment

        # plt.show()
        # plt.savefig(f'./figures & images/segment splined {begin_index} to {end_index}.jpg')

    return res_df, dict_metrics