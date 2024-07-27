from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import datetime
import pandas as pd
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


