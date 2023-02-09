import pandas as pd
import numpy as np

def ms_since_w_b(timestamp):
    """
    Calculate the number of milliseconds that have elapsed since the start of the week for the given timestamp.

    Parameters
    ----------
    timestamp : int
        The timestamp to calculate the elapsed time for, in milliseconds.

    Returns
    -------
    int
        An integer representing the number of milliseconds that have elapsed since the start of the week for the given timestamp.
    """
    timestamp = pd.Timestamp(timestamp, unit='ms')
    week_begin = pd.Period(timestamp, freq='W').start_time
    try:
        ret = (timestamp - week_begin).to_numpy()
        ret = np.timedelta64(ret, "ms")
        return ret.astype(np.int64)
    except:
        # print timestamps that generate errors for investigation
        print(timestamp)


# extract first part
def IP_1(row):
    """
    Return the first octet of an IP address.

    Parameters
    ----------
    row : str
        A string representation of an IP address, in the format "X.X.X.X".

    Returns
    -------
    int
        The first octet of the IP address.

    Raises
    ------
    ValueError
        If the input string is not a valid IP address (i.e., it does not contain four octets).
    """
    parts = row.split('.')
    if len(parts) != 4:
        print(row)
        raise ValueError("Error")
    return int(parts[0])


# extract second part
def IP_2(row):
    """
    Return the second octet of an IP address.

    Parameters
    ----------
    row : str
        A string representation of an IP address, in the format "X.X.X.X".

    Returns
    -------
    int
        The second octet of the IP address.

    Raises
    ------
    ValueError
        If the input string is not a valid IP address (i.e., it does not contain four octets).
    """
    parts = row.split('.')
    if len(parts) != 4:
        raise ValueError("Error")
    return int(parts[1])


# extract third part
def IP_3(row):
    """
    Return the third octet of an IP address.

    Parameters
    ----------
    row : str
        A string representation of an IP address, in the format "X.X.X.X".

    Returns
    -------
    int
        The third octet of the IP address.

    Raises
    ------
    ValueError
        If the input string is not a valid IP address (i.e., it does not contain four octets).
    """
    parts = row.split('.')
    if len(parts) != 4:
        raise ValueError("Error")
    return int(parts[2])


# extract fourth part
def IP_4(row):
    """
    Return the fourth octet of an IP address.

    Parameters
    ----------
    row : str
        A string representation of an IP address, in the format "X.X.X.X".

    Returns
    -------
    int
        The fourth octet of the IP address.

    Raises
    ------
    ValueError
        If the input string is not a valid IP address (i.e., it does not contain four octets).
    """
    parts = row.split('.')
    if len(parts) != 4:
        raise ValueError("Error")
    return int(parts[3])

def get_num_features(features_to_ignore):
    """
    Return the number of features to be used in a model based on certain rules.

    Parameters
    ----------
    features_to_ignore : list of str
        A list of strings representing the features that should be ignored.

    Returns
    -------
    int
        The number of features to be used in the model.
    """
    initial_number = 22
    num_to_ignore = 0
    for feat in features_to_ignore:
        if "log" in feat or "IP" in feat or feat == "Event Name":
            num_to_ignore += 4
        else:
            num_to_ignore += 1

    return initial_number - num_to_ignore