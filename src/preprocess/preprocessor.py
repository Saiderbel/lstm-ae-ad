import pickle
import pandas as pd
from pathlib import Path
import os
import numpy as np
import mapply
import multiprocessing
import category_encoders as ce
import logging


def ms_since_w_b(timestamp):
    """
    Returns number of milliseconds since the week begin
    :param timestamp: UNIX timestamp
    return the number of seconds since week begin
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
    parts = row.split('.')
    if len(parts) != 4:
        print(row)
        raise ValueError("Error")
    return int(parts[0])


# extract second part
def IP_2(row):
    parts = row.split('.')
    if len(parts) != 4:
        raise ValueError("Error")
    return int(parts[1])


# extract third part
def IP_3(row):
    parts = row.split('.')
    if len(parts) != 4:
        raise ValueError("Error")
    return int(parts[2])


# extract fourth part
def IP_4(row):
    parts = row.split('.')
    if len(parts) != 4:
        raise ValueError("Error")
    return int(parts[3])

class Preprocessor():
    def __init__(self, dataset_path, feat_transformers_path, output_full_path):
        """
        Preprocesses data for testing and explaining, assumes training data is already preprocessed
        and the feature transformers/encoders have been saved using pickle.
        :param dataset_path: path to dataset to preprocess
        :param feat_transformers_path: path to feature transformers/encoders
        :param output_full_path: output path
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_full_path)
        self.feat_transformers_path = Path(feat_transformers_path)

    def preprocess(self):
        logging.info("Starting preprocessing!")
        logging.info("Reading Dataframe!")
        #importing the dataset chunkwise to avoid memory saturation
        chunks = []
        for chunk in pd.read_csv(self.dataset_path, sep = "\t", chunksize=55555):
            chunks.append(chunk)

        #a Pandas DataFrame to store the imported Data
        df = pd.concat(chunks)

        logging.info("Removing bad rows!")
        #removing falsely appended rows: these rows contain the features names apparently after falsely appending chunks of the
        #dataframe
        false_rows_idx = df.loc[(df['Source IP'] == 'Source IP')].index
        df = df.drop(false_rows_idx)

        logging.info("Step 1/11: Encoding event name ...")
        #Encode event name: Binary encoding
        #Using the same encoder used for the training
        with open(self.feat_transformers_path / "binary_encoder_event_name.pkl", "rb") as file:
            encoder = pickle.load(file)

        dfbin = encoder.transform(df["Event Name"])
        df = pd.concat([df, dfbin], axis=1)

        logging.info("Step 2/11: Deleting irrelevant features ...")
        #Deleting irrelevant Features
        df = df.drop(columns=["Log Source", "Low Level Category"])

        logging.info("Step 3/11: Encoding source IP address features ...")
        #Encoding IP address features
        #splitting source ip addresses
        source_splits = df['Source IP'].transform([IP_1, IP_2, IP_3, IP_4], axis=0)

        source_splits.columns = ["Source_IP_0", "Source_IP_1", "Source_IP_2", "Source_IP_3"]

        df = pd.concat([df, source_splits], axis = 1)

        logging.info("Step 4/11: Encoding destination IP address features ...")
        #splitting destination ip addresses
        destination_splits = df['Destination IP'].transform([IP_1, IP_2, IP_3, IP_4], axis=0)

        destination_splits.columns = ["Destination_IP_0", "Destination_IP_1", "Destination_IP_2", "Destination_IP_3"]

        df = pd.concat([df, destination_splits], axis = 1)

        logging.info("Step 5/11: Dropping raw features ...")
        df = df.drop(columns=["Source IP", "Destination IP"])
        df = df.drop(columns=["Event Name"])

        logging.info("Step 6/11: Rearranging rows according to temporal feature ...")
        # reorder logs according to timestamp with keeping initial order within same values
        df = df.reset_index().sort_values(by=['Start Time', 'index']).drop(['index'], axis=1)

        # initalize parallel processing library
        mapply.init(
            n_workers=6,
            chunk_size=100,
            max_chunks_per_worker=8,
            progressbar=True
        )

        logging.info("Step 7/11: Calculating milliseconds since week begin ...")
        # calculate milliseconds since week begin for all entries
        df["ms_since_week_begin"] = df["Start Time"].mapply(ms_since_w_b)

        df = df.drop(columns=["Start Time"])

        logging.info("Step 8/11: Standardizing ports ...")
        #Standardization
        #ports
        with open(self.feat_transformers_path / "port_scaler.pkl", "rb") as file:
            port_scaler = pickle.load(file)

        df['Source Port'] = port_scaler.transform(df[['Source Port']])
        df['Destination Port'] = port_scaler.transform(df[['Destination Port']])

        logging.info("Step 9/11: Standardizing magnitude ...")
        #magnitude
        with open(self.feat_transformers_path / "magnitude_scaler.pkl", "rb") as file:
            magnitude_scaler = pickle.load(file)

        df['Magnitude'] = magnitude_scaler.transform(df[['Magnitude']])

        logging.info("Step 10/11: Standardizing ms_since_week_begin ...")
        #milliseconds since week begin
        with open(self.feat_transformers_path / "ms_s_w_b_scaler.pkl", "rb") as file:
            ms_s_w_b_scaler = pickle.load(file)

        df['ms_since_week_begin'] = ms_s_w_b_scaler.transform(df[['ms_since_week_begin']])

        logging.info("Step 11/11: Standardizing IPs ...")
        #ip addresses
        with open(self.feat_transformers_path / "ip_scaler.pkl", "rb") as file:
            ip_scaler = pickle.load(file)

        features_to_scale_together = ["Source_IP_0", "Source_IP_1", "Source_IP_2", "Source_IP_3", "Destination_IP_0",
                                      "Destination_IP_1", "Destination_IP_2", "Destination_IP_3"]

        for i in range(len(features_to_scale_together)):
            df[features_to_scale_together[i]] = ip_scaler.transform(df[[features_to_scale_together[i]]])

        df.to_csv(self.output_path, index=False, sep="\t")

        logging.info("Preprocessing successfully finished ...")
