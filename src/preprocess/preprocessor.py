"""
A class for preprocessing a dataset. Gives the possibility to either preprocess a dataset from scratch, generating new feature
encoders and transformers based on the provided dataset, or use already created feature encoders and transformers.
"""

import pickle
from pathlib import Path
import os
import mapply
import shutil
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import logging
from src.utils.utils import *

class Preprocessor():
    """
    A class for preprocessing a dataset. Gives the possibility to either preprocess a dataset from scratch, generating new feature
    encoders and transformers based on the provided dataset, or use already created feature encoders and transformers.

    config file specified in configs/preprocessor/default.yaml
    Parameters
    ----------
    dataset_path : str or Path
        The path to the dataset to be preprocessed.
    feat_transformers_path : str or Path
        The path where the feature transformers will be saved or loaded from based on create_feature_transformer.
    output_full_path : str or Path
        The path where the preprocessed dataset will be saved.
    create_feature_transformers : bool
        Whether to create the feature transformers for the dataset.
    use_bytes : bool
        Whether to use the Bytes feature.
    """
    def __init__(self, dataset_path, feat_transformers_path, output_full_path, create_feature_transformers, features_to_ignore=[]):
        self.dataset_path = Path(dataset_path)
        print(dataset_path)
        self.output_path = Path(output_full_path)
        self.feat_transformers_path = Path(feat_transformers_path)
        self.create_feature_transformers = create_feature_transformers
        self.features_to_ignore = features_to_ignore
        print(features_to_ignore)

    def preprocess(self):
        """
        Perform preprocessing on a given dataset.

        Returns
        -------
        None
            The function modifies the DataFrame in place.

        Raises
        ------
        ValueError
            If an input timestamp is not in a valid format.
        """

        logging.info("Starting preprocessing!")
        logging.info("Reading Dataframe!")
        #importing the dataset chunkwise to avoid memory saturation
        chunks = []
        for chunk in pd.read_csv(self.dataset_path, chunksize=55555):
            chunks.append(chunk)

        #a Pandas DataFrame to store the imported Data
        df = pd.concat(chunks)

        logging.info("This script considers the following possible features: 'Bytes (custom)', 'Destination IP', 'Destination Port', 'Event Name', 'Log Source', 'Magnitude', 'Source IP', 'Start Time'.")
        possible_features = ['Bytes (custom)', 'Destination IP', 'Destination Port', 'Event Name', 'Log Source', 'Magnitude', 'Source IP', 'Start Time']

        assert possible_features == list(df.columns), "Dataset either has missing or unnexpected features\n" \
                                                                                                    "Expected features: %s \n" \
                                                                                                    "Dataset features: %s \n" % (str(possible_features), str(list(df.columns)))
        logging.info("It allows to ignore features by specifying them in the features_to_ignore argument.")
        logging.info("Ignored features for this run are: %s" %str(self.features_to_ignore))
        logging.info("Removing bad rows!")
        #removing falsely appended rows: these rows contain the features names apparently after falsely appending chunks of the
        #dataframe
        false_rows_idx = df.loc[(df['Source IP'] == 'Source IP')].index
        df = df.drop(false_rows_idx)

        #saves ip features to scale together later
        ip_features_to_scale_together = []

        logging.info("Step 1/15: Encoding log source ip addresses ...")
        if "Log Source" not in self.features_to_ignore:
            strip = lambda x: x[6:]
            df["Log Source"] = df["Log Source"].apply(strip)
            log_source_splits = df['Log Source'].transform([IP_1, IP_2, IP_3, IP_4], axis=0)
            log_source_col_names = ["Log Source_0", "Log Source_1", "Log Source_2", "Log Source_3"]
            log_source_splits.columns = log_source_col_names
            df = pd.concat([df, log_source_splits], axis=1)

            ip_features_to_scale_together = ip_features_to_scale_together + log_source_col_names
            df = df.drop(columns=["Log Source"])
        else:
            logging.info("Log Source ignored!")


        logging.info("Step 2/15: Encoding Source IP address features ...")
        #Encoding IP address features
        #splitting source ip addresses
        if 'Source IP' not in self.features_to_ignore:
            source_ip_splits = df['Source IP'].transform([IP_1, IP_2, IP_3, IP_4], axis=0)
            source_ip_col_names = ["Source IP_0", "Source IP_1", "Source IP_2", "Source IP_3"]
            source_ip_splits.columns = source_ip_col_names
            df = pd.concat([df, source_ip_splits], axis = 1)

            ip_features_to_scale_together = ip_features_to_scale_together + source_ip_col_names
            df = df.drop(columns=["Source IP"])
        else:
            logging.info("Source IP ignored!")


        logging.info("Step 3/15: Encoding Destination IP address features ...")
        #splitting destination ip addresses
        if 'Destination IP' not in self.features_to_ignore:
            destination_splits = df['Destination IP'].transform([IP_1, IP_2, IP_3, IP_4], axis=0)
            destination_ip_col_names = ["Destination IP_0", "Destination IP_1", "Destination IP_2", "Destination IP_3"]
            destination_splits.columns = destination_ip_col_names
            df = pd.concat([df, destination_splits], axis = 1)
            ip_features_to_scale_together = ip_features_to_scale_together + destination_ip_col_names
            df = df.drop(columns=["Destination IP"])
        else:
            logging.info("Destination IP ignored!")



        logging.info("Step 5/15: Rearranging rows according to temporal feature ...")
        # reorder logs according to timestamp with keeping initial order within same values
        df = df.reset_index().sort_values(by=['Start Time', 'index']).drop(['index'], axis=1)
        # initalize parallel processing library
        mapply.init(
            n_workers=5,
            chunk_size=100,
            max_chunks_per_worker=8,
            progressbar=True
        )


        logging.info("Step 6/15: Calculating milliseconds since week begin ...")
        # calculate milliseconds since week begin for all entries
        # not to be ignored
        df["MS Since Week Begin"] = df["Start Time"].mapply(ms_since_w_b)
        df = df.drop(columns=["Start Time"])

        ############################################# Standardization ##################################################
        logging.info("Step 7/15: Standardization ...")
        if self.create_feature_transformers:
            transformers = {}

            logging.info("Step 7.1/15: Fitting Transformers ...")
            if "Destination Port" not in self.features_to_ignore:
                port_scaler = StandardScaler()
                port_scaler.fit(df[["Destination Port"]])
                transformers["port_scaler"] = port_scaler
            else:
                logging.info("Destination Port ignored!")


            if "Event Name" not in self.features_to_ignore:
                event_encoder = ce.BinaryEncoder(cols=["Event Name"])
                event_encoder.fit(df["Event Name"])
                transformers["binary_encoder_event_name"] = event_encoder
            else:
                logging.info("Event Name ignored!")


            if "Bytes (custom)" in df.columns and "Bytes (custom)" not in self.features_to_ignore:
                bytes_scaler = StandardScaler()
                bytes_scaler.fit(df[["Bytes (custom)"]])
                transformers["bytes_scaler"] = bytes_scaler
            else:
                logging.info("Bytes ignored!")


            if "Magnitude" not in self.features_to_ignore:
                magnitude_scaler = StandardScaler()
                magnitude_scaler.fit(df[['Magnitude']])
                transformers["magnitude_scaler"] = magnitude_scaler
            else:
                logging.info("Magnitude ignored!")


            ms_s_w_b_scaler = StandardScaler()
            ms_s_w_b_scaler.fit(df[['MS Since Week Begin']])
            transformers["ms_s_w_b_scaler"] = ms_s_w_b_scaler


            if len(ip_features_to_scale_together)>0:
                features_list = []
                for i in range(len(ip_features_to_scale_together)):
                    features_list.append(df[ip_features_to_scale_together[i]])

                ips = pd.DataFrame(pd.concat(features_list, axis=0))

                ip_scaler = StandardScaler()
                ip_scaler.fit(ips[[0]])
                transformers["ip_scaler"] = ip_scaler


            dataset_name = os.path.basename(self.dataset_path).split(".")[0]
            logging.info("Step 7.2/13: Saving Transformers to %s ..." % self.feat_transformers_path)
            if os.path.exists(self.feat_transformers_path):
                logging.warning("folder exists, deleting ...")
                shutil.rmtree(self.feat_transformers_path)

            os.mkdir(self.feat_transformers_path)

            for name in transformers:
                pickle.dump(transformers[name], open(self.feat_transformers_path / (name + '.pkl'), 'wb'))

        logging.info("Step 8/15: Standardizing ports ...")
        #Standardization
        #ports
        if "Destination Port" not in self.features_to_ignore or "Destination Port" not in self.features_to_ignore:
            with open(self.feat_transformers_path / "port_scaler.pkl", "rb") as file:
                port_scaler = pickle.load(file)

            df['Destination Port'] = port_scaler.transform(df[['Destination Port']])
        else:
            logging.info("Destination Port ignored!")


        logging.info("Step 9/15: Standardizing magnitude ...")
        #magnitude
        if "Magnitude" not in self.features_to_ignore:
            with open(self.feat_transformers_path / "magnitude_scaler.pkl", "rb") as file:
                magnitude_scaler = pickle.load(file)

            df['Magnitude'] = magnitude_scaler.transform(df[['Magnitude']])
        else:
            logging.info("Magnitude ignored!")


        logging.info("Step 10/15: Standardizing MS Since Week Begin ...")
        # TEMPORAL FEATURE NOT TO BE IGNORED
        #milliseconds since week begin
        with open(self.feat_transformers_path / "ms_s_w_b_scaler.pkl", "rb") as file:
            ms_s_w_b_scaler = pickle.load(file)

        df['MS Since Week Begin'] = ms_s_w_b_scaler.transform(df[['MS Since Week Begin']])


        logging.info("Step 11/15: Standardizing IPs ...")
        #ip addresses
        if len(ip_features_to_scale_together) > 0:
            with open(self.feat_transformers_path / "ip_scaler.pkl", "rb") as file:
                ip_scaler = pickle.load(file)

            for i in range(len(ip_features_to_scale_together)):
                df[ip_features_to_scale_together[i]] = ip_scaler.transform(df[[ip_features_to_scale_together[i]]])
        else:
            logging.info("All IPs ignored!")


        logging.info("Step 12/15: Standardizing Magnitude ...")
        #Encode event name: Binary encoding
        #Using the same encoder used for the training
        if "Event Name" not in self.features_to_ignore:
            with open(self.feat_transformers_path / "binary_encoder_event_name.pkl", "rb") as file:
                encoder = pickle.load(file)

            dfbin = encoder.transform(df["Event Name"])
            df = pd.concat([df, dfbin], axis=1)
            df = df.drop(columns=["Event Name"])
        else:
            logging.info("Event Name ignored!")


        logging.info("Step 13/15: Standardizing Bytes ...")
        if "Bytes (custom)" in df.columns and "Bytes (custom)" not in self.features_to_ignore:
            with open(self.feat_transformers_path / "bytes_scaler.pkl", "rb") as file:
                bytes_scaler = pickle.load(file)

            df['bytes_scaler'] = bytes_scaler.transform(df[['Bytes (custom)']])
        else:
            logging.info("Bytes ignored!")


        logging.info("Step 14/15: Dropping ignored features ...")
        df = df.drop(columns=self.features_to_ignore)

        logging.info("Step 15/15: Saving dataset to %s ..." % self.output_path)
        df.to_csv(self.output_path, index=False)

        logging.info("Preprocessing successfully finished ...")

        return df