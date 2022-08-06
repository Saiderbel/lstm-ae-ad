import numpy as np
import os
import torch
import sys
import pandas as pd
from pathlib import Path
from src.datamodules.anomaly_dataset import AnomalyDataset
from src.datamodules.data_module import ADDataModule
from src.explainable.ModelWrapper import ModelWrapper
from argparse import ArgumentParser
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from src.models.model import ADModel
import logging
import shap
import nbformat as nbf
import pickle
shap.initjs()


class Explainer():
    def __init__(self, dataset_path, raw_dataset_path, model_ckpt_path, output_path, maes_path, notebook_path, outlier_threshold=0.99):
        #path to dataset in raw format
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.notebook_path = Path(notebook_path)

        #preprocessed dataset given from the preprocessor
        #self.dataset = dataset

        chunks = []
        for chunk in pd.read_csv(dataset_path, chunksize=55555, sep="\t"):
            chunks.append(chunk)
        # a Pandas DataFrame to store the imported Data
        self.dataset = pd.concat(chunks)

        self.raw_dataset_path = raw_dataset_path

        # importing the dataset chunkwise to avoid memory saturation
        chunks = []
        for chunk in pd.read_csv(raw_dataset_path, chunksize=55555, sep="\t"):
            chunks.append(chunk)
        # a Pandas DataFrame to store the imported Data
        self.dataset_raw = pd.concat(chunks)


        lstr_args = ['--batch_size', '128',
                     '--hidden_dim_1', '256',
                     '--strategy', 'dp',
                     '--hidden_dim_2', '128',
                     '--track_grad_norm', '2'
                     ]

        parser = ArgumentParser()
        # program level args
        parser.add_argument('--seed', default=1234, type=int)
        parser.add_argument('--run_name', default='explain', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--seq_len', default=32, type=int)
        parser.add_argument('--hidden_dim_1', default=640, type=int)
        parser.add_argument('--hidden_dim_2', default=320, type=int)
        # trainer level args
        parser = pl.Trainer.add_argparse_args(parser)
        # model level args
        parser = ADModel.add_model_specific_args(parser)
        if lstr_args is None:
            args = parser.parse_args()  # from sys.argv
        else:
            args = parser.parse_args(lstr_args)

        raw_model = ADModel(**vars(args)).load_from_checkpoint(model_ckpt_path)
        raw_model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        raw_model.to(device)

        model = ModelWrapper(raw_model)
        model.eval()
        model.to(device)
        self.model = model

        #read maes dataset
        maes = pd.read_csv(maes_path, sep="\t")
        maes = maes.sort_values(by="mae", ignore_index=True, ascending=True)

        #set mae threshold value
        self.threshold = maes.loc[int(len(maes.index) * outlier_threshold)].values[0]
        print(self.threshold)

        self.device = device


    def explain(self):
        logging.info("Starting explaining process!")

        batch_size = 32

        maes_test = None

        logging.info("Creating Datamodule!")
        #predict on the dataset
        dm = ADDataModule(self.dataset_path, batch_size=batch_size, seq_len=32)
        dm.setup(stage="predict")
        prd_loader = dm.pred_dataloader()

        logging.info("Predicting!")
        total = int(len(self.dataset.index) / batch_size)
        for i, batch in enumerate(prd_loader):
            batch = batch.to(self.device)
            with torch.no_grad():
                output = self.model(batch).view((batch.shape[0]))
                if i == 0:
                    maes_test = torch.cat((torch.zeros((31)), output))
                else:
                    maes_test = torch.cat((maes_test, output))
            sys.stdout.write("Prediction progress: %d / %d   \r" % (i, total))
            sys.stdout.flush()

        logging.info("Explaining!")
        pos_examples = (maes_test > self.threshold).nonzero(as_tuple=True)[0].detach().numpy()

        #raw_model.to(torch.device('cpu'))
        self.model = self.model.to(torch.device('cpu'))

        dataframes = []
        dataframes_raw = []
        for example in pos_examples:
            dataframes.append(self.dataset.iloc[example - 32:example])
            dataframes_raw.append(self.dataset_raw.iloc[example - 32:example].reset_index())

        background_data = torch.tensor(self.dataset.iloc[0:32].to_numpy().astype(np.float32)).unsqueeze(0)

        explained_output = self.output_path / "explained"
        explained_output.mkdir(parents=True, exist_ok=True)

        for i, example in enumerate(pos_examples):
            save_dir = explained_output / str(example)
            save_dir.mkdir(parents=True, exist_ok=True)
            data = torch.tensor(dataframes[i].to_numpy().astype(np.float32)).unsqueeze(0)
            explainer = shap.DeepExplainer(self.model, background_data)
            shap_values = explainer.shap_values(data).reshape(32, 22)
            expected = explainer.expected_value
            with open(save_dir / "shap_values.npy", "wb") as file:
                pickle.dump(shap_values, file)

            #aggregating features for complete feature imporatances
            source_port = np.expand_dims(shap_values[:, 0], 1)
            destination_port = np.expand_dims(shap_values[:, 1], 1)
            magnitude = np.expand_dims(shap_values[:, 2], 1)
            event_name = shap_values[:, 3:12]
            source_ip = shap_values[:, 13:16]
            destination_ip = shap_values[:, 17:20]
            ms_since_week_begin = np.expand_dims(shap_values[:, 21], 1)

            event_name = np.expand_dims(np.sum(event_name, axis=1), 1)
            source_ip = np.expand_dims(np.sum(source_ip, axis=1), 1)
            destination_ip = np.expand_dims(np.sum(destination_ip, axis=1), 1)

            new_shap = np.hstack((source_port, destination_port, magnitude, event_name, source_ip, destination_ip, ms_since_week_begin))

            with open(save_dir / "shap_values_agg.npy", "wb") as file:
                pickle.dump(new_shap, file)

        nb = nbf.v4.new_notebook()

        cell_1 = """\
import pandas as pd
import shap
from pathlib import Path
import pickle
import numpy as np"""

        cell_2 = """
## Anomaly explainer dashboard\

After running the model on the provided dataset, there has been %d anomalies found.
To get the plots and the relative dataset occurrence, just indicate the anomaly index and run the following cells""" % len(pos_examples)

        anomalies = "["
        for example in pos_examples:
            anomalies += str(example) + ", "

        anomalies = anomalies[:-2] + "]"

        cell_3 = """anomalies = %s""" % anomalies

        cell_33 = """\
#chose here the index of the anomalie to explain
anomaly_to_explain_id = 0"""

        cell_4 = """\
columns = ['Source Port', 'Destination Port', 'Magnitude', 'Event Name_0',
'Event Name_1', 'Event Name_2', 'Event Name_3', 'Event Name_4',
'Event Name_5', 'Event Name_6', 'Event Name_7', 'Event Name_8',
'Event Name_9', 'Source_IP_0', 'Source_IP_1', 'Source_IP_2',
'Source_IP_3', 'Destination_IP_0', 'Destination_IP_1',
'Destination_IP_2', 'Destination_IP_3', 'ms_since_week_begin']

columns_raw = ["source_port", "destination_port", "magnitude", "event_name", "source_ip", "destination_ip",
                 "ms_since_week_begin"]
                 
colums_extended = []
for i in range(32):
    for j in range(7):
        if i != 31:
            colums_extended.append(columns_raw[j] + " (t-" + str(31 - i) + ") " + " i=" + str(i))
        else:
            colums_extended.append(columns_raw[j] + " (t) " + " i=" + str(i))"""
        cell_5 = """\
#importing the dataset chunkwise to avoid memory saturation
chunks = []
for chunk in pd.read_csv("%s", chunksize=55555, sep = "\t"):
    chunks.append(chunk)
#a Pandas DataFrame to store the imported Data
df_raw = pd.concat(chunks)""" % self.raw_dataset_path

        cell_6 = """\
shap_dir = Path("%s") / str(anomalies[anomaly_to_explain_id])

with open(shap_dir / "shap_values.npy", "rb") as file:
    shap_values = pickle.load(file)
    
with open(shap_dir / "shap_values_agg.npy", "rb") as file:
    shap_values_agg = pickle.load(file)""" % str(explained_output)

        cell_7 = """### raw feature importances"""

        cell_8 = """shap.bar_plot(np.sum(shap_values, axis=0), feature_names=columns)"""

        cell_9 = """### Aggregated feature importances"""

        cell_10 = """shap.bar_plot(np.sum(shap_values_agg, axis=0), feature_names=columns_raw)"""

        cell_11 = """### Temporalized and aggregated feature importances"""

        cell_12 = """shap.bar_plot(shap_values_agg.flatten(), feature_names=colums_extended)"""

        cell_13 = """### Dataset slice of the occured anomaly"""

        cell_14 = """df_raw.iloc[anomalies[anomaly_to_explain_id] - 32:anomalies[anomaly_to_explain_id]].reset_index()"""

        nb['cells'] = [nbf.v4.new_code_cell(cell_1),
                       nbf.v4.new_markdown_cell(cell_2),
                       nbf.v4.new_code_cell(cell_3),
                       nbf.v4.new_code_cell(cell_33),
                       nbf.v4.new_code_cell(cell_4),
                       nbf.v4.new_code_cell(cell_5),
                       nbf.v4.new_code_cell(cell_6),
                       nbf.v4.new_markdown_cell(cell_7),
                       nbf.v4.new_code_cell(cell_8),
                       nbf.v4.new_markdown_cell(cell_9),
                       nbf.v4.new_code_cell(cell_10),
                       nbf.v4.new_markdown_cell(cell_11),
                       nbf.v4.new_code_cell(cell_12),
                       nbf.v4.new_markdown_cell(cell_13),
                       nbf.v4.new_code_cell(cell_14)
                       ]

        fname = self.notebook_path / 'explainer.ipynb'

        with open(fname, 'w') as f:
            nbf.write(nb, f)

