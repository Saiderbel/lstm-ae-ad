import numpy as np
import torch
import sys
import pandas as pd
from pathlib import Path
from src.datamodules.data_module import ADDataModule
from src.explainable.ModelWrapper import ModelWrapper
from src.models.model import ADModel
import logging
import shap
shap.initjs()
import nbformat as nbf
import warnings
warnings.filterwarnings("ignore")
import pickle



class Explainer():
    """
    Explainer class for generating explanantions of the anomaly detection module.

    config file specified in configs/explainer/default.yaml
    """

    def __init__(self, dataset_path, raw_dataset_path, model_ckpt_path, output_path, maes_path, notebook_path, features_to_ignore=[], outlier_threshold=0.99):
        """
        Initialize the Explainer instance.

        Parameters
        ----------
        dataset_path : str
            The path to the preprocessed dataset.
        raw_dataset_path : str
            The path to the raw dataset.
        model_ckpt_path : str
            The path to the checkpoint file for the pre-trained model.
        output_path : str
            The path to the directory where the explanations should be saved.
        maes_path : str
            The path to the file containing the mean absolute errors for the train set.
        notebook_path : str
            The path to the notebook that will be used to display the explanations.
        outlier_threshold : float
            The fraction of the dataset that is considered normal (not anomalous).
        features_to_ignore : List[str]
            The features to ignore.
        """
        #path to dataset in raw format
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.notebook_path = Path(notebook_path)
        warnings.filterwarnings("ignore")
        #preprocessed dataset given from the preprocessor
        #self.dataset = dataset

        chunks = []
        for chunk in pd.read_csv(dataset_path, chunksize=55555):
            chunks.append(chunk)
        # a Pandas DataFrame to store the imported Data
        self.dataset = pd.concat(chunks)

        self.raw_dataset_path = raw_dataset_path

        # importing the dataset chunkwise to avoid memory saturation
        chunks = []
        for chunk in pd.read_csv(raw_dataset_path, chunksize=55555):
            chunks.append(chunk)
        # a Pandas DataFrame to store the imported Data
        self.dataset_raw = pd.concat(chunks)
        self.dataset_raw = self.dataset_raw.drop(columns=features_to_ignore)
        self.raw_columns = list(self.dataset_raw.columns)

        raw_model = ADModel.load_from_checkpoint(model_ckpt_path)
        raw_model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        raw_model.to(device)

        model = ModelWrapper(raw_model)
        model.eval()
        model.to(device)
        self.model = model


        #read maes
        maes = torch.load(maes_path)

        #set mae threshold value
        self.threshold = maes[int(len(maes) * outlier_threshold)].item()

        self.device = device
        self.features_to_ignore = features_to_ignore


    def explain(self):
        """
        Generate explanations for the predictions made by the model on the dataset.

        This method generates explanations for the predictions made by the model on the dataset, and saves them to
        a notebook in `output_path`.
        """
        logging.info("Starting explaining process!")
        batch_size = self.model.ad_module.hparams.batch_size
        seq_length = self.model.ad_module.hparams.seq_length
        num_features = self.model.ad_module.hparams.num_features

        maes_test = None

        logging.info("Creating Datamodule!")
        #predict on the dataset
        dm = ADDataModule(self.dataset_path, batch_size=batch_size, seq_len=seq_length)
        dm.setup(stage="predict")
        prd_loader = dm.predict_dataloader()

        logging.info("Predicting!")
        total = int(len(self.dataset.index) / batch_size)
        for i, batch in enumerate(prd_loader):
            batch = batch.to(self.device)
            with torch.no_grad():
                output = self.model(batch).view((batch.shape[0]))
                if i == 0:
                    maes_test = torch.cat((torch.zeros((seq_length - 1)), output))
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
            dataframes.append(self.dataset.iloc[example - seq_length:example])
            dataframes_raw.append(self.dataset_raw.iloc[example - seq_length:example].reset_index())

        background_data = torch.tensor(self.dataset.iloc[0:seq_length].to_numpy().astype(np.float32)).unsqueeze(0)

        explained_output = self.output_path / "explained"
        explained_output.mkdir(parents=True, exist_ok=True)

        columns = list(self.dataset.columns)
        #features_to_consider = ['Bytes (custom)', 'Destination IP', 'Destination Port', 'Event Name', 'Log Source', 'Magnitude', 'Source IP', 'MS Since Week Begin']
        for i, example in enumerate(pos_examples):
            save_dir = explained_output / str(example)
            save_dir.mkdir(parents=True, exist_ok=True)
            data = torch.tensor(dataframes[i].to_numpy().astype(np.float32)).unsqueeze(0)


            explainer = shap.DeepExplainer(self.model, background_data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(data).reshape(self.model.ad_module.hparams.seq_length, num_features)

            with open(save_dir / "shap_values.npy", "wb") as file:
                pickle.dump(shap_values, file)

            features_importances = {}

            #aggregating features for complete feature imporatances
            for feature in self.raw_columns:
                if "Log" in feature or "IP" in feature or feature == "Event Name":
                    # multicolumns featues
                    idx = [i for i in range(len(columns)) if feature in columns[i]]
                    importances = shap_values[:, idx]
                    importances = np.expand_dims(np.sum(importances, axis=1), 1)
                else:
                    #single column feature
                    if feature == "Start Time":
                        importances = np.expand_dims(shap_values[:, columns.index("MS Since Week Begin")], 1)
                    else:
                        importances = np.expand_dims(shap_values[:, columns.index(feature)], 1)


                features_importances[feature] = importances


            aggregated_feature_importances = ()
            for column in self.raw_columns:
                aggregated_feature_importances += (features_importances[column],)


            aggregated_feature_importances = np.hstack(aggregated_feature_importances)

            with open(save_dir / "shap_values_agg.npy", "wb") as file:
                pickle.dump(aggregated_feature_importances, file)

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
columns = %s

columns_raw = %s
                 
colums_extended = []
for i in range(%d):
    for j in range(7):
        if i != 31:
            colums_extended.append(columns_raw[j] + " (t-" + str(31 - i) + ") " + " i=" + str(i))
        else:
            colums_extended.append(columns_raw[j] + " (t) " + " i=" + str(i))""" %(str(list(self.dataset.columns)), str(self.raw_columns), self.model.ad_module.hparams.seq_length)
        cell_5 = """\
#importing the dataset chunkwise to avoid memory saturation
chunks = []
for chunk in pd.read_csv("%s", chunksize=55555):
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

        cell_14 = """df_raw.iloc[anomalies[anomaly_to_explain_id] - %d:anomalies[anomaly_to_explain_id]].reset_index()""" %(self.model.ad_module.hparams.seq_length)

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

if __name__ == '__main__':
    exp = Explainer(dataset_path="/tmp/ad/data/test_pipeline_pro.csv",
                    raw_dataset_path="/tmp/ad/data/2022-12-20_10-07-58_log-data.csv",
                    model_ckpt_path="/tmp/ad/logs/experiments/runs/training/2022-12-20_19-12-17/checkpoints/epoch_002.ckpt",
                    output_path="/tmp/ad/data",
                    notebook_path="/tmp/ad/data",
                    maes_path="/tmp/ad/data/maes.csv",
                    outlier_threshold=0.97)
    exp.explain()