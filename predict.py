import os
import dill

import argparse
import numpy as np
import pandas as pd

from src import config as cfg


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

#####
with open(os.path.join(cfg.final_models_path, cfg.final_pipeline_filename), 'rb') as f:
    pipeline = dill.load(f)
    
with open(os.path.join(cfg.final_models_path, cfg.final_model_filename), 'rb') as f:
    model = dill.load(f)
    
X = pipeline.transform(data)
preds = model.predict(X)

# preds = np.expm1(model.predict(X))


# Example:
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = preds

prediction_df['revenue'] = prediction_df['revenue'].apply(np.expm1)

####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Utility function to calculate RMSLE
# def rmsle(y_true, y_pred):
#     """
#     Calculates Root Mean Squared Logarithmic Error between two input vectors
#     :param y_true: 1-d array, ground truth vector
#     :param y_pred: 1-d array, prediction vector
#     :return: float, RMSLE score between two input vectors
#     """
#     assert y_true.shape == y_pred.shape, \
#         ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
#     return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))
#
#
# ### Example - Calculating RMSLE
# res = rmsle(data['revenue'], prediction_df['revenue'])
# print("RMSLE is: {:.6f}".format(res))


