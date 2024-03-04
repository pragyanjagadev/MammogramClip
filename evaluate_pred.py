import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import sys

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval

predictions_dir: Path = Path('/home/woody/iwi5/iwi5176h/predictions') # where to save predictions
cxr_true_labels_path: Optional[str] = 'data/groundtruth.csv'

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['Target']

print("130==============")
# make test_true
y_pred_avg = np.load(predictions_dir)
print(y_pred_avg[0])
test_pred = y_pred_avg
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
print(test_true)
print(type(test_true))
print(test_true.shape)
print(type(test_pred))
print(test_pred.shape)
sys.exit()
# evaluate model
print("136==============Eavaluation started")
cxr_results = evaluate(test_pred, test_true, cxr_labels)
print(type(cxr_results))
cxr_results.to_csv('cxr_results.csv', index=False)
# boostrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)
print(type(bootstrap_results))
bootstrap_results.to_csv('bootstrap_results.csv', index=False)
# display AUC with confidence intervals
bootstrap_results[1]