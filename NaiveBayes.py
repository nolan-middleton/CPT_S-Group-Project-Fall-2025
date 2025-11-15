#%% Setup

# Imports
from Functions import setup_model, train_naive_bayes, \
    leave_one_out_validation, regular_validation
import numpy as np
import json as json
import sys as sys
import os as os

# Setup Model
training_X,training_Y,do_regular_validation,testing_X,testing_Y=setup_model(
    sys.argv[1]
)

#%% Performing the Evaluation

print("> Naive Bayes on X: " + str(np.shape(training_X)) + "...")
if (do_regular_validation):
    results = regular_validation(
        train_naive_bayes,
        training_X,
        training_Y,
        testing_X,
        testing_Y,
        prior = None
    )
else:
    results = leave_one_out_validation(
        train_naive_bayes,
        training_X,
        training_Y,
        prior = None
    )

#%% Outputting

if (not os.path.isdir(sys.argv[1] + "/Results")):
    os.mkdir(sys.argv[1] + "/Results")
json.dump(results, sys.argv[1] + "/Results/NaiveBayes.json")