#%% Setup

from Functions import train_naive_bayes, leave_one_out_validation, \
    regular_validation
import numpy as np
import sys as sys
import json as json

priors = [None, [1-0.172, 0.172]] # According to PMID: 36326752

#%%% Command-Line Arguments

# Command-line arguments are, IN ORDER:
# - training_X: The name of the file with the training dataset. MANDATORY.
# - training_Y: The name of the file with the training labels. MANDATORY.
# - out_file: The name of the JSON file to dump the output. MANDATORY.
# - testing_X: The name of the file with the testing dataset. OPTIONAL.
# -- If omitted, will perform leave-one-out validation instead.
# - testing_Y: The name of the file with the testing labels. OPTIONAL.
# -- MUST be specified if testing_pts is provided.

# Mandatory arguments
training_X = np.loadtxt(sys.argv[1], delimiter = "\t")
training_Y = np.loadtxt(sys.argv[2], dtype = int, delimiter = "\t")

out_file = sys.argv[3]

# Optional arguments

do_regular_validation = False

if (len(sys.argv) > 3):
    do_regular_validation = True
    testing_X = np.loadtxt(sys.argv[4], delimiter = "\t")
    testing_Y = np.loadtxt(sys.argv[5], delimiter = "\t")

#%% Performing the evaluation

print("> Naive Bayes on X: " + str(np.shape(training_X)) + "...")
results = {}

for i in range(len(priors)):
    prior = priors[i]
    
    if (prior == None):
        prior_key = "noninformative"
    else:
        prior_key = str(prior[i][1])
    
    print(">> Prior = " + prior_key + "...")
    if (do_regular_validation):
        results[prior_key] = regular_validation(
            train_naive_bayes,
            training_X,
            training_Y,
            testing_X,
            testing_Y,
            prior = prior
        )
    else:
        results[prior_key] = leave_one_out_validation(
            train_naive_bayes,
            training_X,
            training_Y,
            prior = prior
        )

#%% Outputting

json.dump(results, out_file)