#%% Setup

from Functions import train_random_forest, leave_one_out_validation, \
    regular_validation
import numpy as np
import sys as sys
import json as json

depths = [1,2,3,4,5,6] # The max_depths to test
n_estimators = [100] + [500 * n for n in range(1,10)] # n_estimators to test

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

print("> Random Forest on X: " + str(np.shape(training_X)) + "...")
results = {}

for depth in depths:
    print(">> Depth = " + str(depth) + "...")
    results[str(depth)] = {}
    for n_estimator in n_estimators:
        print(">>> Number of Estimators: " + str(n_estimator) + "...")
        if (do_regular_validation):
            results[str(depth)][str(n_estimator)] = regular_validation(
                train_random_forest,
                training_X,
                training_Y,
                testing_X,
                testing_Y,
                max_depth = depth,
                n_estimators = n_estimator
            )
        else:
            results[str(depth)][str(n_estimator)] = leave_one_out_validation(
                train_random_forest,
                training_X,
                training_Y,
                max_depth = depth,
                n_estimators = n_estimator
            )

#%% Outputting

json.dump(results, out_file)