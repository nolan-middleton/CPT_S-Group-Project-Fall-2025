#%% Setup

from Functions import train_decision_tree, leave_one_out_validation, \
    regular_validation
import numpy as np
import sys as sys
import json as json

depths = [1,2,3,4,5,6] # The max_depths to test

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

print("> Decision Tree on X: " + str(np.shape(training_X)) + "...")
results = {}

for depth in depths:
    print(">> Depth = " + str(depth))
    if (do_regular_validation):
        results[str(depth)] = regular_validation(
            train_decision_tree,
            training_X,
            training_Y,
            testing_X,
            testing_Y,
            max_depth = depth
        )
    else:
        results[str(depth)] = leave_one_out_validation(
            train_decision_tree,
            training_X,
            training_Y,
            max_depth = depth
        )

#%% Outputting

json.dump(results, out_file)