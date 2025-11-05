#%% Setup

from Functions import train_k_nearest_neighbours, leave_one_out_validation, \
    regular_validation
import numpy as np
import sys as sys
import json as json

ks = [3,4,5,6]
ps = [1,2,3,4]

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

print("> k-Nearest Neighbours on X: " + str(np.shape(training_X)) + "...")
results = {}

for k in range(len(ks)):
    print(">> k = " + str(k) + "...")
    results[str(k)] = {}
    for p in ps:
        print(">> p = " + str(p) + "...")
        if (do_regular_validation):
            results[str(k)][str(p)] = regular_validation(
                train_k_nearest_neighbours,
                training_X,
                training_Y,
                testing_X,
                testing_Y,
                k = k,
                p = p
            )
        else:
            results[str(k)][str(p)] = leave_one_out_validation(
                train_k_nearest_neighbours,
                training_X,
                training_Y,
                k = k,
                p = p
            )

#%% Outputting

json.dump(results, out_file)