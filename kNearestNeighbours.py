#%% Setup

# Imports
from Functions import setup_model, train_k_nearest_neighbours, \
    leave_one_out_validation, regular_validation
import numpy as np
import json as json
import sys as sys
import os as os

# Setup Model
training_X,training_Y,do_regular_validation,testing_X,testing_Y=setup_model(
    sys.argv[1]
)

# Parameters
ks = [3,4,5,6]
ps = [1,2,3,4]

#%% Performing the Evaluation

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

if (not os.path.isdir(sys.argv[1] + "/Results")):
    os.mkdir(sys.argv[1] + "/Results")
json.dump(results, sys.argv[1] + "/Results/kNearestNeighbours.json")