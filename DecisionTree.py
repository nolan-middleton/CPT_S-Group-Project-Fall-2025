#%% Setup

# Imports
from Functions import setup_model, train_decision_tree, \
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
depths = [1,2,3,4,5,6] # The max_depths to test

#%% Performing the Evaluation

print("> Decision Tree on X: " + str(np.shape(training_X)) + "...")
results = {}

for depth in depths:
    print(">> Depth = " + str(depth) + "...")
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

if (not os.path.isdir(sys.argv[1] + "/Results")):
    os.mkdir(sys.argv[1] + "/Results")
with open(sys.argv[1] + "/Results/DecisionTree.json", "w") as file:
    json.dump(results, file)