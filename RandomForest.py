#%% Setup

# Imports
from Functions import setup_model, train_random_forest, \
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
n_estimators = [100] + [500 * n for n in range(1,10)] # n_estimators to test

#%% Performing the Evaluation

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

if (not os.path.isdir(sys.argv[1] + "/Results")):
    os.mkdir(sys.argv[1] + "/Results")
with open(sys.argv[1] + "/Results/RandomForest.json", "w") as file:
    json.dump(results, file)