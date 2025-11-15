#%% Setup

from Functions import get_command_line_args, train_random_forest, \
    leave_one_out_validation, regular_validation
import numpy as np
import json as json

depths = [1,2,3,4,5,6] # The max_depths to test
n_estimators = [100] + [500 * n for n in range(1,10)] # n_estimators to test

training_X, training_Y, out_file, do_regular_validation, \
    testing_X, testing_Y = get_command_line_args()

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