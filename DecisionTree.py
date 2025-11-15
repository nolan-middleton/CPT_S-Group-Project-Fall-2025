#%% Setup

from Functions import get_command_line_args, train_decision_tree, \
    leave_one_out_validation, regular_validation
import numpy as np
import json as json

depths = [1,2,3,4,5,6] # The max_depths to test

training_X, training_Y, out_file, do_regular_validation, \
    testing_X, testing_Y = get_command_line_args()

#%% Performing the evaluation

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

json.dump(results, out_file)