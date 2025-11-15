#%% Setup

from Functions import get_command_line_args, train_k_nearest_neighbours, \
    leave_one_out_validation, regular_validation
import numpy as np
import json as json

ks = [3,4,5,6]
ps = [1,2,3,4]

training_X, training_Y, out_file, do_regular_validation, \
    testing_X, testing_Y = get_command_line_args()

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