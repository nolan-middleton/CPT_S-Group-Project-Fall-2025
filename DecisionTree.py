#%% Setup

# Imports
import functions as F
import numpy as np
import sys as sys

# Setup Model
training_X,training_Y,do_regular_validation,testing_X,testing_Y=F.setup_model(
    sys.argv[1]
)

# Parameters
depths = [1,2,3,4] # The max_depths to test, 2^4 is already 16 buckets

#%% Performing the Evaluation

print("> Decision Tree on X: " + str(np.shape(training_X)) + "...")
results = {}

for depth in depths:
    print(">> Depth = " + str(depth) + "...")
    if (do_regular_validation):
        results[str(depth)] = F.regular_validation(
            F.train_decision_tree,
            training_X,
            training_Y,
            testing_X,
            testing_Y,
            max_depth = depth
        )
    else:
        results[str(depth)] = F.leave_one_out_validation(
            F.train_decision_tree,
            training_X,
            training_Y,
            max_depth = depth
        )

#%% Outputting

F.output_model_results(results, sys.argv[1] + "/Results/DecisionTree.json")