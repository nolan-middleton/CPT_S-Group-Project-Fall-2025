#%% Setup

# Imports
import Functions as F
import numpy as np
import sys as sys

# Setup Model
training_X,training_Y,do_regular_validation,testing_X,testing_Y=F.setup_model(
    sys.argv[1]
)

# Parameters
depths = [1,2,3,4] # The max_depths to test
n_estimators = [100] + [1000 * n for n in range(1,5)] # n_estimators to test

#%% Performing the Evaluation

print("> Random Forest on X: " + str(np.shape(training_X)) + "...")
results = {}

for depth in depths:
    print(">> Depth = " + str(depth) + "...")
    results[str(depth)] = {}
    for n_estimator in n_estimators:
        print(">>> Number of Estimators: " + str(n_estimator) + "...")
        if (do_regular_validation):
            results[str(depth)][str(n_estimator)] = F.regular_validation(
                F.train_random_forest,
                training_X,
                training_Y,
                testing_X,
                testing_Y,
                max_depth = depth,
                n_estimators = n_estimator
            )
        else:
            results[str(depth)][str(n_estimator)]=F.leave_one_out_validation(
                F.train_random_forest,
                training_X,
                training_Y,
                max_depth = depth,
                n_estimators = n_estimator
            )

#%% Outputting

F.output_model_results(results, sys.argv[1] + "/Results/RandomForest.json")