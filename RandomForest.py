#%% Setup

# Imports
import Functions as F

# Parameters
depths = [2,3,4] # The max_depths to test
n_estimators = [100, 200, 500, 1000, 2000] # n_estimators to test

# Defs
def model_function(training_X, training_Y):
    results = {}
    for depth in depths:
        print("=> Depth: " + str(depth) + "...")
        results[str(depth)] = {}
        for n_estimator in n_estimators:
            print("==> Number of Estimators: " + str(n_estimator) + "...")
            results[str(depth)][str(n_estimator)]=F.leave_one_out_validation(
                F.train_random_forest,
                training_X,
                training_Y,
                max_depth = depth,
                n_estimators = n_estimator
            )
    return results

#%% Performing the Evaluation

F.execute_model(model_function, "RandomForest.json")
