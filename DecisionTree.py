#%% Setup

# Imports
import Functions as F

# Parameters

# The max_depths to test, 2^4 is already 16 buckets, and for 200 data points,
# that's already down to ~12-13 per bucket if evenly distributed.
depths = [2,3,4]

# Defs
def model_function(training_X, training_Y):
    results = {}
    for depth in depths:
        print("=> Depth: " + str(depth) + "...")
        results[str(depth)] = F.leave_one_out_validation(
            F.train_decision_tree,
            training_X,
            training_Y,
            max_depth = depth
        )
    return results

#%% Performing the Evaluation

F.execute_model(model_function, "DecisionTree.json")