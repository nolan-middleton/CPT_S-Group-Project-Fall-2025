#%% Setup

# Imports
import Functions as F

# Parameters
ks = [3,4,5]
ps = [1,2,3,4]

# Defs
def model_function(training_X, training_Y):
    results = {}
    for k in ks:
        print("=> k: " + str(k) + "...")
        results[str(k)] = {}
        for p in ps:
            print("=> p: " + str(p) + "...")
            results[str(k)][str(p)] = F.leave_one_out_validation(
                F.train_k_nearest_neighbours,
                training_X,
                training_Y,
                k = k,
                p = p
            )
    return results

#%% Performing the Evaluation

F.execute_model(model_function, "kNearestNeighbours.json")