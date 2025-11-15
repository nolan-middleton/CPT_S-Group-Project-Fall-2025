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
ks = [3,4,5]
ps = [1,2,3,4]

#%% Performing the Evaluation

print("> k-Nearest Neighbours on X: " + str(np.shape(training_X)) + "...")
results = {}

for k in range(len(ks)):
    print(">> k = " + str(k) + "...")
    results[str(k)] = {}
    for p in ps:
        print(">> p = " + str(p) + "...")
        if (do_regular_validation):
            results[str(k)][str(p)] = F.regular_validation(
                F.train_k_nearest_neighbours,
                training_X,
                training_Y,
                testing_X,
                testing_Y,
                k = k,
                p = p
            )
        else:
            results[str(k)][str(p)] = F.leave_one_out_validation(
                F.train_k_nearest_neighbours,
                training_X,
                training_Y,
                k = k,
                p = p
            )

#%% Outputting

F.output_model_results(results,sys.argv[1]+"/Results/kNearestNeighbours.json")