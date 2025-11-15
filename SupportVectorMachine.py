#%% Setup

# Imports
from Functions import setup_model, train_support_vector_machine, \
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
kernels = ["rbf", "poly", "poly", "poly", "poly"]
degrees = [0, 1, 2, 3, 4]
Cs = [1E-5 * 10**i for i in range(11)]

#%% Performing the Evaluation

print("> Support Vector Machine on X: " + str(np.shape(training_X)) + "...")
results = {}

for i in range(len(kernels)):
    if (kernels[i] == "rbf"):
        kernel_key = kernels[i]
    else:
        kernel_key = kernels[i] + str(degrees[i])
    print(">> Kernel = " + kernel_key + "...")
    results[kernel_key] = {}
    for C in Cs:
        print(">> C = " + str(C) + "...")
        if (do_regular_validation):
            results[kernel_key][str(C)] = regular_validation(
                train_support_vector_machine,
                training_X,
                training_Y,
                testing_X,
                testing_Y,
                C = C,
                kernel = kernels[i],
                degree = degrees[i]
            )
        else:
            results[kernel_key][str(C)] = leave_one_out_validation(
                train_support_vector_machine,
                training_X,
                training_Y,
                C = C,
                kernel = kernels[i],
                degree = degrees[i]
            )

#%% Outputting

if (not os.path.isdir(sys.argv[1] + "/Results")):
    os.mkdir(sys.argv[1] + "/Results")
with open(sys.argv[1] + "/Results/SupportVectorMachine.json", "w") as file:
    json.dump(results, file)