#%% Setup

from Functions import train_support_vector_machine, leave_one_out_validation, \
    regular_validation
import numpy as np
import sys as sys
import json as json

kernels = ["rbf", "poly", "poly", "poly", "poly"]
degrees = [0, 1, 2, 3, 4]
Cs = [1E-5 * 10**i for i in range(11)]

#%%% Command-Line Arguments

# Command-line arguments are, IN ORDER:
# - training_X: The name of the file with the training dataset. MANDATORY.
# - training_Y: The name of the file with the training labels. MANDATORY.
# - out_file: The name of the JSON file to dump the output. MANDATORY.
# - testing_X: The name of the file with the testing dataset. OPTIONAL.
# -- If omitted, will perform leave-one-out validation instead.
# - testing_Y: The name of the file with the testing labels. OPTIONAL.
# -- MUST be specified if testing_pts is provided.

# Mandatory arguments
training_X = np.loadtxt(sys.argv[1], delimiter = "\t")
training_Y = np.loadtxt(sys.argv[2], dtype = int, delimiter = "\t")

out_file = sys.argv[3]

# Optional arguments

do_regular_validation = False

if (len(sys.argv) > 3):
    do_regular_validation = True
    testing_X = np.loadtxt(sys.argv[4], delimiter = "\t")
    testing_Y = np.loadtxt(sys.argv[5], delimiter = "\t")

#%% Performing the evaluation

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

json.dump(results, out_file)