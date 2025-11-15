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
kernels = ["rbf", "poly", "poly", "poly"]
degrees = [0, 1, 2, 3]
Cs = [1E-3 * 10**i for i in range(7)]

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
            results[kernel_key][str(C)] = F.regular_validation(
                F.train_support_vector_machine,
                training_X,
                training_Y,
                testing_X,
                testing_Y,
                C = C,
                kernel = kernels[i],
                degree = degrees[i]
            )
        else:
            results[kernel_key][str(C)] = F.leave_one_out_validation(
                F.train_support_vector_machine,
                training_X,
                training_Y,
                C = C,
                kernel = kernels[i],
                degree = degrees[i]
            )

#%% Outputting

F.output_model_results(
    results,
    sys.argv[1] + "/Results/SupportVectorMachine.json"
)