#%% Setup

# Imports
import Functions as F

# Parameters
kernels = ["rbf", "poly", "poly"]
degrees = [0, 1, 2]
Cs = [1E-3 * 10**i for i in range(7)]

# Defs
def model_function(training_X, training_Y):
    results = {}
    for i in range(len(kernels)):
        if (kernels[i] == "poly"):
            if (degrees[i] == 1):
                kernelKey = "linear"
            elif (degrees[i] == 2):
                kernelKey = "quadratic"
            elif (degrees[i] == 3):
                kernelKey = "cubic"
        else:
            kernelKey = kernels[i]
        print("=> Kernel: " + str(kernelKey) + "...")
        results[kernelKey] = {}
        for C in Cs:
            print("==> C: " + str(C) + "...")
            results[kernelKey][str(C)] = F.leave_one_out_validation(
                F.train_support_vector_machine,
                training_X,
                training_Y,
                kernel = kernels[i],
                degree = degrees[i],
                C = C
            )
    return results

#%% Performing the Evaluation

F.execute_model(model_function, "SupportVectorMachine.json")