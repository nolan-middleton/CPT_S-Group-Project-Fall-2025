#%% Setup

# Imports
import Functions as F
import numpy as np
import sys as sys

# Setup Model
training_X,training_Y,do_regular_validation,testing_X,testing_Y=F.setup_model(
    sys.argv[1]
)

#%% Performing the Evaluation

print("> Naive Bayes on X: " + str(np.shape(training_X)) + "...")
if (do_regular_validation):
    results = F.regular_validation(
        F.train_naive_bayes,
        training_X,
        training_Y,
        testing_X,
        testing_Y,
        prior = None
    )
else:
    results = F.leave_one_out_validation(
        F.train_naive_bayes,
        training_X,
        training_Y,
        prior = None
    )

#%% Outputting

F.output_model_results(results, sys.argv[1] + "/Results/NaiveBayes.json")