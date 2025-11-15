#%% Setup

from Functions import get_command_line_args, train_naive_bayes, \
    leave_one_out_validation, regular_validation
import numpy as np
import json as json

priors = [None, [1-0.172, 0.172]] # According to PMID: 36326752

training_X, training_Y, out_file, do_regular_validation, \
    testing_X, testing_Y = get_command_line_args()

#%% Performing the evaluation

print("> Naive Bayes on X: " + str(np.shape(training_X)) + "...")
results = {}

for i in range(len(priors)):
    prior = priors[i]
    
    if (prior == None):
        prior_key = "noninformative"
    else:
        prior_key = str(prior[i][1])
    
    print(">> Prior = " + prior_key + "...")
    if (do_regular_validation):
        results[prior_key] = regular_validation(
            train_naive_bayes,
            training_X,
            training_Y,
            testing_X,
            testing_Y,
            prior = prior
        )
    else:
        results[prior_key] = leave_one_out_validation(
            train_naive_bayes,
            training_X,
            training_Y,
            prior = prior
        )

#%% Outputting

json.dump(results, out_file)