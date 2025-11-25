#%% Setup

# Imports
import Functions as F

# Defs
def model_function(training_X, training_Y):
    results = F.leave_one_out_validation(
        F.train_naive_bayes,
        training_X,
        training_Y
    )
    return results

#%% Performing the Evaluation

F.execute_model(model_function, "NaiveBayes.json")