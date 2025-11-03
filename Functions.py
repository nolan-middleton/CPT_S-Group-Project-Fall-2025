#%% Setup

# This file holds all the functions for the different models we're going to be
# testing.

# Imports
import numpy as np
import sklearn.tree as tree

#%% Function Definitions

#%%% Evaluation and Validation

def confusion_matrix(Y_hat, Y):
    '''
    Generates a 2x2 array representing the confusion matrix.

    Parameters
    ----------
    Y_hat : np.ndarray of int
        The guesses.
    Y : np.ndarray of int
        The correct answers.

    Returns
    -------
    np.ndarray of shape (2,2) of int, where cell (i,j) is the number of
    guesses i when the true answer was j.
    '''
    M = np.zeros((2,2), dtype = int)
    M[0,0] = np.sum((Y_hat == 0) & (Y == 0))
    M[0,1] = np.sum((Y_hat == 0) & (Y == 1))
    M[1,0] = np.sum((Y_hat == 1) & (Y == 0))
    M[1,1] = np.sum((Y_hat == 1) & (Y == 1))
    return M

def leave_one_out_validation(train_f, test_f, X, Y, *args, **kwargs):
    '''
    Performs leave-one-out validation to measure the performance of a model.

    Parameters
    ----------
    train_f : function
        The function to train the model with. Must take in X and Y as its first
        two arguments, then any other *args and **kwargs. Must return a model
        that is passed to test_f.
    test_f : function
        The function to evaluate the model with. Must take in a model (the
        output of train_f) as its first argument, then X, then Y. Must return
        a 2x2 confusion matrix.
    X : np.ndarray of float
        The dataset. Each ROW must be a data point.
    Y : np.ndarray of int
        The labels.
    *args :
        Passed to train_f.
    **kwargs :
        Passed to test_f.

    Returns
    -------
    np.ndarray representing an aggregate confusion matrix.
    '''
    n = np.shape(X)[0]
    I = np.arange(n)
    M = np.zeros((2,2), dtype = int)
    for i in range(n):
        keep = I != i
        this_X = X[keep,:]
        this_Y = Y[keep]
        
        model = train_f(this_X, this_Y, *args, **kwargs)
        M += test_f(model, X[i:(i+1),:], Y[i:(i+1)])
    return M

#%%% Models

def train_decision_tree(X, Y, max_depth = 6):
    '''
    Trains a decision tree on a given dataset with given labels.

    Parameters
    ----------
    X : np.ndarray of float
        The training data. Each ROW must be a data point.
    Y : np.ndarray of int
        The training labels. The entry at index i must be the label for the
        training data point in X at row i.
    max_depth : int, optional
        The maximum depth to allow the classifier to reach. The default is 6.

    Returns
    -------
    DecisionTreeClassifier trained on the supplied data.
    '''
    DT = tree.DecisionTreeClassifier(
        criterion = "entropy",
        max_depth = max_depth
    )
    DT.fit(X, Y)
    
    return DT

def test_decision_tree(DT, X, Y):
    '''
    Evaluates a pretrained decision tree on testing data.

    Parameters
    ----------
    DT : DecisionTreeClassifier
        The trained decision tree.
    X : np.ndarray of float
        The testing data to evaluate the tree on.
    Y : np.ndarray of int
        The labels for the testing data.

    Returns
    -------
    np.ndarray representing the confusion matrix.
    '''
    return confusion_matrix(DT.predict(X), Y)