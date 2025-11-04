#%% Setup

# This file holds all the functions for the different models we're going to be
# testing.

# Imports
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#%% Function Definitions

#%%% Evaluation and Validation

def confusion_matrix(Y_hat, Y):
    '''
    Generates a 2x2 array representing the confusion matrix.
     |N  P <-- Actual
    -+-----
    N|TN FN
    P|FP TP
    ^
    Guesses
    
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

def accuracy(M):
    '''
    Computes the accuracy from a confusion matrix.
    (TP + TN) / (TP + FP + TN + FN)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the accuracy.
    '''
    return np.trace(M) / np.sum(M)

def sensitivity(M):
    '''
    Computes the sensitivity from a confusion matrix.
    TP / (TP + FN)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the sensitivity.
    '''
    return M[1,1] / np.sum(M[:,1])

TPR = sensitivity
recall = sensitivity

FPR = lambda M : 1 - sensitivity(M)

def specificity(M):
    '''
    Computes the specificity from a confusion matrix.
    TN / (TN + FP)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the specificity.
    '''
    return M[0,0] / np.sum(M[:,0])

TNR = specificity
FNR = lambda M : 1 - specificity(M)

def precision(M):
    '''
    Computes the precision from a confusion matrix.
    TP / (TP + FP)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the precision.
    '''
    return M[1,1] / np.sum(M[1,:])

PPV = precision

FDR = lambda M : 1 - precision(M)

def NPV(M):
    '''
    Computes the negative predictive value from a confusion matrix.
    TN / (TN + FN)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the negative predictive value.
    '''
    return M[0,0] / np.sum(M[0,:])

FOR = lambda M : 1 - NPV(M)

def phi(M):
    '''
    Computes the phi coefficient from a confusion matrix.
    sqrt(TPR*TNR*PPV*NPV) - sqrt(FNR*FPR*FOR*FDR)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the phi coefficient.
    '''
    return np.sqrt(TPR(M)*TNR(M)*PPV(M)*NPV(M)) - \
        np.sqrt(FNR(M)*FPR(M)*FOR(M)*FDR(M))

MCC = phi

def F1(M):
    '''
    Computes the F1 score from a confusion matrix.
    2TP/(2TP+FP+FN)

    Parameters
    ----------
    M : np.ndarray of int
        The confusion matrix. Must have shape (2,2).

    Returns
    -------
    float representing the F1 score.
    '''
    return 2*M[1,1]/(2*M[1,1] + M[1,0] + M[0,1])

def evaluate(M):
    '''
    Generates a dictionary of all the model's performance statistics that can
    be easily JSON-serialized.

    Parameters
    ----------
    M : np.ndarray of int
        The model's confusion matrix.

    Returns
    -------
    dict containing the model evaluation statistics.
    '''
    
    # The Python JSON library doesn't play nicely with the numpy data types,
    # so I've found it's better to just explicitly convert any data you want
    # to save in JSON files to the regular Python data types. -Nolan
    return {
        "accuracy": float(accuracy(M)),
        "sensitivity": float(sensitivity(M)),
        "specificity": float(specificity(M)),
        "precision": float(precision(M)),
        "NPV": float(NPV(M)),
        "F1": float(F1(M)),
        "phi": float(phi(M)),
        "confusion_matrix": {
            "TP": int(M[1,1]),
            "TN": int(M[0,0]),
            "FP": int(M[1,0]),
            "FN": int(M[0,1])
        }
    }

def leave_one_out_validation(train_f, X, Y, *args, **kwargs):
    '''
    Performs leave-one-out validation to measure the performance of a model.

    Parameters
    ----------
    train_f : function
        The function to train the model with. Must take in X and Y as its first
        two arguments, then any other *args and **kwargs. Must return a model
        that has the .predict() method.
    X : np.ndarray of float
        The dataset. Each ROW must be a data point.
    Y : np.ndarray of int
        The labels.
    *args :
        Passed to train_f.
    **kwargs :
        Passed to train_f.

    Returns
    -------
    dict of evaluation statistics.
    '''
    n = np.shape(X)[0]
    I = np.arange(n)
    M = np.zeros((2,2), dtype = int)
    for i in range(n):
        keep = I != i
        this_X = X[keep,:]
        this_Y = Y[keep]
        
        model = train_f(this_X, this_Y, *args, **kwargs)
        M += confusion_matrix(model.predict(X[i:(i+1),:]), Y[i:(i+1)])
    return evaluate(M)

def regular_validation(train_f,train_X,train_Y,test_X,test_Y,*args,**kwargs):
    '''
    Performs regular validation on the dataset with a separate test set.

    Parameters
    ----------
    train_f : function
        The function to train the model with. Must take in X and Y as its first
        two arguments, then any other *args and **kwargs. Must return a model
        that has the .predict() method.
    train_X : np.ndarray of float
        The training dataset. Each ROW must be a data point.
    train_Y : np.ndarray of int
        The training labels.
    test_X : np.ndarray of float
        The testing dataset. Each ROW must be a data point.
    test_Y : np.ndarray of int
        The testing labels.
    *args :
        passed to train_f.
    **kwargs :
        passed to train_f.

    Returns
    -------
    dict of evaluation statistics.
    '''
    model = train_f(train_X, train_Y, *args, **kwargs)
    M = confusion_matrix(model.predict(test_X), test_Y)
    return evaluate(M)

#%%% Training Models

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
    DT = DecisionTreeClassifier(
        criterion = "entropy",
        max_depth = max_depth
    )
    DT.fit(X, Y)
    return DT

def train_random_forest(X, Y, n_estimators = 100, max_depth = 6):
    '''
    Trains a random forest on a given dataset with given labels.

    Parameters
    ----------
    X : np.ndarray of float
        The training data. Each ROW must be a data point.
    Y : np.ndarray of int
        The training labels. The entry at index i must be the label for the
        training data point in X at row i.
    n_estimators : int, optional
        The number of estimators to use in the classifier. The default is 100.
    max_depth : int, optional
        The maximum depth to allow the classifier to reach. The default is 6.

    Returns
    -------
    RandomForestClassifier trained on the supplied data.
    '''
    RF = RandomForestClassifier(
        n_estimators = n_estimators,
        criterion = "entropy",
        max_depth = max_depth
    )
    RF.fit(X, Y)
    return RF

def train_naive_bayes(X, Y, priors = None):
    '''
    Trains a naive Bayes classifier on a given dataset with given labels.

    Parameters
    ----------
    X : np.ndarray of float
        The training data. Each ROW must be a data point.
    Y : np.ndarray of int
        The training labels. The entry at index i must be the label for the
        training data point in X at row i.
    priors : np.ndarray of float, optional
        The prior probabilities of each class, stored in an array. Defaults
        to None.

    Returns
    -------
    GaussianNB trained on the supplied data.
    '''
    NBC = GaussianNB(priors = priors)
    NBC.fit(X, Y)
    return NBC

def train_support_vector_machine(X, Y, C = 1.0, kernel = "rbf", degree = 1):
    '''
    Trains a support vector machine on a given dataset with given labels.

    Parameters
    ----------
    X : np.ndarray of float
        The training data. Each ROW must be a data point.
    Y : np.ndarray of int
        The training labels. The entry at index i must be the label for the
        training data point in X at row i.
    C : float, optional
        The regularization parameter for the SVC. Defaults to 1.0.
    kernel : string, optional
        The kernel to use when training. Defaults to "rbf".
    degree : int, optional
        The degree of the kernel if the kernel is "poly".

    Returns
    -------
    SVC trained on the supplied data.
    '''
    SVM = SVC(C = C, kernel = kernel, degree = degree)
    SVM.fit(X, Y)
    return SVM

def train_k_nearest_neighbours(X, Y, k = 3, p = 2):
    '''
    Trains a support vector machine on a given dataset with given labels.

    Parameters
    ----------
    X : np.ndarray of float
        The training data. Each ROW must be a data point.
    Y : np.ndarray of int
        The training labels. The entry at index i must be the label for the
        training data point in X at row i.
    k : int, optional
        The number of neighbours to consider near. Defaults to 3.
    p : float, optional
        The power of the Minkowski distance. Defaults to 2.

    Returns
    -------
    KNeighborsClassifier trained on the supplied data.
    '''
    kNN = KNeighborsClassifier(n_neighbors = k, p = p)
    kNN.fit(X, Y)
    return kNN