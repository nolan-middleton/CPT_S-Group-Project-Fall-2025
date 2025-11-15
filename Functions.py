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
import os as os
import numpy.char as ch
import json as json

#%% Function Definitions

def reformat_data(raw_GDS_filename, output_folder):
    '''
    Reads in a SOFT file and spits out a data folder of easily-readable
    data in .txt, .tsv, and .json format.

    Parameters
    ----------
    raw_GDS_filename : str
        The name of the SOFT file to read in.
    output_folder : str
        The name of the folder to dump the output into.

    Returns
    -------
    A dictionary containing information on the data subsets.
    '''
    print("~~~ Reformatting Data ~~~")
    
    # Set up output folder
    print("- Setting up output directory...")
    if (not os.path.isdir(output_folder)):
        os.mkdir(output_folder)
    
    # Read in data
    print("- Reading data...")
    lines = []
    with open(raw_GDS_filename) as file:
        for line in file:
            lines.append(line)

    # Sections in SOFT files are delimited with "^"
    print("- Sectioning...")
    sections = np.arange(len(lines))[["^" in L for L in lines]]
    sections = np.append(sections, len(lines))
    sections = [
        lines[sections[i]:sections[i+1]]
        for i in range(len(sections)-1)
    ]

    # The data table begins with "!dataset_table_begin" and ends with
    # "!dataset_table_end"
    print("- Finding data table...")
    for i in range(len(lines)):
        if ("!dataset_table_begin" in lines[i]):
            start = i + 1
        elif ("!dataset_table_end" in lines[i]):
            end = i
            break
    cols = np.array(lines[start].replace("\n","").split("\t"))

    # Read in each of the data subsets and their corresponding data columns
    print("- Saving data subsets...")
    subsets = []
    for section in sections:
        name = section[0].split("=")[0].replace(" ","").replace("^","")
        
        if (name == "SUBSET"):
            subsets.append({})
            for L in section:
                if ("!subset_description" in L):
                    subsets[-1]["label"] = L.split("=")[1].replace(
                        " ",
                        ""
                    ).replace("\n", "")
                elif ("!subset_sample_id" in L):
                    subsets[-1]["entries"] = L.split("=")[1].replace(
                        " ",
                        ""
                    ).replace("\n", "").split(",")
            datacols = np.isin(cols, subsets[-1]["entries"])
            subsets[-1]["data"] = np.loadtxt(
                raw_GDS_filename,
                comments = "!",
                delimiter = "\t",
                skiprows = start + 1,
                usecols = np.arange(len(cols))[datacols]
            )
    
    # Remove subsets, only keep the higher-level classes
    remove = []
    for i in range(len(subsets)):
        for j in range(len(subsets)):
            if (i != j) and (j not in remove):
                if (set(subsets[i]["entries"]) <= set(subsets[j]["entries"])):
                    remove.append(i)
                    break
    for i in range(len(remove)-1, -1, -1):
        subsets.pop(remove[i])

    # Save this
    for subset in subsets:
        fname = subset["label"] + ".tsv"
        np.savetxt(output_folder+"/"+fname, subset["data"], delimiter = "\t")
        subset["data"] = fname

    # Read in the gene data (but not all the gene data)
    print("- Saving gene data...")
    geneData = np.loadtxt(
        raw_GDS_filename,
        dtype = str,
        comments = "!",
        delimiter = "\t",
        skiprows = start,
        max_rows = end - start,
        usecols = np.arange(len(cols))[
            np.isin(
                cols,
                [
                    "ID_REF",
                    "IDENTIFIER",
                    "Gene title",
                    "Gene symbol",
                    "Gene ID",
                    "GenBank Accession"
                ]
            )
        ]
    )

    # Save this
    np.savetxt(
        output_folder + "/gene_data.tsv",
        geneData[1:,:],
        fmt = "%s",
        delimiter = "\t",
        header = "\t".join(geneData[0,:])
    )
    
    # GO Stuff
    print("- Saving GO data...")
    if (not os.path.isdir(output_folder + "/GO")):
        os.mkdir(output_folder + "/GO")
    GO_cols = ch.startswith(cols, "GO")
    
    files = [
        open(output_folder+"/GO/"+cols[GO_cols][i].split(":")[1]+".txt", "w")
        for i in range(np.sum(GO_cols))
    ]
    indices = np.arange(len(cols))[GO_cols]
    for line in lines[start + 1:end]:
        entries = line.replace("\n", "").split("\t")
        
        for i in range(len(files)):
            files[i].write(entries[indices[i]].replace("///", "\t") + "\n")
    
    for file in files:
        file.close()
    
    print("~~~ Reformatted Data ~~~")
    return subsets

#%%% Evaluation

def confusion_matrix(Y_hat, Y):
    '''
    Generates a kxk (where k is the number of classes) array representing the
    confusion matrix.
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
    np.ndarray of shape (k,k) of int, where cell (i,j) is the number of
    guesses i when the true answer was j.
    '''
    k = np.max(Y) + 1
    M = np.zeros((k,k), dtype = int)
    for i in range(k):
        for j in range(k):
            M[i,j] = np.sum((Y_hat == i) & (Y == j))
    return M

#%% Model Functions

def setup_model(directory):
    '''
    Searches the directory for the training datasets and any testing datasets.

    Parameters
    ----------
    directory : str
        The directory to search in. Must contain "training_X.tsv" and
        "training_Y.tsv". Any other files must be in pairs named "<name>_X.tsv"
        and "<name>_Y.tsv". The "<>_X.tsv" files must contain the data points
        where each ROW is a separate data point. The "<>_Y.tsv" files must
        contain the labels.

    Returns
    -------
    training_X : np.ndarray of float64
        The training data points.
    training_Y : np.ndarray of int64
        The training labels.
    do_regular_validation : bool
        A boolean indicating whether or not to do regular validation with the
        test sets.
    testing_X : dict
        The test datasets.
    testing_Y : dict
        The test labels.
    '''
    # Mandatory arguments
    training_X = np.loadtxt(directory + "/training_X.tsv", delimiter = "\t")
    training_Y = np.loadtxt(
        directory + "/training_Y.tsv",
        dtype = int,
        delimiter = "\t"
    )

    # Optional arguments
    do_regular_validation = False

    files = np.unique(
        [f[:-6] for f in os.listdir(directory) if ".tsv" in f]
    )
    testing_X = {}
    testing_Y = {}
    for file in files:
        if (file != "training"):
            testing_X[file] = np.loadtxt(
                directory + "/" + file + "_X.tsv",
                delimiter = "\t"
            )
            testing_Y[file] = np.loadtxt(
                directory + "/" + file + "_Y.tsv",
                delimiter = "\t",
                dtype = int
            )
    
    return (
        training_X,
        training_Y,
        do_regular_validation,
        testing_X,
        testing_Y
    )

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
    print("~~~ Leave-One-Out Validation ~~~")
    n = np.shape(X)[0]
    I = np.arange(n)
    k = np.max(Y)
    M = np.zeros((k,k), dtype = int)
    predictions = np.zeros((n,n), dtype = int)
    for i in range(n):
        print("- " + str(i + 1) + "/" + str(n))
        keep = I != i
        this_X = X[keep,:]
        this_Y = Y[keep]
        
        model = train_f(this_X, this_Y, *args, **kwargs)
        Y_hat = model.predict(X[i:(i+1),:])
        M += confusion_matrix(Y_hat, Y[i:(i+1)])
        
        predictions[:,i] = Y_hat
    
    print("~~~ Leave-One-Out Validated ~~~")
    return {"prediction": predictions.tolist(), "results": M.tolist()}

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
    test_X : dict of np.ndarray of float
        The testing datasets. Each entry must be a dataset wherein each ROW
        must be a data point.
    test_Y : dict of np.ndarray of int
        The testing labels.
    *args :
        passed to train_f.
    **kwargs :
        passed to train_f.

    Returns
    -------
    dict of evaluation statistics.
    '''
    print("~~~ Regular Validation ~~~")
    model = train_f(train_X, train_Y, *args, **kwargs)
    print("- Training...")
    train_predictions = model.predict(train_X)
    returnDict = {
        "training": {
            "prediction": train_predictions.tolist(),
            "results": confusion_matrix(train_predictions, train_Y).tolist()
        },
        "testing": {}
    }
    print("- Testing...")
    for key in test_X:
        pred = model.predict(test_X[key])
        returnDict["testing"][key] = {
            "prediction": pred.tolist(),
            "results": confusion_matrix(pred, test_Y[key]).tolist()
        }
    print("~~~ Regular Validated ~~~")
    return returnDict

def output_model_results(results, name):
    '''
    Outputs the results of the model evaluation.

    Parameters
    ----------
    results : dict
        The dictionary of results.
    name : str
        The name of the file to save it as.

    Returns
    -------
    None.
    '''
    directory = "/".join(results.split("/")[:-1])
    if (not os.path.isdir(directory)):
        os.mkdir(directory)
    with open(directory + "/" + name, "w") as file:
        json.dump(results, file)

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