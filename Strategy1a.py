#%% Setup

# Imports
import numpy as np
import os as os

#%% Compiling Data
print("> Compiling data...")
datasets = np.loadtxt("datasets.txt", dtype = str).tolist()

for dataset in datasets:
    print(">> " + dataset + "...")
    
    print(">>> Getting classes...")
    with open(dataset + "/metadata.txt") as file:
        for line in file:
            if ("Classes: " in line):
                classes = line.replace("\n", "").split(": ")[1].split(", ")
                break
    labels = [c.split(".tsv")[0] for c in classes]
    
    print(">>> Loading and combining data...")
    data = [np.loadtxt(dataset+"/Data/"+c, delimiter="\t") for c in classes]
    
    training_X = np.column_stack(tuple(data)).transpose()
    training_Y = np.concatenate(
        tuple([np.repeat(i, np.shape(data[i])[1]) for i in range(len(data))])
    )
    
    if (not os.path.isdir(dataset + "/1a")):
        os.mkdir(dataset + "/1a")
    
    print(">>> Saving...")
    np.savetxt(dataset + "/1a/training_X.tsv", training_X, delimiter = "\t")
    np.savetxt(dataset+"/1a/training_Y.tsv",training_Y,delimiter="\t",fmt="%d")