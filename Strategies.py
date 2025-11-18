#%% Setup

# Imports
import numpy as np
import os as os
from sklearn.decomposition import PCA
import Functions as F
import tensorflow.keras as keras
import gc as gc

# Variables
PCA_components = [2,3,4]
VAE_layers = [5000, 2500, 1250, 725, 512]

#%% Main Loop
datasets = np.loadtxt("datasets.txt", dtype = str).tolist()

for dataset in datasets:
    print("> " + dataset + "...")
    
    #%%% Setup
    
    print(">> Setup...")
    with open(dataset + "/metadata.txt") as file:
        for line in file:
            if ("Classes: " in line):
                classes = line.replace("\n", "").split(": ")[1].split(", ")
                break
    labels = [c.split(".tsv")[0] for c in classes]
    
    for i in range(5):
        if (not os.path.isdir(dataset + "/" + str(i + 1) + "a")):
            os.mkdir(dataset + "/" + str(i + 1) + "a")
        if (not os.path.isdir(dataset + "/" + str(i + 1) + "b")):
            os.mkdir(dataset + "/" + str(i + 1) + "b")
    
    #%%% Strategy 1a: No dimensionality reduction nor data augmentation
    
    print(">> Strategy 1a: Nothing...")
    data = [np.loadtxt(dataset+"/Data/"+c, delimiter="\t") for c in classes]
    
    plain_X = np.column_stack(tuple(data)).transpose()
    plain_Y = np.concatenate(
        tuple([np.repeat(i, np.shape(data[i])[1]) for i in range(len(data))])
    )
    
    np.savetxt(dataset + "/1a/training_X.tsv", plain_X, delimiter = "\t")
    np.savetxt(dataset+"/1a/training_Y.tsv",plain_Y,delimiter="\t",fmt="%d")
    
    #%%% Strategy 2a: a priori Grouping
    
    print(">> Strategy 2a: a priori Grouping...")
    GO_categories = os.listdir(dataset + "/Data/GO")
    GO_categories = list(
        set(
            [
                file.replace(" ID","").replace(".txt","")
                for file in GO_categories
            ]
        )
    )
    
    for cat in GO_categories:
        print(">>> " + cat + "...")
        if (not os.path.isdir(dataset + "/2a/" + cat)):
            os.mkdir(dataset + "/2a/" + cat)
        
        IDs = []
        with open(dataset + "/Data/GO/" + cat + " ID.txt") as file:
            for line in file:
                IDs.append(np.array(line[:-1].split("\t")))
        IDs = [ID[np.char.startswith(ID, "GO:")] for ID in IDs]
        IDs = [ID if len(ID) > 0 else np.array(["-1"]) for ID in IDs]
        IDs = [np.char.replace(ID, "GO:", "").astype(int) for ID in IDs]
        all_IDs = np.unique(np.concatenate(tuple(IDs)))
        all_IDs = all_IDs[all_IDs >= 0]
        np.savetxt(
            dataset + "/2a/" + cat + "/IDs.txt",
            all_IDs,
            delimiter = "\t",
            fmt = "%s"
        )
        
        print("=> Grouping...")
        grouped_X = np.zeros((np.shape(plain_X)[0], len(all_IDs)))
        for i in range(len(all_IDs)):
            if (i % 100 == 0):
                print("==> " + str(i + 1) + "/" + str(len(all_IDs)) + "...")
            included_genes = [all_IDs[i] in ID_list for ID_list in IDs]
            grouped_X[:,i] = np.mean(plain_X[:,included_genes], axis = 1)
        np.savetxt(
            dataset + "/2a/" + cat + "/training_X.tsv",
            grouped_X,
            delimiter = "\t"
        )
        np.savetxt(
            dataset + "/2a/" + cat + "/training_Y.tsv",
            plain_Y,
            delimiter = "\t",
            fmt = "%d"
        )
    
    #%%% Strategy 3a: PCA
    
    print(">> Strategy 3a: PCA...")
    PCA_solver = PCA(max(PCA_components))
    PCA_solver.fit(plain_X)
    PCA_X = PCA_solver.transform(plain_X)
    for dims in PCA_components:
        if (not os.path.isdir(dataset + "/3a/" + str(dims))):
            os.mkdir(dataset + "/3a/" + str(dims))
        
        np.savetxt(
            dataset + "/3a/" + str(dims) + "/training_X.tsv",
            PCA_X[:,:dims],
            delimiter = "\t"
        )
        np.savetxt(
            dataset + "/3a/" + str(dims) + "/training_Y.tsv",
            plain_Y,
            delimiter = "\t",
            fmt = "%d"
        )
    
    #%%% Strategy 4a: VAE
    
    print(">>> Strategy 4a: VAE...")
    normalized_X = np.tanh(np.log10(plain_X))
    encoder, decoder = F.VAE_encoder_decoder(np.shape(plain_X)[1], VAE_layers)
    vae = F.VAE(encoder, decoder)
    vae.compile(optimizer = keras.optimizers.Adam())
    vae.fit(normalized_X, epochs = 20, batch_size = np.shape(plain_X)[0])
    gc.collect()