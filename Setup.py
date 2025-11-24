#%% Setup

# Imports
from Functions import reformat_data
import os as os
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, NMF

# Input SOFT files
soft_files = [
    "Raw/GDS1615_full.soft",
    "Raw/GDS2373_full.soft",
    "Raw/GDS2771_full.soft",
    "Raw/GDS3795_full.soft",
    "Raw/GDS4206_full.soft",
    "Raw/GDS4228_full.soft",
    "Raw/GDS4267_full.soft",
    "Raw/GDS5205_full.soft",
    "Raw/GDS963_full.soft",
]

# Output directories
output_dirs = [
    "UlcerativeColitisAndCrohns",
    "SquamousCellLungCarcinomas",
    "SmokerEpithelialCells",
    "MDS",
    "PediatricALL",
    "HIV",
    "JuvenileIdiopathicArthritis",
    "Glioblastoma",
    "MacularDegeneration"
]

# Variables
num_strategies = 5 # Nothing, a priori grouping, PCA, kernelized PCA, NMF
components = np.arange(2,10)

#%% Reformat raw data

print("> Reformatting...")
genes = np.zeros(0, dtype = "<U32")
for i in range(len(soft_files)):
    print(">> " + soft_files[i] + "...")
    if (not os.path.isdir(output_dirs[i])):
        os.mkdir(output_dirs[i])
    subsets = reformat_data(soft_files[i], output_dirs[i]+"/Data")
    with open(output_dirs[i] + "/metadata.txt", "w") as file:
        file.write("Source file: " + soft_files[i] + "\n")
        file.write("Classes: ")
        file.write(", ".join([subset["data"] for subset in subsets]) + "\n")

with open("datasets.txt", "w") as file:
    for D in output_dirs:
        file.write(D + "\n")

#%% Implementing Dimensionality Reduction Strategies

print("> Dimensionality reduction strategies...")
for dataset in output_dirs:
    print(">> " + dataset + "...")
    
    #%%% Setup
    
    print(">>> Setup...")
    with open(dataset + "/metadata.txt") as file:
        for line in file:
            if ("Classes: " in line):
                classes = line.replace("\n", "").split(": ")[1].split(", ")
                break
    labels = [c.split(".tsv")[0] for c in classes]
    
    for i in range(num_strategies):
        if (not os.path.isdir(dataset + "/" + str(i + 1))):
            os.mkdir(dataset + "/" + str(i + 1))
    
    #%%% Strategy 1a: No Dimensionality Reduction nor Data Augmentation
    
    print(">>> Strategy 1: Nothing...")
    data = [np.loadtxt(dataset+"/Data/"+c, delimiter="\t") for c in classes]
    
    plain_X = np.column_stack(tuple(data)).transpose()
    plain_Y = np.concatenate(
        tuple([np.repeat(i, np.shape(data[i])[1]) for i in range(len(data))])
    )
    
    np.savetxt(dataset + "/1/training_X.tsv", plain_X, delimiter = "\t")
    np.savetxt(dataset+"/1/training_Y.tsv",plain_Y,delimiter="\t",fmt="%d")
    
    #%%% Strategy 2a: a priori Grouping
    
    print(">>> Strategy 2: a priori Grouping...")
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
        print("=> " + cat + "...")
        if (not os.path.isdir(dataset + "/2/" + cat)):
            os.mkdir(dataset + "/2/" + cat)
        
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
            dataset + "/2/" + cat + "/IDs.txt",
            all_IDs,
            delimiter = "\t",
            fmt = "%s"
        )
        
        print("==> Grouping...")
        grouped_X = np.zeros((np.shape(plain_X)[0], len(all_IDs)))
        for i in range(len(all_IDs)):
            if (i % 1000 == 0):
                print("==> " + str(i + 1) + "/" + str(len(all_IDs)) + "...")
            included_genes = [all_IDs[i] in ID_list for ID_list in IDs]
            grouped_X[:,i] = np.mean(plain_X[:,included_genes], axis = 1)
        np.savetxt(
            dataset + "/2/" + cat + "/training_X.tsv",
            grouped_X,
            delimiter = "\t"
        )
        np.savetxt(
            dataset + "/2/" + cat + "/training_Y.tsv",
            plain_Y,
            delimiter = "\t",
            fmt = "%d"
        )
    
    #%%% Strategy 3: PCA
    
    print(">>> Strategy 3: PCA...")
    PCA_solver = PCA(max(components))
    PCA_solver.fit(plain_X)
    PCA_X = PCA_solver.transform(plain_X)
    for dim in components:
        if (not os.path.isdir(dataset + "/3/" + str(dim))):
            os.mkdir(dataset + "/3/" + str(dim))
        
        np.savetxt(
            dataset + "/3/" + str(dim) + "/training_X.tsv",
            PCA_X[:,:dim],
            delimiter = "\t"
        )
        np.savetxt(
            dataset + "/3/" + str(dim) + "/training_Y.tsv",
            plain_Y,
            delimiter = "\t",
            fmt = "%d"
        )
    
    #%%% Strategy 4: Kernelized PCA
    
    print(">>> Strategy 4: Kernelized PCA...")
    kernel_PCA_solver = KernelPCA(None)
    kernel_PCA_solver.fit(plain_X)
    kernel_PCA_X = kernel_PCA_solver.transform(plain_X)
    for dim in components:
        if (not os.path.isdir(dataset + "/4/" + str(dim))):
            os.mkdir(dataset + "/4/" + str(dim))
        
        np.savetxt(
            dataset + "/4/" + str(dim) + "/training_X.tsv",
            kernel_PCA_X[:,:dim],
            delimiter = "\t"
        )
        np.savetxt(
            dataset + "/4/" + str(dim) + "/training_Y.tsv",
            plain_Y,
            delimiter = "\t",
            fmt = "%d"
        )
    
    #%%% Strategy 5: NMF
    
    print(">>> Strategy 5: NMF...")
    nonnegative_X = 1 / (1 + np.exp(-plain_X)) # Must have nonnegative values
    for dim in components:
        print("=> " + str(dim) + "...")
        NMF_solver = NMF(dim)
        NMF_solver.fit(nonnegative_X)
        NMF_X = NMF_solver.transform(nonnegative_X)
        
        if (not os.path.isdir(dataset + "/5/" + str(dim))):
            os.mkdir(dataset + "/5/" + str(dim))
        
        np.savetxt(
            dataset + "/5/" + str(dim) + "/training_X.tsv",
            kernel_PCA_X[:,:dim],
            delimiter = "\t"
        )
        np.savetxt(
            dataset + "/5/" + str(dim) + "/training_Y.tsv",
            plain_Y,
            delimiter = "\t",
            fmt = "%d"
        )