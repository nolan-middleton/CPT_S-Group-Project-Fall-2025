#%% Setup

# Imports
import numpy as np
import os as os
import multiomics_open_research as mor
import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import pickle
import multiomics_open_research.bulk_rna_bert.pretrained as prt
import multiomics_open_research.common.preprocess as pre

# Load data
common_gene_ids = np.loadtxt(
    "../multiomics-open-research/data/bulkrnabert/common_gene_id.txt",
    dtype = str
)

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

    if (not os.path.isdir(dataset + "/4a")):
        os.mkdir(dataset + "/4a")
    
    plain_X = np.loadtxt(dataset+"/1a/training_X.tsv", delimiter = "\t")
    plain_Y=np.loadtxt(dataset+"/1a/training_Y.tsv",delimiter="\t",dtype=int)
    
    gene_data = np.loadtxt(
        dataset + "/Data/gene_data.tsv",
        delimiter = "\t",
        dtype = str
    )
    included = gene_data[:,1] != "--Control"
    
    gene_translations = np.loadtxt(
        dataset + "/Data/geneIDConversion.csv",
        delimiter = ",",
        dtype = str,
        usecols = [0,1],
        quotechar = "\"",
        skiprows = 1
    )
    
    ENSGs = np.zeros(np.shape(gene_data)[0], gene_translations.dtype)
    for i in range(np.shape(gene_data)[0]):
        if (included[i]):
            ENSGs[i]=gene_translations[
                gene_translations[:,0] == gene_data[i,0].upper(),
                1
            ][0]
    
    #%%% Reduce the dataset to include only the common_gene_ids
    
    included = np.isin(ENSGs, common_gene_ids)
    order = np.argsort(ENSGs[included])
    header = ENSGs[included][order]
    
    reduced_X = plain_X[:,included][:,order]
    
    # Some of these reads are identical. We'll add them together
    U = np.unique(header)
    converted_X = np.zeros((np.shape(reduced_X)[0], len(U)))
    for i in range(len(U)):
        converted_X[:,i] = np.sum(reduced_X[:,header == U[i]], axis = 1)
    
    np.savetxt(
        dataset + "/4a/reformatted_X.csv",
        converted_X,
        delimiter = ",",
        header = ",".join(U),
        comments = ""
    )
    
    #%%% Reload data into pretrained model
    
    params,forward_fn,tokenizer,config = prt.get_bulkrnabert_pretrained_model(
        "bulk_rna_bert_gtex_encode"
    )
    forward_fn = hk.transform(forward_fn)
    
    rna_seq_df = pd.read_csv(dataset + "/4a/reformatted_X.csv")
    rna_seq_array = pre.preprocess_omic(rna_seq_df, config)
    tokens_ids = tokenizer.batch_tokenize(rna_seq_array)
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    
    #%%% Run it through model
    
    random_key = jax.random.PRNGKey(0)
    outs = forward_fn.apply(params, random_key, tokens_ids)
    
    # Get mean embeddings from layer 4
    gene_expression_mean_embeddings = outs["embeddings_4"].mean(axis=1)