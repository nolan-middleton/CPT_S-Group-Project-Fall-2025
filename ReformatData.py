#%% Setup

# Imports
import numpy as np
import os as os

# Read in data
lines = []
with open("GDS963_full.soft") as file:
    for line in file:
        lines.append(line)

lines = np.array(lines)

# Sections in SOFT files are delimited with "^"
sections = np.arange(len(lines))[["^" in L for L in lines]]
sections = np.append(sections, len(lines))
sections = [
    lines[sections[i]:sections[i+1]]
    for i in range(len(sections)-1)
]

# The data table begins with "!dataset_table_begin" and ends with
# "!dataset_table_end"
start = np.arange(len(lines))[lines == "!dataset_table_begin\n"][0] + 1
end = np.arange(len(lines))[lines == "!dataset_table_end\n"][0]
cols = np.array(lines[start].replace("\n","").split("\t"))

#%% Data values

# Read in each of the data subsets and their corresponding data columns
subsets = []
for section in sections:
    name = section[0].split("=")[0].replace(" ","").replace("^","")
    
    if (name == "SUBSET"):
        subsets.append({})
        subsets[-1]["label"] = section[
            ["!subset_description" in L for L in section]
        ][0].split("=")[1].replace(" ","").replace("\n","")
        subsets[-1]["entries"] = section[
            ["!subset_sample_id" in L for L in section]
        ][0].split("=")[1].replace(" ","").replace("\n","").split(",")
        datacols = np.isin(cols, subsets[-1]["entries"])
        subsets[-1]["data"] = np.loadtxt(
            "GDS963_full.soft",
            comments = "!",
            delimiter = "\t",
            skiprows = start + 1,
            usecols = np.arange(len(cols))[datacols]
        )

# Save this
for subset in subsets:
    fname = subset["label"] + ".tsv"
    np.savetxt(fname, subset["data"], delimiter = "\t")
    subset["data"] = fname

#%% Normal Gene Data

# Read in the gene data (but not all the gene data)
geneData = np.loadtxt(
    "GDS963_full.soft",
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
    "gene_data.tsv",
    geneData[1:,:],
    fmt = "%s",
    delimiter = "\t",
    header = "\t".join(geneData[0,:])
)

#%% Save everything
