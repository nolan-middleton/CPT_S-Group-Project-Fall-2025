# Imports
from Functions import reformat_data
import os as os

soft_files = [
    "Raw/GDS963_full.soft",
    "Raw/GDS3795_full.soft",
    "Raw/GDS4206_full.soft"
]
output_dirs = [
    "MacularDegeneration",
    "MyelodysplasticSyndrome",
    "AcuteLymphblasticLeukemia"
]

for i in range(len(soft_files)):
    if (not os.path.isdir(output_dirs[i])):
        os.mkdir(output_dirs[i])
    subsets = reformat_data(soft_files[i], output_dirs[i] + "/Data")
    with open(output_dirs[i] + "/metadata.txt", "w") as file:
        file.write("Source file: " + soft_files[i] + "\n")
        file.write("Classes: ")
        file.write(", ".join([subset["data"] for subset in subsets]) + "\n")

with open("datasets.txt", "w") as file:
    for D in output_dirs:
        file.write(D + "\n")