# Datasets

**Name:** Macular degeneration and dermal fibroblast response to sublethal oxidative stress.
- **GEO accession number:** GDS963
- **Size:** 36×12625

**Name:** Myelodysplastic syndrome: CD34+ hematopoietic stem cells.
- **GEO accession number:** GDS3795
- **Size:** 200×54675

**Name:** Pediatric acute leukemia patients with early relapse: white blood cells.
- **GEO accession number:** GDS4206
- **Size:** 197×54675

# Approaches

The primary challenges posed by these datasets (and many biological datasets) are the low number of data points and the high dimensionality. One way we could “pitch” this project would be to try a wide variety of methods to combat these challenges and see which ones work the best. We could treat it as sort of “a survey of strategies used to boost performance on biological datasets” or something similar. I think we should try the following algorithms and see which strategy helps the performance of each one:

- Decision Tree (DT)
- Random Forest (RF)
- Naïve Bayes Classifier (NBC)
- Support Vector Machine (SVM)
- *k*-Nearest Neighbours (*k*-NN)

## Strategies to Combat High Dimensionality

The high dimensionality of the datasets poses numerous problems (the “curse of dimensionality”) especially given how few data points each one has.

### Strategy 1: Do nothing
One way of dealing with any problem is to just ignore it. This could provide us with a good base point to compare performance with. We should anticipate that the results will be pretty bad, and this would give us an opportunity to discuss the issues with using the dataset as-is. While the DT and RF may not be as affected by the high dimension, the NBC, SVM, and *k*-NN will all likely suffer from the high dimension.

### Strategy 2: Data Grouping
Since we’re working in a biological system, we could group related genes together using *a priori* biological knowledge (i.e. GO categories, which are provided in the SOFT file for the dataset). This would reduce the dimension from expression of individual genes to average expression of gene sets/pathways. We could see how the algorithms perform on the reduced dimension of the dataset. To do this aggregation, we could probably take the mean (arithmetic or geometric) of the values for each data point corresponding to each gene in the category.

### Strategy 3: Dimensionality Reduction
Another strategy to combat the absurdly high dimension is to use Principle Component Analysis (PCA) to reduce the number of dimensions to something more manageable, then train the algorithms on the dataset in the subspace.

### Strategy 4: Data Encoding
We could use the Variational Auto-Encoder (VAE) to encode the data to a lower-dimensional space in a generalizable but nonlinear way that preserves more information.

### Strategy 5: Cheat
There are preexisting models on gene expression data (see some of the models trained on https://huggingface.co/datasets/tahoebio/Tahoe-100M), so we could also try using the embeddings/representations from a preexisting model and running our algorithms on the data in that preexisting representation space.

## Strategies to Combat Low Number of Data Points
The low number of data points poses a major problem for many algorithms and for evaluating the models, but there are a few strategies we could use to combat this issue.

### Strategy a: Do Nothing
Again, we could try not doing anything, which could provide us with a good base point to compare performance with. We should again anticipate that the results will be pretty bad, and this would give us an opportunity to discuss the issues with using the dataset as-is.

We could discuss the impracticality of doing an 80-20 split, since we’d then only be training on 28 data points and validating on 8. A validation/test set of 8 is very uninformative, since you can only assess accuracy in 12.5% intervals.

So, we’d go with leave-one-out validation. However, this poses a problem for methods like the DT and RF, since, without a validation set, we’d need to just prune the tree at some prespecified depth, which is difficult to prescribe ahead of time.

### Strategy b: Simulated Data
We could generate new data points by sampling random points from the VAE. Because the VAE encodes points in a generalizable way, sampling from the representation space and decoding back to the data space should give us data points similar to the points in the original dataset. We can then train on this augmented dataset. In this case, maybe we could reserve the original dataset for testing/validation (I’m not sure how sensible that would be, since we’re using the dataset to generate the encoding in the first place)?

# Expected Results
So, in the end, we would have a lot of data to talk about and discuss in the paper. We’re going to be comparing 5 different strategies for combatting the high dimensionality (columns) and 2 strategies for combatting the low number of data points (rows), resulting in 5×2=10 different combinations. For each combination, we’d be testing all 5 models.

<table>
	<tr>
		<th></th>
		<th></th>
		<th colspan="5"> Strategies to Combat High Dimensionality</th>
	</tr>
    <tr>
      	<td style="border-bottom:1px solid black"></td>
      	<td style="border-bottom:1px solid black;border-right:1px solid black"></td>
        <td style="border-bottom:1px solid black">1</td>
        <td style="border-bottom:1px solid black">2</td>
        <td style="border-bottom:1px solid black">3</td>
        <td style="border-bottom:1px solid black">4</td>
        <td style="border-bottom:1px solid black">5</td>
  	</tr>
	<tr>
		<th rowspan="3", style="text-align:right">Strategies to Combat Low Number of Data Points</th>
		<td style="border-right:1px solid black;text-align:right">a</td>
		<td>1a</td>
		<td>2a</td>
		<td>3a</td>
		<td>4a</td>
		<td>5a</td>
	</tr>
	<tr>
		<td style="border-right:1px solid black;text-align:right">b</td>
		<td>1b</td>
		<td>2b</td>
		<td>3b</td>
		<td>4b</td>
		<td>5b</td>
	</tr>
</table>

So, each cell of this table corresponds to a different, modified dataset that has been transformed/augmented in some way with some combination of strategies to combat the high dimensionality and low number of data points. For each of these strategy combinations, we’d train and evaluate all 5 models. This means we’ll train and evaluate the DT, RF, NBC, SVM, and k-NN with 1a, then we’d do it again with 1b, and so on. We’d be training and evaluating 5×10=50 models.

# Goals/To Do

It seems like a lot of work (and it probably is), but I think it’s doable. We should set up a program that can just accept a training dataset, then we just pass it the modified training datasets. Once it’s all set up, between the lot of us, we should have enough computing power to get this done in a feasible amount of time. Since there are 5 of us, we could each take on one of the models and each person could run the scripts necessary to get those results for each model.

So, we’d need to:

1. Set up a script that takes in a set of data, then trains and evaluates all 5 models, then spits out the confusion matrices and performance statistics
2. Set up the 15 different training datasets augmented/transformed according to the different strategies
3. Run the script from step 1 on all the datasets
4. Analyze the results and put together a paper/presentation.
