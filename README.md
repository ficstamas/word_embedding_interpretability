# Interpretability of Word Embeddings

Provides a tool to generate interpretable word vectors, from existing embedding spaces.

##### Related Work

[Senel et. al., Semantic Structure and Interpretability of Word Embeddings](https://arxiv.org/abs/1711.00331).

##### Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Reproduction](#reproducing-the-results-from-the-papers)
  - [TSD](#tsd)
  - [MSZNY](#mszny2021-conference-on-hungarian-computational-linguistics)
- [Citation](#citation)


# Requirements

Python 3.8+

Every dependency can be found in the [requirements.txt](requirements.txt) file.<br>
`pip install -r requirements.txt`

# Usage

```
interpretability_cli.py [-h] --embedding_path <EMBEDDING_PATH> 
                               [-dense]
                               [-numpy] 
                               [--lines_to_read LINES_TO_READ]
                               [--smc_path SMC_PATH] 
                               [--smc_loader SMC_LOADER]
                               [--smc_method SMC_METHOD] 
                               [--smc_rate SMC_RATE]
                               [--seed SEED] 
                               [--kde_kernel KDE_KERNEL]
                               [--kde_bandwidth KDE_BANDWIDTH] 
                               [--distance DISTANCE]
                               [--distance_weight {none,relative_frequency}]
                               [--workspace WORKSPACE] 
                               [--name NAME]
                               [--processes PROCESSES] 
                               [--model MODEL] 
                               [-load]
                               [-save] 
                               [-overwrite]
                               [-interpretability]
                               [--accuracy ACCURACY] 
                               [--relaxation RELAXATION]
                               [--test TEST] 
                               [--test_weights TEST_WEIGHTS]
                               [--mcmc_acceptance MCMC_ACCEPTANCE]
                               [--mcmc_noise MCMC_NOISE]
```

#### General Parameters
- **seed** - Random seed used through the whole runtime
- **processes** - The maximum number of processes utilized (It's capped to the number of available physical cores if more than that is provided)
- **save** - Save the model
- **load** - Load a saved model
- **overwrite** - Whether to overwrite existing projects (If not used and a project exists at the given path then the program terminates with exit code 0)
- **workspace** - A workspace is created at the provided path (**Required**)
- **name** - Name of the experiment (Creates a folder with the provided name in the workspace folder)
- **model** - Model used for interpretable embedding calculation
  - _default_ - Used for static embeddings
  - _contextual_ - Used for contextual embeddings
  - _mcmc_ - Markov chain Monte Carlo simulation based method

#### Source Embedding Related Parameters
- **embedding_path** - Path to the input embedding (**Required**)
- **dense** - Use if the input is a dense embedding
- **numpy** - Use if the input is a numpy object (_.npy_,_.npz_)
- **lines_to_read** - Number of lines (or vectors) to read from the embedding

#### Semantic Category Related Parameters
- **smc_path** - Path to the semantic categories
- **smc_loader** - Loader defines the format of semantic categories in the file system
- **smc_method** - Method to drop words from semantic categories
- **smc_rate** - The percentage of words to drop

#### Model Specific Parameters
##### Contextual Embedding specific
- **test** - Path to the test words. 
- **test_weights** - Path to NumPy array which contains the weights for the test words.
##### MCMC Model specific
- **mcmc_acceptance** - The minimum number of accepted estimation during Metropolis–Hastings algorithm
- **mcmc_noise** - The percentage of noise applied to every semantic category.

#### Distance Related Parameters
- **distance** - The utilized distance
  - _hellinger_ - Continuous form of Hellinger distance, relies on Kernel Density Estimation
  - _bhattacharyya_ - Continuous form of Bhattacharyya distance, relies on Kernel Density Estimation
  - _hellinger_normal_ - Closed form of Hellinger distance which assumes Normal distribution
  - _bhattacharyya_normal_ - Closed form of Bhattacharyya distance which assumes Normal distribution
  - _hellinger_exponential_ - Closed form of Hellinger distance which assumes Exponential distribution
  - _bhattacharyya_exponential_ - Closed form of Bhattacharyya distance which assumes Exponential distribution
- **distance_weight** - Provides a weight to calculated distances
  - _none_ - Don't use any (constant 1 multiplier)
  - _relative_frequency_ - Relative frequency of category words
- **kde_kernel** - Applied kernel for Kernel Density Estimation (for available kernels [see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity))
- **kde_bandwidth** - Bandwidth of the kernel

#### Validation Related Parameters
- **interpretability** - Calculates the interpretability score
- **relaxation** - Relaxation parameter for the interpretability score
- **accuracy** - Accuracy calculation method
  - _word_retrieval_test_ - Word Retrieval Test
  - _accuracy_ - Can use additional relaxation parameter after an @ symbol e.g.: accuracy@10 (Only available for contextual models atm.)


# Reproducing the results from the papers

## TSD
The paper which was submitted to the **23rd International Conference on Text, Speech and Dialogue** conference is available [here](docs/tsd_paper.pdf)

GloVe can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip) and the SemCat dataset is available [here](https://github.com/avaapm/SEMCATdataset2018).

Run [tsd_experiments.sh](experiments/tsd_expriments.sh) after changing the _glove_path_ and _semcat_path_ variables. Furthermore change the _proc_ variable according to your CPU cores (Default: 30), but it is not going to spawn/fork more processes than the available number of physical cores.

## MSZNY2021 (Conference on Hungarian Computational Linguistics)
To reproduce the results from the MSZNY paper, download the following embeddings:

- [Hungarian Fasttext (Wiki)](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hu.vec)
- [Hungarian Fasttext Aligned (Wiki)](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hu.align.vec)
- [Szeged Word Vectors](http://www.inf.u-szeged.hu/~szantozs/fasttext/)
- [Semantic Categories](https://github.com/ficstamas/multilingual_semantic_categories/blob/master/categories/semcat_en-de-hu.json)

We generated the sparse embedding spaces with the following [script](https://github.com/begab/interpretability_aaai2020/blob/master/src/sparse_coding/sparse_coding.py). Parameters can be found in [mszny_sparse.sh](experiments/mszny_sparse.sh).

After changing some parameters in [mszny.sh](experiments/mszny.sh) you can run the script.
<br>
Parameters to change:
- _path_ - points at the folder which contains the embeddings
- _embeddings_ - if the name of the embeddings have changed
- _workspace_ - to set the output path
- _proc_ - which defines the number of utilized processes
# Citation 

If you are using the code or relying on the paper please cite the following paper:

```
@InProceedings{10.1007/978-3-030-58323-1_21,
  author="Ficsor, Tam{\'a}s
  and Berend, G{\'a}bor",
  editor="Sojka, Petr
  and Kope{\v{c}}ek, Ivan
  and Pala, Karel
  and Hor{\'a}k, Ale{\v{s}}",
  title="Interpreting Word Embeddings Using a Distribution Agnostic Approach Employing Hellinger Distance",
  booktitle="Text, Speech, and Dialogue",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="197--205",
  abstract="Word embeddings can encode semantic and syntactic features and have achieved many recent successes in solving NLP tasks. Despite their successes, it is not trivial to directly extract lexical information out of them. In this paper, we propose a transformation of the embedding space to a more interpretable one using the Hellinger distance. We additionally suggest a distribution-agnostic approach using Kernel Density Estimation. A method is introduced to measure the interpretability of the word embeddings. Our results suggest that Hellinger based calculation gives a  1.35{\%} improvement on average over the Bhattacharyya distance in terms of interpretability and adapts better to unknown words.",
  isbn="978-3-030-58323-1"
}
```
