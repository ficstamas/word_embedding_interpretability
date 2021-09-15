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
  - [ACL-IJCNLP 2021 Student Workshop](#acl-ijcnlp-2021-student-workshop)
- [Citation](#citation)


# Requirements

Python 3.8+

Every dependency can be found in the [requirements.txt](requirements.txt) file.<br>
`pip install -r requirements.txt`

If you wish to use the sparse method from preprocessing you need to install [SPAMS](http://spams-devel.gforge.inria.fr) 
separately.

# Usage

The code base is separated into 3 modules.
- [Preprocessing](src/preprocessing-cli.py) (optional, you can do your preprocessing steps separately)
  - **train** - Path to the embedding space (for training)
  - **path** - Path to a project folder (later you have to set the same folder for the calculations)
  - **configuration** - Path to a configuration JSON file (e.g. [this](config/example.json))
    - It builds up like this:```{
  "<priority>": {
    "name": "<name>",
    "params": {
      ...
    }
}, ...```
    - _priority_ modifies the execution order
    - _method_ can be:
      - With same parameters as in numpy: _std_, _norm_, _center_
      - _whiten_
        - parameters - method: 'zca', 'pca', 'cholesky', 'zca_cor', 'pca_cor'
      - _sparse_ (only on systems where [SPAMS](http://spams-devel.gforge.inria.fr) is supported)
        - parameters: [spams.trainDL](http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5) and [spams.lasso](http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec15)
  - **jobs** - **Deprecated**
- [Calculation of Distance Matrix](src/distance-cli.py)
  - **train** - Path to the embedding space (for training)
  - **no_transform** - Flag to do not apply preprocessing step if a config exists in the project folder
  - **train_labels** - Path to file containing labels (for SemCor `*.data.xml` file)
  - **label_processor** - Method to load labels into memory (Right now `semcor-lexname` only)
  - **path** - Path to project (a place to save files, or the same as provided during preprocessing)
  - **distance** - Distance to apply (`'bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal',
                                 'bhattacharyya_exponential', 'hellinger_exponential'`)
  - **kde_kernel** - Kernel to use if `bhattacharyya` or `hellinger` was provided as distance. (`'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'`)
  - **kde_bandwidth** - Bandwidth to use for kernel density estimation 
  - **jobs** - Number of processes to use during distance calculation (it is always `min(provided, number_of_physical_cores)`)
- [Evaluation](src/distance-cli.py)
  - **test** - Path to the embedding space (for evaluation)
  - **no_transform** - Flag to do not apply preprocessing step if a config exists in the project folder
  - **test_labels** - Path to file containing labels (for SemCor `*.data.xml` file)
  - **label_processor** - Method to load labels into memory (Right now `semcor-lexname` only)
  - **path** - Path to project (a place to save files, or the same as provided during preprocessing)
  - **save** - To save the interpretable space
  - **label_frequency** - Applying label frequency based weighting on the output embedding.
  - **evaluation_method** - How to measure interpretability (`argmax` only)
  - **devset_name** - Name of the devset (good if you wish to use one for parameter selection)


# Reproducing the results from the papers

## TSD
The paper which was submitted to the **23rd International Conference on Text, Speech and Dialogue** conference is available [here](docs/tsd_paper.pdf)

GloVe can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip) and the SemCat dataset is available [here](https://github.com/avaapm/SEMCATdataset2018).

## MSZNY2021 (Conference on Hungarian Computational Linguistics)
Link to the [paper](http://publicatio.bibl.u-szeged.hu/20761/).

To reproduce the results from the MSZNY paper, download the following embeddings:

- [Hungarian Fasttext (Wiki)](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hu.vec)
- [Hungarian Fasttext Aligned (Wiki)](https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.hu.align.vec)
- [Szeged Word Vectors](http://www.inf.u-szeged.hu/~szantozs/fasttext/)
- [Semantic Categories](https://github.com/ficstamas/multilingual_semantic_categories/blob/master/categories/semcat_en-de-hu.json)

We generated the sparse embedding spaces with the following [script](https://github.com/begab/interpretability_aaai2020/blob/master/src/sparse_coding/sparse_coding.py). Parameters can be found in [mszny_sparse.sh](experiments/mszny_sparse.sh).


## ACL-IJCNLP 2021 Student Workshop

Paper is available [here](docs/ACL_IJCNLP_SRW_2021_Proceedings.pdf)

We included the [configuration file](config/aclsw.json) for the preprocessing step. 
We generated the sparse embeddings separately with the following [script](https://github.com/begab/interpretability_aaai2020/blob/master/src/sparse_coding/sparse_coding.py).

# Citation 

If you are using the code or relying on the paper please cite the following paper(s):

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
