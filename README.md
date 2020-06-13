# Interpretability of Word Embeddings

# Requirements

Python 3.8+

Every dependency can be found in the [requirements.txt](requirements.txt).<br>
`pip install -r requirements.txt`

# TSD recreation of results
Glove can be downloaded [here](http://nlp.stanford.edu/data/glove.6B.zip) and the SemCat dataset is available [here](https://github.com/avaapm/SEMCATdataset2018).

Run [tsd_experiment.sh](experiments/tsd_expriments.sh) after changing the _glove_path_ and _semcat_path_ variables. Furthermore change the _proc_ variable according to your CPU cores (Default: 30), but it is not going to spawn/fork more processes than the available number of physical cores.

# Example
`python interpretability_cli.py --embedding_path '../shared/embeddings/glove/dense/glove.6B.300d.txt' -dense --lines_to_read 50000 --smc_path '../shared/semcat/Categories/' --smc_method 'category_center' --processes 6 --distance 'hellinger_normal' --workspace 'test' -save -interpretability`