# Interpretability of Word Embeddings

# Requirements

Python 3.8+

Every dependency can be found in the [requirements.txt](requirements.txt).<br>
`pip install -r requirements.txt`

# Example
`python interpretability_cli.py --embedding_path '../shared/embeddings/glove/dense/glove.6B.300d.txt' -dense --lines_to_read 50000 --smc_path '../shared/semcat/Categories/' --smc_method 'category_center' --processes 6 --distance 'hellinger_normal' --workspace 'test' -save -interpretability`