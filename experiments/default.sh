#!/usr/bin/env bash

workspace="baseline"
methods=("hellinger" "bhattacharyya")
proc=20

for method in ${methods[*]}
do
    python ../interpretability_cli.py --embedding_path '../shared/embeddings/glove/dense/glove.6B.300d.txt' -dense --lines_to_read 50000 --smc_path '../shared/semcat/Categories/' --smc_method 'random' --smc_rate 0.4 --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "accuracy" -save -accuracy;
    python ../interpretability_cli.py --embedding_path '../shared/embeddings/glove/dense/glove.6B.300d.txt' -dense --lines_to_read 50000 --smc_path '../shared/semcat/Categories/' --smc_method 'category_center' --smc_rate 0.4 --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "interpret" -save -interpretability;
done