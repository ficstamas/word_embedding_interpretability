#!/usr/bin/env bash

workspace="mcmc"
noises=(0.2 0.5 1.0 2.0 3.0)
methods=("hellinger" "bhattacharyya")
proc=20

for method in ${methods[*]}
do
  for noise in ${noises[*]}
  do
    python ../interpretability_cli.py --embedding_path '../shared/embeddings/glove/dense/glove.6B.300d.txt' -dense --lines_to_read 50000 --smc_path '../shared/semcat/Categories/' --smc_method 'random' --smc_rate 0.4 --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "accuracy_${noise}" --model 'mcmc' --mcmc_noise "$noise" -save -accuracy;
    python ../interpretability_cli.py --embedding_path '../shared/embeddings/glove/dense/glove.6B.300d.txt' -dense --lines_to_read 50000 --smc_path '../shared/semcat/Categories/' --smc_method 'category_center' --smc_rate 0.4 --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "interpret_${noise}" --model 'mcmc' --mcmc_noise "$noise" -save -interpretability;
  done
done