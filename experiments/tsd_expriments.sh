#!/usr/bin/env bash

workspace="workspace/tsd"
methods=("hellinger_normal" "bhattacharyya_normal" "hellinger" "bhattacharyya")
proc=30
number_of_words=50000
glove_path='../shared/embeddings/glove/dense/glove.6B.300d.txt'
semcat_path='../shared/semcat/Categories/'
drop_rate=(0.0 0.2 0.4)

cd ../

for method in ${methods[*]}
do
  python interpretability_cli.py --embedding_path "$glove_path" -dense --lines_to_read "$number_of_words" --smc_path "$semcat_path" --smc_method 'random' --smc_rate 0.4 --seed 404 --processes $proc --distance "$method" --workspace "${workspace}/${method}" --name "accuracy" -save -accuracy;
  for dr in ${drop_rate[*]}
  do
    python interpretability_cli.py --embedding_path "$glove_path" -dense --lines_to_read "$number_of_words" --smc_path "$semcat_path" --smc_method 'category_center' --smc_rate "$dr" --seed 404 --processes $proc --distance "$method" --workspace "${workspace}/${method}" --name "${dr}_interpret" -save -interpretability;
  done
done