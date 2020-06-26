#!/usr/bin/env bash

workspace="/data/ficstamas/workspace/bert_mod"
methods=("hellinger" "bhattacharyya")
proc=30
cd ../
for method in ${methods[*]}
do
    python interpretability_cli.py --embedding_path 'data/bert/semcor.bert.dense.l24.npy' -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path 'data/semcor/semcor.data.xml' --processes $proc --distance "$method" --workspace $workspace --name "${method}_dense" -save --accuracy "accuracy@1";
done

for method in ${methods[*]}
do
    python interpretability_cli.py --embedding_path 'data/bert/semcor.bert.sparse.l24.npz' -sparse -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path 'data/semcor/semcor.data.xml' --processes $proc --distance "$method" --workspace $workspace --name "${method}_sparse" -save --accuracy "accuracy@1";
done