#!/usr/bin/env bash

workspace="/workspace/bert"
methods=("hellinger")
proc=30
semcor="data/semcor/semcor.data.xml"
bert="data/bert/semcor.bert.dense.l24.npy"
test_weights="data/bert/all.bert.dense.l24.npy"
test_words="data/semcor/semcor.eval.data.xml"
cd ../
for method in ${methods[*]}
do
    python interpretability_cli.py --test_words_weights "$test_weights" --test_words "$test_words" --embedding_path "$bert" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --workspace $workspace --name "${method}_dense" -save --accuracy "accuracy@1";
done