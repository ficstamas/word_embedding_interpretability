#!/usr/bin/env bash

workspace="/data/ficstamas/workspace/sense-bert"
proc=30
semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
test_words="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

path="/data/ficstamas/representations/"

dense_layers=("sensebert-base-uncased.layer_12.npy" "sensebert-large-uncased.layer_24.npy")
cd ../

# Dense
for dense in ${dense_layers[*]}
do
  methods=("hellinger" "bhattacharyya")
  embedding="${path}semcor.$dense"
  test_embedding="${path}ALL.$dense"
  IFS='.' read -ra TYPE <<< "$dense"

  for method in ${methods[*]}
  do
      python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${TYPE[0]}_${TYPE[1]}_${method}_dense" -save --accuracy "accuracy@1";
  done

  # Normal
  methods=("hellinger_normal" "bhattacharyya_normal")
  for method in ${methods[*]}
  do
      python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${TYPE[0]}_${TYPE[1]}_${method}_dense" -save --accuracy "accuracy@1";
  done
done

sparse_layers=("semcor.sensebert-base-uncased.layer_12.npy_normTrue_K1500_lda0.05_##REP##.sensebert-base-uncased.layer_12.npy_normTrue_K1500_lda0.05.npz" "semcor.sensebert-large-uncased.layer_24.npy_normTrue_K1500_lda0.05_##REP##.sensebert-large-uncased.layer_24.npy_normTrue_K1500_lda0.05.npz")
# Sparse
for sparse in ${sparse_layers[*]}
do
  methods=("hellinger" "bhattacharyya")
  sm=("semcor" "ALL")
  embedding="${path}${sparse/\#\#REP\#\#/${sm[0]}}"
  test_embedding="${path}${sparse/\#\#REP\#\#/${sm[1]}}"
  IFS='.' read -ra TYPE <<< "$sparse"

  for method in ${methods[*]}
  do
      python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${TYPE[1]}_${TYPE[2]}_${method}_sparse" -save --accuracy "accuracy@1";
  done

  # Normal
  methods=("hellinger_normal" "bhattacharyya_normal")
  for method in ${methods[*]}
  do
      python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${TYPE[1]}_${TYPE[2]}_${method}_sparse" -save --accuracy "accuracy@1";
  done
done