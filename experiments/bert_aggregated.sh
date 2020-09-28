#!/usr/bin/env bash

# general params

workspace="/data/ficstamas/workspace/bert_new/"
proc=30
semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
test_words="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

path="/data/berend/representations/bert-large-cased"
methods=("hellinger" "bhattacharyya" "hellinger_normal" "bhattacharyya_normal")

cd ../


bert_type="bert-large-cased"

for method in ${methods[*]}
do
    type="dense"
    layer="layer_21-22-23-24"

    embedding="${path}/semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy"
    test_embedding="${path}/ALL.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy"
    python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${bert_type}_${layer}_${method}_${type}" -save --accuracy "accuracy@1";

    layer="layer_24"
    embedding="${path}/semcor.data.xml_bert-large-cased_avg_False_layer_24.npy"
    test_embedding="${path}/ALL.data.xml_bert-large-cased_avg_False_layer_24.npy"
    python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${bert_type}_${layer}_${method}_${type}" -save --accuracy "accuracy@1";

    # Sparse

    layer="layer_21-22-23-24"
    type="sparse"
    embedding="${path}/semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05_semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05.npz"
    test_embedding="${path}/semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05_ALL.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05.npz"
    python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${bert_type}_${layer}_${method}_${type}" -save --accuracy "accuracy@1";

    layer="layer_24"
    embedding="${path}/semcor.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05_semcor.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05.npz"
    test_embedding="${path}/semcor.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05_ALL.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05.npz"
    python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${bert_type}_${layer}_${method}_${type}" -save --accuracy "accuracy@1";
done

#bert-large
#ALL.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy
#semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy
#
#semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05_ALL.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05.npz
#semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05_semcor.data.xml_bert-large-cased_avg_False_layer_21-22-23-24.npy_normTrue_K1500_lda0.05.npz
#
#bert-base
#
#ALL.data.xml_bert-large-cased_avg_False_layer_24.npy
#semcor.data.xml_bert-large-cased_avg_False_layer_24.npy
#
#semcor.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05_ALL.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05.npz
#semcor.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05_semcor.data.xml_bert-large-cased_avg_False_layer_24.npy_normTrue_K1500_lda0.05.npz

