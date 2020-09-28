#!/usr/bin/env bash

# general params

workspace="/data/ficstamas/workspace/sense-bert_modified-distance/"
proc=40
semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
test_words="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

path="/data/ficstamas/representations/"

# Dense paths

dense_embeddings=("sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy" \
                  "sensebert-large-uncased/semcor.sensebert-large-uncased.layer_21-22-23-24.npy")

dense_eval_embeddings=("sensebert-base-uncased/ALL.sensebert-base-uncased.layer_9-10-11-12.npy" \
                       "sensebert-large-uncased/ALL.sensebert-large-uncased.layer_21-22-23-24.npy")

dense_embedding_name=("sensebert-base-uncased_layer-9-10-11-12_dense" "sensebert-large-uncased_layer-21-22-23-24_dense")
# Sparse paths

sparse_embeddings=("sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05_semcor.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05.npz" \
                   "sensebert-large-uncased/semcor.sensebert-large-uncased.layer_21-22-23-24.npy_normTrue_K1500_lda0.05_semcor.sensebert-large-uncased.layer_21-22-23-24.npy_normTrue_K1500_lda0.05.npz")

sparse_eval_embeddings=("sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05_ALL.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05.npz" \
                        "sensebert-large-uncased/semcor.sensebert-large-uncased.layer_21-22-23-24.npy_normTrue_K1500_lda0.05_ALL.sensebert-large-uncased.layer_21-22-23-24.npy_normTrue_K1500_lda0.05.npz")

sparse_embedding_name=("sensebert-base-uncased_layer-9-10-11-12_sparse" "sensebert-large-uncased_layer-21-22-23-24_sparse")

methods=("hellinger" "bhattacharyya" "hellinger_normal" "bhattacharyya_normal")

cd ../

for method in ${methods[*]}
do
#    :'for i in "${!dense_embeddings[@]}"
#    do
#        embedding="${path}/${dense_embeddings[$i]}"
#        test_embedding="${path}/${dense_eval_embeddings[$i]}"
#        python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${dense_embedding_name[$i]}_${method}" -save --accuracy "accuracy@1";
#    done'
    for i in "${!sparse_embeddings[@]}"
    do
        embedding="${path}/${sparse_embeddings[$i]}"
        test_embedding="${path}/${sparse_eval_embeddings[$i]}"
        python interpretability_cli.py --test_words_weights "$test_embedding" --test_words "$test_words" --embedding_path "$embedding" -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes $proc --distance "$method" --model "contextual" --workspace $workspace --name "${sparse_embedding_name[$i]}_${method}" -save --accuracy "accuracy@1";
    done
done

