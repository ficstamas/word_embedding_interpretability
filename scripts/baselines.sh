#!/usr/bin/env bash

semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
all="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"
out_tree=/data/ficstamas/workspace/random_forest_baseline.txt
out_log=/data/ficstamas/workspace/logreg_baseline.txt

# sensebert
train=/data/berend/representations/sensebert-large-uncased/semcor.data.xml_sensebert-large-uncased_avg_False_layer_21-22-23-24.npy
test=/data/berend/representations/sensebert-large-uncased/ALL.data.xml_sensebert-large-uncased_avg_False_layer_21-22-23-24.npy

python -m scripts.random_forest --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "sensebert-large-baseline" --output "$out_tree"
python -m scripts.logistic_regressor --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "sensebert-large-baseline" --output "$out_log"

train=/data/berend/representations/sensebert-base-uncased/semcor.data.xml_sensebert-base-uncased_avg_False_layer_9-10-11-12.npy
test=/data/berend/representations/sensebert-base-uncased/ALL.data.xml_sensebert-base-uncased_avg_False_layer_9-10-11-12.npy

python -m scripts.random_forest --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "sensebert-base-baseline" --output "$out_tree"
python -m scripts.logistic_regressor --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "sensebert-base-baseline" --output "$out_log"

# bert
train=/data/ficstamas/representations/bert-large-uncased/semcor.data.xml_bert-large-uncased_avg_False_layer_21-22-23-24.npy
test=/data/ficstamas/representations/bert-large-uncased/ALL.data.xml_bert-large-uncased_avg_False_layer_21-22-23-24.npy

python -m scripts.random_forest --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "bert-large-baseline" --output "$out_tree"
python -m scripts.logistic_regressor --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "bert-large-baseline" --output "$out_log"

train=/data/ficstamas/representations/bert-base-uncased/semcor.data.xml_bert-base-uncased_avg_False_layer_9-10-11-12.npy
test=/data/ficstamas/representations/bert-base-uncased/ALL.data.xml_bert-base-uncased_avg_False_layer_9-10-11-12.npy

python -m scripts.random_forest --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "bert-base-baseline" --output "$out_tree"
python -m scripts.logistic_regressor --train_vectors "$train" --test_vectors "$test" --semcor_train="$semcor" --semcor_test="$all" --name "bert-base-baseline" --output "$out_log"