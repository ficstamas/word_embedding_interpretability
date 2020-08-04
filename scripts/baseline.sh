#!/usr/bin/env bash

cd ..

semcor_train="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
semcor_test="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

bert_train="/data/berend/representations/semcor.data.xml_bert-large-cased_avg_False_layer_24.npy"
bert_test="/data/berend/representations/ALL.data.xml_bert-large-cased_avg_False_layer_24.npy"

bert_base_train="/data/berend/representations/semcor.data.xml_bert-base-cased_avg_False_layer_12.npy"
bert_base_test="/data/berend/representations/ALL.data.xml_bert-base-cased_avg_False_layer_12.npy"

sense_bert_train="/data/ficstamas/representations/sensebert-large-uncased/semcor.sensebert-large-uncased.layer_24.npy"
sense_bert_test="/data/ficstamas/representations/sensebert-large-uncased/ALL.sensebert-large-uncased.layer_24.npy"

sense_bert_base_train="/data/ficstamas/representations/sensebert-base-uncased/semcor.sensebert-base-uncased.layer_12.npy"
sense_bert_base_test="/data/ficstamas/representations/sensebert-base-uncased/ALL.sensebert-base-uncased.layer_12.npy"

python -m scripts.category_center_distance_baselines --train "$bert_train" --test "$bert_test" --semcor_train "$semcor_train" --semcor_test "$semcor_test" --output "/data/ficstamas/workspace/baseline_results/bert-large-cased/"
python -m scripts.category_center_distance_baselines --train "$bert_base_train" --test "$bert_base_test" --semcor_train "$semcor_train" --semcor_test "$semcor_test" --output "/data/ficstamas/workspace/baseline_results/bert-base-cased/"
python -m scripts.category_center_distance_baselines --train "$sense_bert_train" --test "$sense_bert_test" --semcor_train "$semcor_train" --semcor_test "$semcor_test" --output "/data/ficstamas/workspace/baseline_results/sensebert-large-cased/"
python -m scripts.category_center_distance_baselines --train "$sense_bert_base_train" --test "$sense_bert_base_test" --semcor_train "$semcor_train" --semcor_test "$semcor_test" --output "/data/ficstamas/workspace/baseline_results/sensebert-base-cased/"