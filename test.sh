#!/usr/bin/env bash

semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
all="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

train="/data/berend/representations_unreduced/sensebert-large-uncased/semcor.data.xml_sensebert-large-uncased_avg_False_layer_21-22-23-24.npy"
test="/data/berend/representations_unreduced/sensebert-large-uncased/ALL.data.xml_sensebert-large-uncased_avg_False_layer_21-22-23-24.npy"
workspace="/data/ficstamas/test/"
config="/home/ficstamas/word_embedding_interpretability_dev/config/default.json"

python -m src.preprocessing-cli --train "$train" --path "$workspace" --configuration "$config"
python -m src.distance-cli --train "$train" --train_labels "$semcor" --label_processor "semcor-lexname" --path "$workspace" --distance "hellinger_normal" --jobs "30"
python -m src.evaluation-cli --test "$test" --test_labels "$all" --label_processor "semcor-lexname" --path "$workspace" --evaluation_method "argmax" -label_frequency
