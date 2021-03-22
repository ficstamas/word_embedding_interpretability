#!/usr/bin/env bash

semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
all="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

train="/data/berend/representations_unreduced/sensebert-large-uncased/semcor.data.xml_sensebert-large-uncased_avg_False_layer_21-22-23-24.npy"
test="/data/berend/representations_unreduced/sensebert-large-uncased/ALL.data.xml_sensebert-large-uncased_avg_False_layer_21-22-23-24.npy"

jobs=30

# Sparse
workspace="/data/ficstamas/workspace_tests/test_s/"
config="/home/ficstamas/word_embedding_interpretability_dev/config/test_s.json"

# python -m src.preprocessing-cli --train "$train" --path "$workspace" --configuration "$config"
# python -m src.distance-cli --train "$train" --train_labels "$semcor" --label_processor "semcor-lexname" --path "$workspace" --distance "hellinger_normal" --jobs "${jobs}"
# python -m src.evaluation-cli --test "$test" --test_labels "$all" --label_processor "semcor-lexname" --path "$workspace" --evaluation_method "argmax" -label_frequency

# Sparse -> White

workspace="/data/ficstamas/workspace_tests/test_sw/"
config="/home/ficstamas/word_embedding_interpretability_dev/config/test_sw.json"

# python -m src.preprocessing-cli --train "$train" --path "$workspace" --configuration "$config"
# python -m src.distance-cli --train "$train" --train_labels "$semcor" --label_processor "semcor-lexname" --path "$workspace" --distance "hellinger_normal" --jobs "${jobs}"
# python -m src.evaluation-cli --test "$test" --test_labels "$all" --label_processor "semcor-lexname" --path "$workspace" --evaluation_method "argmax" -label_frequency

# White -> Sparse

workspace="/data/ficstamas/workspace_tests/test_ws/"
config="/home/ficstamas/word_embedding_interpretability_dev/config/test_ws.json"

# python -m src.preprocessing-cli --train "$train" --path "$workspace" --configuration "$config"
# python -m src.distance-cli --train "$train" --train_labels "$semcor" --label_processor "semcor-lexname" --path "$workspace" --distance "hellinger_normal" --jobs "${jobs}"
# python -m src.evaluation-cli --test "$test" --test_labels "$all" --label_processor "semcor-lexname" --path "$workspace" --evaluation_method "argmax" -label_frequency

# White -> Sparse -> White

workspace="/data/ficstamas/workspace_tests/test_wsw/"
config="/home/ficstamas/word_embedding_interpretability_dev/config/test_wsw.json"

# python -m src.preprocessing-cli --train "$train" --path "$workspace" --configuration "$config"
# python -m src.distance-cli --train "$train" --train_labels "$semcor" --label_processor "semcor-lexname" --path "$workspace" --distance "hellinger_normal" --jobs "${jobs}"
# python -m src.evaluation-cli --test "$test" --test_labels "$all" --label_processor "semcor-lexname" --path "$workspace" --evaluation_method "argmax" -label_frequency

workspace="/data/ficstamas/workspace_tests/test_whiten_new/"
config="/home/ficstamas/word_embedding_interpretability_dev/config/test_whiten_new.json"

python -m src.preprocessing-cli --train "$train" --path "$workspace" --configuration "$config"
python -m src.distance-cli --train "$train" --train_labels "$semcor" --label_processor "semcor-lexname" --path "$workspace" --distance "hellinger_normal" --jobs "${jobs}"
python -m src.evaluation-cli --test "$test" --test_labels "$all" --label_processor "semcor-lexname" --path "$workspace" --evaluation_method "argmax" -label_frequency
