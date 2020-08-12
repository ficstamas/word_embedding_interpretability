#!/usr/bin/env bash

semcor_train="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
semcor_test="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"
path_str="/data/ficstamas/workspace/sense-bert/"
path=/data/ficstamas/workspace/sense-bert/

out="/data/ficstamas/workspace/baseline_results/logreg_results.txt"

cd ..

for f in ${path}*; do
    if [[ -d "$f" ]]; then
        IFS='/' read -ra FOLDERS <<< "$f"
        train="${f}/saves/transformed_space.npy"
        test="${f}/saves/validation_transformed_space.npy"
        name="${FOLDERS[-1]}"
        python -m scripts.logistic_regressor --train_vectors "$train" --test_vectors "$test" --semcor_train "$semcor_train" --semcor_test "$semcor_test" --name "$name" --output "$out"
    fi
done