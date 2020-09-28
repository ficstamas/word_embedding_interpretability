#!/usr/bin/env bash

workspace="/data/ficstamas/workspace/sparse-sensebert-bert-uncased/"
proc=30
semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
test_words="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

path="/data/ficstamas/representations/"

dimensions=("1500" "2000" "3000")
lambdas=("0.05" "0.1" "0.2")
methods=("hellinger" "bhattacharyya" "hellinger_normal" "bhattacharyya_normal")
complexities=("base" "large")
models=("bert" "sensebert")
sm=("semcor" "ALL")

cd ../;

# bert
i=0
for dimension in ${dimensions[*]}
do
  for complexity in ${complexities[*]}
  do
    for lambda in ${lambdas[*]}
    do
      if [[ "$lambda" == "0.05" ]] && [[ "$dimension" == "1500" ]]
      then
        continue
      fi
      for model in ${models[*]}
      do
        for method in ${methods[*]}
        do

          file=""
          layer=""
          train=""
          test=""
          name=""

          if [[ "$model" == "bert" ]]; then
            # bert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi
            file="semcor.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_##REP##.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}.npz"
            train="${path}bert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}bert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[1]}}"
            name="bert-${complexity}-uncased_layer-${layer}_sparse_lda-${lambda}_K${dimension}_${method}"
          else
            # sensebert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi
            file="semcor.sensebert-${complexity}-uncased.layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_##REP##.sensebert-${complexity}-uncased.layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}.npz"
            train="${path}sensebert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}sensebert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[1]}}"
            name="sensebert-${complexity}-uncased_layer-${layer}_sparse_lda-${lambda}_K${dimension}_${method}"
          fi
          echo ${i};
          #echo ${file};
          #echo ${layer};
          #echo ${train};
          #echo ${test};
          #echo ${name};
          #echo "=============================================";
          i=$((i+1));
          python interpretability_cli.py --test_words_weights "$test" --test_words "$test_words" --embedding_path "$train" -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes "$proc" --distance "$method" --model "contextual" --workspace $workspace --name "${name}" -save --accuracy "accuracy@1";
          echo "$i/128 -- $?" >> "progress.txt";

        done
      done
    done
  done
done

