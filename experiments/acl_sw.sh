#!/usr/bin/env bash


proc=40
semcor="/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml"
test_words="/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml"

path="/data/ficstamas/representations/whitened/"

dimensions=("1500" "3000")
complexities=("base" "large")
lambdas=("0.05" "0.1" "0.2")
models=("bert" "sensebert")
methods=("hellinger_normal")
sm=("semcor" "ALL")
whitenings=("zca")

cd ../;

# bert
i=0
for dimension in ${dimensions[*]}
do
  for complexity in ${complexities[*]}
  do
    for lambda in ${lambdas[*]}
    do
      for model in ${models[*]}
      do
        for method in ${methods[*]}
        do
          for white in ${whitenings[*]}
          do
          workspace="/data/ficstamas/workspace/eacl-whitened-${white}/"
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
            file="semcor.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_##REP##.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_whitened-${white}.npy"
            train="${path}${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}${file/\#\#REP\#\#/${sm[1]}}"
            name="bert-${complexity}-uncased_layer-${layer}_sparse_lda-${lambda}_K${dimension}_${method}"
          else
            # sensebert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi

            file="semcor.data.xml_sensebert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_##REP##.data.xml_sensebert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_whitened-${white}.npy"
            train="${path}${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}${file/\#\#REP\#\#/${sm[1]}}"
            name="sensebert-${complexity}-uncased_layer-${layer}_sparse_lda-${lambda}_K${dimension}_${method}"
          fi
          echo "=============================================";
          i=$((i+1));
          python interpretability_cli.py --test_weights "$test" --test "$test_words" --embedding_path "$train" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes "$proc" --distance "$method" --model "contextual" --workspace $workspace --name "${name}" -load --accuracy "accuracy@1" -accuracy_preprocessed -accuracy_recalculate;
          done
        done
      done
    done
  done
done

# dense

for complexity in ${complexities[*]}
  do
    for model in ${models[*]}
    do
        for method in ${methods[*]}
        do
          for white in ${whitenings[*]}
          do
          workspace="/data/ficstamas/workspace/eacl-whitened-${white}/"
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
            file="##REP##.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}_whitened-${white}.npy"
            train="${path}${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}${file/\#\#REP\#\#/${sm[1]}}"
            name="bert-${complexity}-uncased_layer-${layer}_dense_${method}"
          else
            # sensebert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi

            file="##REP##.data.xml_sensebert-${complexity}-uncased_avg_False_layer_${layer}_whitened-${white}.npy"
            train="${path}${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}${file/\#\#REP\#\#/${sm[1]}}"
            name="sensebert-${complexity}-uncased_layer-${layer}_dense_${method}"
          fi
          i=$((i+1));
          python interpretability_cli.py --test_weights "$test" --test "$test_words" --embedding_path "$train" -dense -numpy --lines_to_read -1 --smc_loader='semcor' --smc_path "$semcor" --processes "$proc" --distance "$method" --model "contextual" --workspace $workspace --name "${name}" -load --accuracy "accuracy@1" -accuracy_preprocessed -accuracy_recalculate;
          done
        done
    done
  done

