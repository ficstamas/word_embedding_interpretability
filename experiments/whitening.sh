#!/usr/bin/env bash
path="/data/berend/representations_unreduced/"

dimensions=("1500" "3000")
complexities=("base" "large")
lambdas=("0.05" "0.1" "0.2")
models=("bert" "sensebert")
sm=("semcor" "ALL")
output_path="/data/ficstamas/representations/whitened/"


method="zca_cor"

cd ../scripts/ || exit 1;

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

          file=""
          layer=""
          train=""
          test=""
          name=""

          if [[ "$model" == "bert" ]]; then
            continue
            # bert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi
            file="semcor.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_##REP##.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}.npz"
            train="${path}bert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}bert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[1]}}"
          else
            # sensebert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi

            file="semcor.data.xml_sensebert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}_##REP##.data.xml_sensebert-${complexity}-uncased_avg_False_layer_${layer}.npy_normTrue_K${dimension}_lda${lambda}.npz"
            train="${path}sensebert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}sensebert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[1]}}"
          fi
          i=$((i+1));
          python whitening.py --matrix "$train" --test "$test" --output_folder "$output_path" --method "$method";
      done
    done
  done
done

# dense

for complexity in ${complexities[*]}
  do
    for model in ${models[*]}
    do
          file=""
          layer=""
          train=""
          test=""
          name=""

          if [[ "$model" == "bert" ]]; then
            continue
            # bert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi
            file="##REP##.data.xml_bert-${complexity}-uncased_avg_False_layer_${layer}.npy"
            train="${path}bert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}bert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[1]}}"
          else
            # sensebert
            if [[ "$complexity" == "base" ]]; then
              layer="9-10-11-12"
            else
              layer="21-22-23-24"
            fi

            file="##REP##.data.xml_sensebert-${complexity}-uncased_avg_False_layer_${layer}.npy"
            train="${path}sensebert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[0]}}"
            test="${path}sensebert-${complexity}-uncased/${file/\#\#REP\#\#/${sm[1]}}"
          fi
          i=$((i+1));
          python whitening.py --matrix "$train" --test "$test" --output_folder "$output_path" --method "$method";
    done
  done

