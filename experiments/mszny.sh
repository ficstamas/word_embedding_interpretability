#!/usr/bin/env bash

embeddings=("wiki.hu.vec" "wiki.hu.align.vec" "hu.szte.w2v.fasttext.vec")
path="/data/ficstamas/representations/fasttext/"
workspace="workspace/mszny"
methods=("hellinger" "bhattacharyya" "hellinger_normal" "bhattacharyya_normal")
vectors=200000
droprates=(0.1 0.2 0.3 0.4)
proc=40
cd ../

for embedding in ${embeddings[*]}
do
  for method in ${methods[*]}
  do
    for droprate in ${droprates[*]}
    do
      python interpretability_cli.py --embedding_path "${path}$embedding" -dense --lines_to_read "${vectors}" --smc_path 'data/semcat/semcat_en-de-hu.json' --smc_method 'random' --smc_rate "$droprate" --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "${embedding}-${method}-dr${droprate}" -save --accuracy "word_retrieval_test";
    done
    python interpretability_cli.py --embedding_path "${path}$embedding" -dense --lines_to_read "${vectors}" --smc_path 'data/semcat/semcat_en-de-hu.json' --smc_method 'category_center' --smc_rate 0.4 --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "${embedding}-${method}_interpret" -save -interpretability;
  done
done

#!/usr/bin/env bash
Ks=(1000 1500 2000)
lambdas=(0.05 0.1 0.2)

# hu.szte.w2v.fasttext.vec_K1000_l0.05.npz
for embedding in ${embeddings[*]}
do
  for K in ${Ks[*]}
  do
    for lambda in ${lambdas[*]}
    do
      for method in ${methods[*]}
      do
        for droprate in ${droprates[*]}
        do
          python interpretability_cli.py --embedding_path "${path}${embedding}_K${K}_l${lambda}_top200000_NormTrue.emb" --lines_to_read "${vectors}" --smc_path 'data/semcat/semcat_en-de-hu.json' --smc_method 'random' --smc_rate "$droprate" --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "${embedding}-${method}-K${K}-l${lambda}-dr${droprate}_sparse" -save --accuracy "word_retrieval_test";
        done
        python interpretability_cli.py --embedding_path "${path}${embedding}_K${K}_l${lambda}_top200000_NormTrue.emb" --lines_to_read "${vectors}" --smc_path 'data/semcat/semcat_en-de-hu.json' --smc_method 'category_center' --smc_rate 0.4 --seed 0 --processes $proc --distance "$method" --workspace $workspace --name "${embedding}-${method}-K${K}-l${lambda}_sparse_interpret" -save -interpretability;
      done
    done
  done
done