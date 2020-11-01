#!/usr/bin/env bash

embeddings=("cc.hu.300.vec" "wiki.hu.align.vec" "hu.szte.w2v.fasttext.vec")
Ks=(1000 1500 2000)
lambdas=(0.05 0.1 0.2)
path="/data/ficstamas/representations/fasttext/"

# shellcheck disable=SC2164
cd "/home/berend/interpretability_aaai2020/src/sparse_coding"

for embedding in ${embeddings[*]}
do
  for K in ${Ks[*]}
  do
    for lambda in ${lambdas[*]}
    do
      output_file="${embedding}_K${K}_l${lambda}.npz"
      python sparse_coding.py "${path}${embedding}" "${path}${output_file}" "$K" "$lambda" "DLSC"
    done
  done
done