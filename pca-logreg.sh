l=(44 124 204 285 365 446 526 607 687);

dense_semcor=/data/ficstamas/representations/sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy.pca-44.npy;
dense_all=/data/ficstamas/representations/sensebert-base-uncased/ALL.sensebert-base-uncased.layer_9-10-11-12.npy.pca-44.npy;
sparse_semcor=/data/ficstamas/representations/sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05_semcor.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05.npz.pca-44.npy;
sparse_all=/data/ficstamas/representations/sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05_ALL.sensebert-base-uncased.layer_9-10-11-12.npy_normTrue_K1500_lda0.05.npz.pca-44.npy;
semcor=/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml;
all=/data/berend/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml;

for n in ${l[*]}
do
  p="/data/ficstamas/representations/sensebert-base-uncased/semcor.sensebert-base-uncased.layer_9-10-11-12.npy.pca-${n}.npy";
  p2="/data/ficstamas/representations/sensebert-base-uncased/ALL.sensebert-base-uncased.layer_9-10-11-12.npy.pca-${n}.npy";
  python -m scripts.logistic_regressor --semcor_train $semcor --semcor_test $all --output /data/ficstamas/workspace/sensebert_pca_logreg.txt --name "sensebert_dense_base_$n" --train_vectors $p --test_vectors $p2;
done
