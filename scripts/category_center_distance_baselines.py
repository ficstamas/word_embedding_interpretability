import os
import json
import sys
from argparse import ArgumentParser
import numpy as np
from interpretability.loader.semcor import read_with_path, Semcor
from sklearn.preprocessing import StandardScaler, Normalizer
import scipy.sparse as sp
import tqdm
import matplotlib.pyplot as plt


def calculate_score(train_weights: np.ndarray, test_weights, _semcor, output):
    # loading train and test words from semcor
    semcor, word_vector_labels, eval_vector_labels = _semcor

    # creating masks for the semantic categories
    mask = np.zeros([word_vector_labels.__len__()], dtype=np.int32)

    for i in tqdm.trange(mask.shape[0]):
        if word_vector_labels[i] == '<unknown>':
            mask[i] = -1
            continue
        lexname = word_vector_labels[i]
        lexname_id = semcor.c2i[lexname]
        mask[i] = lexname_id

    # calculating the category centers
    category_centers = None

    for i in tqdm.trange(semcor.c2i.__len__()):
        if semcor.i2c[i] == 'adj.ppl':
            continue
        indexes = np.where(mask == i)
        category_center = np.mean(train_weights[indexes, :][0], axis=0)
        # print(category_center.shape)
        if category_centers is None:
            category_centers = np.array([category_center])
        else:
            category_centers = np.concatenate([category_centers, [category_center]], axis=0)

    # calculating the distances
    print("Calculating dot product")
    # return
    category_centers = Normalizer('l2').transform(category_centers)
    category_distance = np.dot(test_weights, category_centers.T)

    p = os.path.join(output, "category_distance.npy")
    np.save(p, category_distance)

    # checking the labels
    true_labels = np.zeros([eval_vector_labels.__len__()], dtype=np.int64)
    for i in tqdm.trange(true_labels.shape[0]):
        if eval_vector_labels[i] == '<unknown>':
            true_labels[i] = -1
            continue
        true_labels[i] = semcor.c2i[eval_vector_labels[i]]

    mask = np.zeros(true_labels.shape[0], dtype=np.bool)

    mask[np.where(true_labels != -1)] = True
    true_labels_masked = true_labels[mask]
    predicted_labels_masked = np.argmax(category_distance[mask, :], axis=1)
    res = true_labels_masked[true_labels_masked == predicted_labels_masked].shape[0] / true_labels_masked.shape[0]
    return res


if __name__ == '__main__':
    parser = ArgumentParser(description='')

    # Embedding
    parser.add_argument("--train", type=str, required=True,
                        help="Training weights")
    parser.add_argument("--test", type=str, required=True,
                        help="Testing weights")
    parser.add_argument("--semcor_train", type=str, required=True,
                        help="semcor training data")
    parser.add_argument("--semcor_test", type=str, required=True,
                        help="semcor testing words")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path")

    args = parser.parse_args()

    if args.train.endswith(".npz"):
        train = sp.load_npz(args.train)
        train = train.toarray()
    else:
        train = np.load(args.train)

    if args.test.endswith(".npz"):
        test = sp.load_npz(args.test)
        test = test.toarray()
    else:
        test = np.load(args.test)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    sm, l, k = read_with_path(args.semcor_train, args.semcor_test)

    res = calculate_score(train, test, (sm, l, k), args.output)

    results = {
        "train": args.train,
        "test": args.test,
        "semcor_train": args.semcor_train,
        "semcor_test": args.semcor_test,
        "output": args.output,
        "result": res
    }
    # save
    fp = open(os.path.join(args.output, "category_distance.json"), mode='w', encoding='utf8')
    json.dump(results, fp, indent=4)
    fp.close()

