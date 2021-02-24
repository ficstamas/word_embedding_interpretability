from interpretability.core.config import Config
from interpretability.reader.semcor import Semcor
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from sklearn.preprocessing import StandardScaler, Normalizer
import tqdm
import os
import sys


def accuracy(eval_vector_labels: dict, config=None, relaxation=1, weight=None, preprocessed=False, wt=None,
             recalculate=False, border=None):
    if config is None:
        config = Config(access=True)

    distance_mem = SharedMemory(config.data.distance_matrix)
    distance_matrix = np.ndarray(shape=config.data.distance_matrix_shape,
                                 buffer=distance_mem.buf)

    distances = distance_matrix[:, :, 0]
    signs = distance_matrix[:, :, 1]

    try:
        config.data.load_test_word_weights()
    except NotImplementedError as e:
        config.logger.error("Try other file type for the test word weight file!\n {0}".format(e))
        sys.exit(1)

    # performing the usual operations
    config.logger.info("Performing L1 normalization...")

    # normalized = Normalizer('l1').transform(distances.T).T
    normalized = distances / np.linalg.norm(distances, 1, axis=0)

    config.logger.info("Performing sign correction...")

    sign_corrected = normalized * signs

    config.logger.info("Performing standard scaling...")

    if not preprocessed:
        scaled = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(config.data.test_word_weights)
    else:
        scaled = config.data.test_word_weights

    if weight is None:
        transformed_space = scaled.dot(sign_corrected)
    else:
        transformed_space = scaled.dot(sign_corrected) * weight

    if preprocessed:
        semcat = config.semantic_categories.categories

        freq = np.zeros(semcat.i2c.__len__())
        for key in wt:
            if wt[key] != '<unknown>':
                id = semcat.c2i[wt[key]]
                freq[id] = freq[id] + 1

        freq_normed = freq / np.linalg.norm(freq, 2)
        transformed_space = transformed_space + transformed_space * freq_normed

    config.logger.info("Transformed space calculated!")

    np.save(os.path.join(config.project.models, "validation_transformed_space.npy"), transformed_space)

    config.logger.info(f"Saved transformed space for validation ({config.data.test_word_weights_path})")

    all_results = {}

    if any([not os.path.exists(os.path.join(config.project.results, f"{config.distance.weight_name}accuracy-{ds}.npz")) for ds in [key for key in border] + ["ALL"]]) or \
            recalculate:
        config.logger.info("Calculating ordered accuracy matrix...")
        prev_end = 0
        for dataset in border:
            upper_bound = border[dataset]
            in_data = transformed_space[prev_end:upper_bound+1, :]
            test_labels, argmax_matrix = accuracy_matrix_calculation(in_data,eval_vector_labels, config, prev_end, dataset)
            prev_end = upper_bound + 1
            all_results[dataset] = [test_labels, argmax_matrix]
        test_labels, argmax_matrix = accuracy_matrix_calculation(transformed_space, eval_vector_labels, config, 0, "ALL")
        all_results["ALL"] = [test_labels, argmax_matrix]
    else:
        config.logger.info("Loading ordered accuracy matrix...")

        for dataset in border:
            arrs = np.load(os.path.join(config.project.results, f"{config.distance.weight_name}accuracy-{dataset}.npz"))
            test_labels = arrs['labels']
            argmax_matrix = arrs['values']
            all_results[dataset] = [test_labels, argmax_matrix]
        arrs = np.load(os.path.join(config.project.results, f"{config.distance.weight_name}accuracy-ALL.npz"))
        test_labels = arrs['labels']
        argmax_matrix = arrs['values']
        all_results["ALL"] = [test_labels, argmax_matrix]

    for dataset in all_results:
        params = {
            "test_labels": all_results[dataset][0],
            "argmax_matrix": all_results[dataset][1],
            "relaxation": relaxation,
            "config": config,
            "dataset": dataset
        }
        accuracy_score(**params)


def accuracy_score(test_labels, argmax_matrix, relaxation, config, dataset):
    score = []

    for i in range(argmax_matrix.shape[0]):
        appended = False
        for j in range(1, relaxation + 1):
            value = argmax_matrix[i, -j]
            if value == test_labels[i]:
                score.append(1 / j)
                appended = True
        if not appended:
            score.append(0)

    config.logger.info(f"Accuracy is {np.mean(score)}")
    fd = open(os.path.join(config.project.results, f"{config.distance.weight_name}accuracy-{dataset}.txt"),
              mode='a')
    fd.write(f"{np.mean(score)}@{relaxation}\n")
    fd.close()


def accuracy_matrix_calculation(transformed_space, eval_vector_labels, config, offset, dataset):
    test_weights = None
    test_labels = []
    argmax_matrix = None

    semcor: Semcor
    semcor = config.semantic_categories.categories

    j = 0
    for i in tqdm.trange(transformed_space.shape[0]):
        if eval_vector_labels[offset+i] == '<unknown>':
            continue
        if test_weights is None and eval_vector_labels[offset+i] != '<unknown>':
            sw = transformed_space[i, :]
            test_weights = sw[np.newaxis, :]
            test_labels.append(semcor.c2i[eval_vector_labels[offset+i]])
            j += 1
        else:
            if eval_vector_labels[offset+i] != '<unknown>':
                test_weights = np.concatenate([test_weights, transformed_space[np.newaxis, i, :]], axis=0)
                test_labels.append(semcor.c2i[eval_vector_labels[offset+i]])
                j += 1

    for i in tqdm.trange(test_weights.shape[0]):
        word_vector = np.array([test_weights[i, :].T])
        lex_indexes = np.array([np.arange(word_vector.shape[1])])
        pairs = np.append(word_vector, lex_indexes, axis=0)
        sorted_word_vector = pairs[:, pairs[0, :].argsort()]
        if argmax_matrix is None:
            argmax_matrix = sorted_word_vector[np.newaxis, 1]
        else:
            argmax_matrix = np.concatenate([argmax_matrix, sorted_word_vector[np.newaxis, 1]], axis=0)
    test_labels = np.array(test_labels)
    np.savez(os.path.join(config.project.results, f"{config.distance.weight_name}accuracy-{dataset}.npz"),
             labels=test_labels.astype(np.int8), values=argmax_matrix.astype(np.int8))
    return test_labels, argmax_matrix
