from interpretability.core.config import Config
import numpy as np
import os
import queue
import copy
from sklearn.preprocessing import StandardScaler, Normalizer
import tqdm
import math
from multiprocessing.shared_memory import SharedMemory


def accuracy(config: Config):
    """
    Calculates the accuracy of the transformation
    :param config:
    :return:
    """
    # Loading memories
    # Embedding
    embedding = config.embedding.embedding
    weights_mem = SharedMemory(embedding.embedding_memory_name)
    w = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                   buffer=weights_mem.buf)
    # Distance space
    dist_mem = SharedMemory(config.data.distance_matrix)
    distance_matrix = np.ndarray(shape=config.data.distance_matrix_shape, buffer=dist_mem.buf)
    # i2w
    i2w_mem = SharedMemory(embedding.i2w_memory_name)
    i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)
    # semcat
    semcat = config.semantic_categories.categories

    val_size = [1, 3, 5]

    # Transform
    embedding_std = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(w)

    values = {}

    # calculating k-s for sparsing coefficients
    calc_k = lambda k, i: [math.floor(k // 2 + (k - k // 2) * 2 / 3), k] if i == 1 else calc_k(k // 2, i - 1) + [math.floor(k // 2 + (k - k // 2) * 2 / 3), k]

    _k = calc_k(w.shape[1], 5)

    for validation in tqdm.tqdm(val_size):
        k_vector = []
        # Iterating over possible K values
        for k in _k:
            sparse_w_b = copy.deepcopy(distance_matrix[:, :, 0])
            # Select top K values
            for i in range(distance_matrix.shape[0]):
                q = queue.PriorityQueue()
                for j in range(distance_matrix.shape[1]):
                    q.put(distance_matrix[i, j, 0])
                for j in range(distance_matrix.shape[1] - k):
                    q.get()
                sparse_w_b[i, sparse_w_b[i] < q.get()] = 0

            w_nb = Normalizer('l1').transform(sparse_w_b).T

            # Sign correction
            w_nsb = w_nb * distance_matrix[:, :, 1]

            # Sparse I matrix
            sparse_i = embedding_std.dot(w_nsb)
            # Calculating summed means to categories
            averages = []
            weights = []
            for cat in range(semcat.i2c.__len__()):
                # number of test words
                n = semcat.dropped_words[semcat.i2c[cat]].__len__()
                dim = sparse_i[:, cat]
                # indexes to know which weight represented which word after sorting
                indexes = np.array([np.arange(sparse_i.shape[0])])
                # appending the indexes to the weights
                i_dim = np.append([dim], indexes, axis=0).T
                # sorting it
                dim_sorted = i_dim[i_dim[:, 0].argsort()]
                # getting the top n weight
                top_n = dim_sorted[-(n * validation):]
                bot_n = dim_sorted[:n * validation]
                # getting the associated words
                words_top = set([i2w[int(ind)] for ind in top_n[:, 1]])
                words_bottom = set([i2w[int(ind)] for ind in bot_n[:, 1]])
                # test words
                cat_words = set(semcat.dropped_words[semcat.i2c[cat]])
                # intersection with the category
                com_words_top = words_top.intersection(cat_words)
                com_words_bot = words_bottom.intersection(cat_words)
                # percent * weight
                percentage = max([com_words_top.__len__(), com_words_bot.__len__()]) / cat_words.__len__() * n * 100
                # saving value
                averages.append(percentage)
                # saving weight
                weights.append(n)
            # getting the weighted sum
            weighted_average = sum(averages) / sum(weights)
            # saving K related accuracy
            k_vector.append(weighted_average)
        # saving method related accuracy
        values[str(validation)] = k_vector

    opath = os.path.join(os.getcwd(), config.project.results, "accuracy.txt")
    f = open(opath, mode="w", encoding="utf8")
    for key in values:
        f.write(f"# Validation size: {key}\n")
        i = 0
        for val in values[key]:
            f.write(f"{_k[i]} {val}\n")
            i += 1
        f.write("\n")
    f.close()