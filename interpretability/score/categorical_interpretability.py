import numpy as np
from interpretability.loader.embedding import Embedding
from interpretability.loader.semcat import SemCat
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from interpretability.core.config import Config


def V_p(V, n_j, lamb, config: Config):
    embedding = config.embedding.embedding
    # i2w
    i2w_mem = SharedMemory(embedding.i2w_memory_name)
    i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)
    return set([i2w[int(o)] for o in V[1, -lamb * n_j:]])


def V_n(V, n_j, lamb, config: Config):
    embedding = config.embedding.embedding
    # i2w
    i2w_mem = SharedMemory(embedding.i2w_memory_name)
    i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)
    return set([i2w[int(o)] for o in V[1, :lamb * n_j]])


def is_p(i, j, config: Config, lamb):
    # Embedding
    embedding = config.embedding.embedding
    weights_mem = SharedMemory(embedding.embedding_memory_name)
    w = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                   buffer=weights_mem.buf)

    V_1 = np.array([w[:, j]])
    V_2 = np.array([np.arange(V_1.shape[1])])
    V = np.append(V_1, V_2, axis=0)
    V_sorted = V[:, V[0, :].argsort()]

    S = set(config.semantic_categories.categories.vocab[config.semantic_categories.categories.i2c[i]])
    n_i = config.semantic_categories.categories.vocab[config.semantic_categories.categories.i2c[i]].__len__()

    v_p = V_p(V_sorted, n_i, lamb, config)
    v_n = V_n(V_sorted, n_i, lamb, config)

    IS_p = S.intersection(v_p).__len__() / n_i * 100
    IS_n = S.intersection(v_n).__len__() / n_i * 100

    return max(IS_p, IS_n)


def j_star(i: int, distance_matrix: np.ndarray):
    return int(np.argmax(distance_matrix[:, i]).astype(dtype=np.int))


def is_i(i: int, config: Config, distance_matrix: np.ndarray, lamb):
    IS_ji = []
    D = distance_matrix.shape[0]

    # Then we go through the embedding dimensions
    for j in range(D):
        IS_ji.append(is_p(i, j, config, lamb))

    # picking the max by W_b max->i
    return IS_ji[j_star(i, distance_matrix)]


def score_dist(config: Config, distance_matrix: np.ndarray, lamb):
    IS_i = []

    # seq is the concept dimension indexes
    for i in range(config.semantic_categories.categories.i2c.__len__()):
        IS_i.append(is_i(i, config, distance_matrix, lamb))
    return IS_i


def score(config: Config, distance_matrix: np.ndarray, proc=5, lamb=5, avg=False):
    IS_i = []

    pool = multiprocessing.Pool(processes=proc)

    inputs = []
    for i in range(lamb):
        inputs.append([config, distance_matrix, i+1])

    with pool as p:
        result = p.starmap(score_dist, inputs)

    for res in result:
        IS_i += res

    if avg:
        return sum(IS_i)/C
    else:
        return IS_i
