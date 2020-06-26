from .default import DefaultModel
from interpretability.core.config import Config
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import tqdm
import os
from sklearn.neighbors import KernelDensity
import scipy
import matplotlib.pyplot as plt
import seaborn as sns


class ContextualModel(DefaultModel):
    def __init__(self):
        super(ContextualModel, self).__init__()

    @staticmethod
    def _process(source: str, modulus: int, config: Config):
        """
        Calculating distance matrix
        :param modulus:
        :param config:
        :return:
        """
        # preparing access for shared memories
        embedding = config.embedding.embedding

        if source == "embedding":
            # Embedding to use (first pass)
            weights_mem = SharedMemory(embedding.embedding_memory_name)
            w = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                           buffer=weights_mem.buf)
            # Results will be saved here
            dist_mem = SharedMemory(config.data.distance_matrix)
            distance_matrix = np.ndarray(shape=config.data.distance_matrix_shape, buffer=dist_mem.buf)
        else:
            # Embedding to use (second pass)
            weights_mem = SharedMemory(config.data.transformed_space)
            w = np.ndarray(shape=config.data.transformed_space_shape, buffer=weights_mem.buf)
            # Results will be saved here
            dist_mem = SharedMemory(config.data.transformed_space_distance_matrix)
            distance_matrix = np.ndarray(shape=config.data.transformed_space_distance_matrix_shape, buffer=dist_mem.buf)

        i2w_mem = SharedMemory(embedding.i2w_memory_name)
        i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)

        semcat = config.semantic_categories.categories

        # Iterating over the dimensions of the embedding
        for i in tqdm.trange(w.shape[1], unit='dim', desc=f'PID -> {os.getpid()}\t'):
            if i % config.project.processes == modulus:
                dimension = w[:, i]
                # Iterating over the semantic categories
                for j in range(config.semantic_categories.categories.i2c.__len__()):
                    word_indexes = np.zeros(shape=[w.shape[0], ], dtype=np.bool)
                    # One-hot selection vector for in-category words
                    for k in range(w.shape[0]):
                        try:
                            if i2w[str(k)] == semcat.i2c[j]:
                                word_indexes[k] = True
                        except KeyError:
                            continue
                        except IndexError:
                            continue

                    # Populate P with category word weights
                    _p = dimension[word_indexes]
                    # Populate Q with out of category word weights
                    _q = dimension[~word_indexes]
                    # calculating distance
                    distance, sign = config.distance.function(_p, _q, config)
                    distance_matrix[i, j, 0] = distance
                    distance_matrix[i, j, 1] = sign
        weights_mem.close()
        dist_mem.close()
        w2i_mem.close()