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


class MCMCModel(DefaultModel):
    def __init__(self):
        super(MCMCModel, self).__init__()

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
        # Setting random seed
        np.random.seed(config.semantic_categories.seed)

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

        w2i_mem = SharedMemory(embedding.w2i_memory_name)
        w2i = embedding.buff_to_dict(w2i_mem, embedding.w2i_memory_size)

        semcat = config.semantic_categories.categories

        # Iterating over the dimensions of the embedding
        for i in tqdm.trange(w.shape[1], unit='dim', desc=f'PID -> {os.getpid()}\t'):
            if i % config.project.processes == modulus:
                dimension = w[:, i]
                # Perturbation
                estimation = MCMCModel.mcmc(dimension)
                # Iterating over the semantic categories
                for j in range(config.semantic_categories.categories.i2c.__len__()):
                    word_indexes = np.zeros(shape=[w.shape[0], ], dtype=np.bool)
                    # One-hot selection vector for in-category words
                    for word in semcat.vocab[semcat.i2c[j]]:
                        try:
                            word_indexes[w2i[word]] = True
                        except KeyError:
                            continue

                    # Populate P with category word weights
                    _p = dimension[word_indexes]
                    # Populate Q with out of category word weights
                    _q = dimension[~word_indexes]

                    samples = estimation.rvs(size=_p.shape[0]//3)

                    _p = np.concatenate([_p, np.array(samples)])
                    # calculating distance
                    distance, sign = config.distance.function(_p, _q, config)
                    distance_matrix[i, j, 0] = distance
                    distance_matrix[i, j, 1] = sign
        weights_mem.close()
        dist_mem.close()
        w2i_mem.close()

    @staticmethod
    def mcmc(target: np.ndarray, minimum_acceptance=200):
        """
        Metropolis–Hastings algorithm for Markov Chain Monte Carlo
        :param target: Discrete values of a distribution to approximate
        :param minimum_acceptance: The length of the chain (The minimum number of accepted std)
        :return: norm object from scipy (use .rvs to get random samples)
        """
        # Initial value
        target_std = np.std(target)
        target_mean = np.mean(target)
        x = [target_mean, target_std]
        accepted = []
        rejected = []
        bar = tqdm.tqdm(total=minimum_acceptance)
        while accepted.__len__() != minimum_acceptance:
            x_new = [x[0], np.random.normal(x[1], target_std)]
            x_likelihood = MCMCModel.log_likelihood_normal(x, target)
            x_new_likelihood = MCMCModel.log_likelihood_normal(x_new, target)
            if MCMCModel.accept(x_likelihood + np.log(MCMCModel.prior(x)),
                                x_new_likelihood + np.log(MCMCModel.prior(x_new))):
                x = x_new
                accepted.append(x_new)
                bar.update(1)
            else:
                rejected.append(x_new)

        accepted = np.array(accepted)
        ind = int(np.floor(accepted.shape[0]*0.25))
        sigma = np.mean(accepted[ind:, 1])

        return scipy.stats.norm(target_mean, sigma)

    @staticmethod
    def log_likelihood_normal(x, data):
        return np.sum(np.log(scipy.stats.norm(x[0], x[1]).pdf(data)))

    @staticmethod
    def accept(x, x_new):
        if x_new > x:
            return True
        else:
            constraint = np.random.uniform(0, 1)
            return np.exp(x_new-x) > constraint

    @staticmethod
    def prior(x):
        if x[1] <= 0:
            return 0
        else:
            return 1



