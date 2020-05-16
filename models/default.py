from .abstract import Model
from interpretability.core.config import Config
import numpy as np
from multiprocessing import cpu_count, Pool
from multiprocessing.shared_memory import SharedMemory
import tqdm
import os
from sklearn.preprocessing import StandardScaler, Normalizer


class DefaultModel(Model):
    def __init__(self):
        self.config = Config()
        # shapes of matrices which will store some kind of result at one point during the runtime
        self.dist_shape = (self.config.embedding.embedding.embedding_memory_shape[1],
                           self.config.semantic_categories.categories.i2c.__len__(), 2)

        self.trans_shape = (self.config.embedding.embedding.embedding_memory_shape[0],
                            self.dist_shape[1])

        self.trans_dist_shape = (self.trans_shape[1], self.trans_shape[1], 2)

        self.config.data.distance_matrix_shape = self.dist_shape
        self.config.data.transformed_space_shape = self.trans_shape
        self.config.data.transformed_space_distance_matrix_shape = self.trans_dist_shape
        self._memory_initiated = False

    def run(self):

        if not self._memory_initiated:
            # initing shared memories for results
            self.config.data.init_shared_memory(self.config.data.distance_matrix, self.dist_shape, np.float,
                                                self.config)
            self.config.data.init_shared_memory(self.config.data.transformed_space, self.trans_shape, np.float,
                                                self.config)
            self.config.data.init_shared_memory(self.config.data.transformed_space_distance_matrix,
                                                self.trans_dist_shape,
                                                np.float, self.config)
            self._memory_initiated = True

        cores = cpu_count()

        # preparing params for multiprocessing jobs
        params = [["embedding", i, self.config] for i in range(self.config.project.processes)]

        # run mp
        with Pool(min(self.config.project.processes, cores)) as pool:
            pool.starmap(DefaultModel._process, params)

        self.config.logger.info(f"Distance matrix calculation is done!")

        dist_mem = SharedMemory(self.config.data.distance_matrix)
        distance_matrix = np.ndarray(shape=self.config.data.distance_matrix_shape, buffer=dist_mem.buf)

        distances = distance_matrix[:, :, 0]
        signs = distance_matrix[:, :, 1]

        self.config.logger.info("Performing L1 normalization...")

        normalized = Normalizer('l1').transform(distances.T).T

        self.config.logger.info("Performing sign correction...")

        sign_corrected = normalized*signs

        self.config.logger.info("Performing standard scaling...")

        embedding = self.config.embedding.embedding
        weights_mem = SharedMemory(embedding.embedding_memory_name)
        w = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                       buffer=weights_mem.buf)

        scaled = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(w)

        transformed_mem = SharedMemory(self.config.data.transformed_space)
        transformed_space = np.ndarray(shape=self.config.data.transformed_space_shape, buffer=transformed_mem.buf)
        transformed_space[:, :] = scaled.dot(sign_corrected)[:, :]

        self.config.logger.info("Transformed space calculated!")

        self.config.logger.info("Calculating the distance space of the transformed space...")
        # Getting the distance space of the transformed space
        # preparing params for multiprocessing jobs
        params = [["transformed_space", i, self.config] for i in range(self.config.project.processes)]

        # run mp
        with Pool(min(self.config.project.processes, cores)) as pool:
            pool.starmap(DefaultModel._process, params)

        self.config.logger.info("Done!")

    def save(self):
        """
        Saving distance matrix, transformed space and distance matrix of transformed space
        :return:
        """
        transformed_mem = SharedMemory(self.config.data.transformed_space)
        transformed_space = np.ndarray(shape=self.config.data.transformed_space_shape, buffer=transformed_mem.buf)

        dist_mem = SharedMemory(self.config.data.distance_matrix)
        distance_matrix = np.ndarray(shape=self.config.data.distance_matrix_shape, buffer=dist_mem.buf)

        transformed_space_distance_mem = SharedMemory(self.config.data.transformed_space_distance_matrix)
        transformed_space_distance_matrix = np.ndarray(shape=self.config.data.transformed_space_distance_matrix_shape,
                                                       buffer=transformed_space_distance_mem.buf)

        np.save(os.path.join(self.config.project.models, "distance_matrix.npy"), distance_matrix)
        self.config.logger.info(f"Distance matrix is saved at "
                                f"{os.path.join(self.config.project.models, 'distance_matrix.npy')}")

        np.save(os.path.join(self.config.project.models, "transformed_space.npy"), transformed_space)
        self.config.logger.info(f"Transformed space is saved at "
                                f"{os.path.join(self.config.project.models, 'transformed_space.npy')}")

        np.save(os.path.join(self.config.project.models, "transformed_space_distance_matrix.npy"),
                transformed_space_distance_matrix)
        self.config.logger.info(f"Distance matrix of transformed space is saved at "
                                f"{os.path.join(self.config.project.models, 'transformed_space_distance_matrix.npy')}")

    def load(self):
        if not self._memory_initiated:
            # initing shared memories for results
            self.config.data.init_shared_memory(self.config.data.distance_matrix, self.dist_shape, np.float,
                                                self.config)
            self.config.data.init_shared_memory(self.config.data.transformed_space, self.trans_shape, np.float,
                                                self.config)
            self.config.data.init_shared_memory(self.config.data.transformed_space_distance_matrix,
                                                self.trans_dist_shape,
                                                np.float, self.config)
            self._memory_initiated = True

        transformed_mem = SharedMemory(self.config.data.transformed_space)
        transformed_space = np.ndarray(shape=self.config.data.transformed_space_shape, buffer=transformed_mem.buf)

        dist_mem = SharedMemory(self.config.data.distance_matrix)
        distance_matrix = np.ndarray(shape=self.config.data.distance_matrix_shape, buffer=dist_mem.buf)

        transformed_space_distance_mem = SharedMemory(self.config.data.transformed_space_distance_matrix)
        transformed_space_distance_matrix = np.ndarray(shape=self.config.data.transformed_space_distance_matrix_shape,
                                                       buffer=transformed_space_distance_mem.buf)

        distance_matrix[:, :, :] = np.load(os.path.join(self.config.project.models, "distance_matrix.npy"))[:, :, :]
        self.config.logger.info(f"Distance matrix is loaded from "
                                f"{os.path.join(self.config.project.models, 'distance_matrix.npy')}")

        transformed_space[:, :] = np.load(os.path.join(self.config.project.models, "transformed_space.npy"))[:, :]
        self.config.logger.info(f"Transformed space is loaded from "
                                f"{os.path.join(self.config.project.models, 'transformed_space.npy')}")

        transformed_space_distance_matrix[:, :, :] = np.load(
            os.path.join(self.config.project.models,
                         "transformed_space_distance_matrix.npy")
        )[:, :, :]
        self.config.logger.info(f"Distance matrix of transformed space is loaded from "
                                f"{os.path.join(self.config.project.models, 'transformed_space_distance_matrix.npy')}")

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

        w2i_mem = SharedMemory(embedding.w2i_memory_name)
        w2i = embedding.buff_to_dict(w2i_mem, embedding.w2i_memory_size)

        semcat = config.semantic_categories.categories

        # Iterating over the dimensions of the embedding
        for i in tqdm.trange(w.shape[1], unit='dim', desc=f'PID -> {os.getpid()}\t'):
            if i % config.project.processes == modulus:
                dimension = w[:, i]
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
                    # calculating distance
                    distance, sign = config.distance.function(_p, _q, config)
                    distance_matrix[i, j, 0] = distance
                    distance_matrix[i, j, 1] = sign
        weights_mem.close()
        dist_mem.close()
        w2i_mem.close()

