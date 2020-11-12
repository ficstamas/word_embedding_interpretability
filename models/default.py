from .abstract import Model
from interpretability.core.config import Config
import numpy as np
from multiprocessing import cpu_count, Pool, Process
from multiprocessing.shared_memory import SharedMemory
import tqdm
import os
from sklearn.preprocessing import StandardScaler, Normalizer
from multiprocessing import Queue, Manager
from queue import Empty


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

        progress_manager = Manager()

        task_manager = Manager()
        task_queue = task_manager.Queue()
        progress_queue = progress_manager.Queue()

        for i in range(self.config.embedding.embedding.embedding_memory_shape[1]):
            for j in range(self.config.semantic_categories.categories.i2c.__len__()):
                task_queue.put((i, j))

        # preparing params for multiprocessing jobs
        params = [["embedding", task_queue, self.config, progress_queue] for _ in range(self.config.project.processes)]

        progress = Process(target=self._progress_bar, args=(progress_queue, task_queue.qsize()))
        progress.start()
        # run mp
        with Pool(min(self.config.project.processes, cores)) as pool:
            pool.starmap(self._process, params)

        progress.join()

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

        task_manager = Manager()
        task_queue = task_manager.Queue()
        progress_queue = progress_manager.Queue()

        for i in range(self.config.semantic_categories.categories.i2c.__len__()):
            for j in range(self.config.semantic_categories.categories.i2c.__len__()):
                task_queue.put((i, j))

        # Getting the distance space of the transformed space
        # preparing params for multiprocessing jobs
        params = [["transformed_space", task_queue, self.config, progress_queue] for _ in range(self.config.project.processes)]

        progress = Process(target=self._progress_bar, args=(progress_queue, task_queue.qsize()))
        progress.start()

        # run mp
        with Pool(min(self.config.project.processes, cores)) as pool:
            pool.starmap(self._process, params)

        progress.join()

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
    def _process(source: str, task_queue: Queue, config: Config, progress_queue: Queue):
        """
        Calculating distance matrix
        :param task_queue:
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

        while True:
            try:
                task = task_queue.get(True, 0.5)
            except Empty:
                # config.logger.info(f"Task Queue is empty")
                break
            # Iterating over the dimensions of the embedding
            # for i in tqdm.trange(w.shape[1], unit='dim', desc=f'PID -> {os.getpid()}\t'):
            i = task[0]
            j = task[1]
            dimension = w[:, i]
            word_indexes = np.zeros(shape=[w.shape[0], ], dtype=np.bool)
            # One-hot selection vector for in-category words
            for word in semcat.vocab[semcat.i2c[j]]:
                try:
                    word_indexes[w2i[word]] = True
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
            progress_queue.put(0)
            # config.logger.info(f"Tasks left: {task_queue.qsize()}")
        weights_mem.close()
        dist_mem.close()
        w2i_mem.close()

    @staticmethod
    def _progress_bar(queue: Queue, total):
        progress = tqdm.tqdm(total=total, unit='dim', desc=f'Progress\t')
        while True:
            try:
                _ = queue.get(True, 0.5)
                progress.n += 1
                progress.update(0)
                if progress.n == total:
                    break
            except Empty:
                continue

    def relative_frequency(self):
        pass

