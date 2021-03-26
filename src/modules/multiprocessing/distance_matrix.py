from src.modules.distance import DISTANCE_MAP
from multiprocessing import Queue, Manager
from queue import Empty
import tqdm
from src.modules.utilities.labels import Labels
from src.modules.load.numpy import Embedding
from src.modules.utilities.memory import construct_shared_memory_name
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import cpu_count, Pool, Process
import os
from src.modules.utilities.logging import Logger
import scipy.sparse as sp
import itertools


class Distance:
    def __init__(self, shape: tuple, distance: str, distance_params: dict, labels: Labels, embedding: Embedding):
        """
        Handles multiprocessing to calculate the Distance Matrix
        :param shape: Shape of Distance Matrix
        :param distance: Distance to use
        :param distance_params: Parameters for distance function
        :param labels: Labels object
        :param embedding: Embedding object
        """
        # Progress and Task Queues
        self._progress_manager = Manager()
        self._task_manager = Manager()
        self.task_queue = self._task_manager.Queue()
        self.progress_queue = self._progress_manager.Queue()
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.task_queue.put((i, j))

        # params
        self.distance_params = distance_params
        self.labels = labels
        self.embedding = embedding
        self.distance = distance
        self._memory_list = []
        self.log = Logger().logger

        # shared memory
        memory_name = construct_shared_memory_name("distance")
        distance_matrix = np.zeros(shape=shape, dtype=np.float)

        # Creating shared memory objects
        memory = SharedMemory(memory_name, create=True, size=distance_matrix.nbytes)
        # Creating numpy array from buffer
        buf = np.ndarray(distance_matrix.shape, dtype=distance_matrix.dtype, buffer=memory.buf)
        # Copying content
        buf[:, :] = distance_matrix[:, :]

        self.memory_info = {"name": memory_name, "size": distance_matrix.nbytes,
                            "shape": shape, "dtype": distance_matrix.dtype}
        self._memory_list.append(memory)

    def run(self, jobs=2):
        """
        Initiate process
        :param jobs: Number of processes to spawn/fork
        :return:
        """
        _jobs = min(jobs, cpu_count())
        self.log.info(f"Setting up multiprocessing task with {_jobs} jobs.")
        # preparing params for multiprocessing jobs
        params = [[self.task_queue, self.progress_queue, self.labels, self.embedding,
                   self.distance, self.memory_info, self.distance_params] for _ in range(_jobs)]

        # progress bar
        self.log.debug("Starting Multiprocessing Progress Bar")
        progress = Process(target=self._progress_bar, args=(self.progress_queue, self.task_queue.qsize()))
        progress.start()
        self.log.info("Starting Jobs...")
        with Pool(_jobs) as pool:
            pool.starmap(self._process, params)

        progress.join()
        self.log.debug("Joining Multiprocessing Progress Bar")

    @staticmethod
    def selection_map(fltr, label):
        return fltr in label

    @staticmethod
    def _process(task_queue: Queue, progress_queue: Queue, labels: Labels, embedding: Embedding, distance: str,
                 distance_matrix: dict, distance_params: dict):
        category_index_cache = {}
        while True:
            try:
                task = task_queue.get(True, 0.5)
            except Empty:
                break
            i = task[0]  # dimension index
            j = task[1]  # category index
            dimension = embedding.get_dimension("train", i)
            # Generating Semantic Category mask
            if j not in category_index_cache:
                word_indexes = list(itertools.starmap(Distance.selection_map, zip(itertools.repeat(labels.i2l[j]), labels.labels)))
                word_indexes = np.array(word_indexes, dtype=np.bool)
                category_index_cache[j] = np.where(word_indexes)
            else:
                word_indexes = np.zeros(dimension.shape[0], dtype=np.bool)
                word_indexes[category_index_cache[j]] = True
            # Populate P with category word weights
            _p = dimension[word_indexes]
            if type(_p) is sp.csc.csc_matrix:
                _p = np.reshape(_p.toarray(), (_p.shape[0]))  # np.reshape(_p, (157))
            # Populate Q with out of category word weights
            _q = dimension[~word_indexes]
            if type(_q) is sp.csc.csc_matrix:
                _q = np.reshape(_q.toarray(), (_q.shape[0]))
            # calculating distance
            if _p.shape[0] == 0:  # if for some reason there is no annotated entry
                distance_value, sign = 0, 1
            else:
                distance_value, sign = DISTANCE_MAP[distance](_p, _q, **distance_params)
            Distance.set(distance_matrix, i, j, (distance_value, sign))
            progress_queue.put(0)

    def save(self, path: str):
        """
        Saves the distance matrix
        :param path: Path
        :return:
        """
        info = self.memory_info
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        np.save(os.path.join(path, "model/distance_matrix.npy"), matrix)

    def free(self):
        """
        Free allocated memory spaces
        :return:
        """
        for mem in self._memory_list:
            mem.unlink()

    @staticmethod
    def set(memory_info: dict, i: int, j: int, value: tuple):
        """
        Set an entry of distance matrix
        :param memory_info: Dict which contains information about the Distance Matrix Shared Memory object
                            Required keys: name, dtype, shape
        :param i: Dimension
        :param j: Semantic Category
        :param value: Values in (value, sign) format
        :return:
        """
        info = memory_info
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        matrix[i, j, 0] = value[0]
        matrix[i, j, 1] = value[1]

    @staticmethod
    def _progress_bar(queue: Queue, total):
        """
        Process which handles the progress bar
        :param queue: Queue object
        :param total: size of the progress bar
        :return:
        """
        progress = tqdm.tqdm(total=total, unit='coeff', desc=f'[DistanceMatrix]Progress\t:')
        while True:
            try:
                _ = queue.get(True, 0.5)
                progress.update()
                if progress.n == total:
                    break
            except Empty:
                continue
