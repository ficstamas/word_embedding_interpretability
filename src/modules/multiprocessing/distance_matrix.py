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
import copy
import os
from src.modules.utilities.logging import Logger


class Distance:
    def __init__(self, shape: tuple, distance: str, distance_params: dict, labels: Labels, embedding: Embedding):
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
    def _process(task_queue: Queue, progress_queue: Queue, labels: Labels, embedding: Embedding, distance: str,
                 distance_matrix: dict, distance_params: dict):
        while True:
            try:
                task = task_queue.get(True, 0.5)
            except Empty:
                break
            i = task[0]  # dimension index
            j = task[1]  # category index
            dimension = embedding.get_dimension("train", i)
            word_indexes = np.zeros(shape=[dimension.shape[0], ], dtype=np.bool)
            # One-hot selection vector for in-category words
            for k, label in enumerate(labels.labels):
                if label.__len__() == 0:
                    continue
                if labels.i2l[j] in label:
                    word_indexes[k] = True
            # Populate P with category word weights
            _p = dimension[word_indexes]
            # Populate Q with out of category word weights
            _q = dimension[~word_indexes]
            # calculating distance
            distance_value, sign = DISTANCE_MAP[distance](_p, _q, **distance_params)
            Distance.set(distance_matrix, i, j, (distance_value, sign))
            progress_queue.put(0)

    def get(self):
        info = self.memory_info
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        return copy.copy(matrix)

    def save(self, path: str):
        info = self.memory_info
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        np.save(os.path.join(path, "model/distance_matrix.npy"), matrix)

    def free(self):
        for mem in self._memory_list:
            mem.unlink()

    @staticmethod
    def set(memory_info: dict, i: int, j: int, value: tuple):
        info = memory_info
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        matrix[i, j, 0] = value[0]
        matrix[i, j, 1] = value[1]

    @staticmethod
    def _progress_bar(queue: Queue, total):
        progress = tqdm.tqdm(total=total, unit='coeff', desc=f'Progress\t')
        while True:
            try:
                _ = queue.get(True, 0.5)
                progress.n += 1
                progress.update(0)
                if progress.n == total:
                    break
            except Empty:
                continue
