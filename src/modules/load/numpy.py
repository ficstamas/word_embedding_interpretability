import numpy as np
from scipy.sparse import load_npz
import os
from src.modules.utilities.logging import Logger
from src.modules.utilities.memory import construct_shared_memory_name
from multiprocessing.shared_memory import SharedMemory
import copy


class Embedding:
    def __init__(self):
        self._memory_list = []
        self.memory_info = {"train": {"name": None, "size": None, "shape": (), "dtype": None},
                            "test": {"name": None, "size": None, "shape": (), "dtype": None}}
        self.logger = Logger().logger

    def load(self, train_path: str, test_path=None):
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Path not exists: {train_path}")
        if test_path is not None and not os.path.exists(test_path):
            raise FileNotFoundError(f"Path not exists: {test_path}")
        T, E = None, None
        if train_path.endswith(".npy"):
            T = np.load(train_path)
        elif train_path.endswith(".npz"):
            T = load_npz(train_path).toarray()
        else:
            raise FileExistsError("Unsupported file format!")

        self._allocate_memory(T, "train")
        del T

        if test_path is None:
            return
        if test_path.endswith(".npy"):
            E = np.load(test_path)
        elif test_path.endswith(".npz"):
            E = load_npz(test_path).toarray()
        else:
            raise FileExistsError("Unsupported file format!")
        self._allocate_memory(E, "test")
        del E

    def _allocate_memory(self, matrix: np.ndarray, name: str):
        # Constructing memory name to avoid collision
        memory_name = construct_shared_memory_name(name)
        self.memory_info[name]["name"] = memory_name
        self.memory_info[name]["size"] = matrix.nbytes
        self.memory_info[name]["shape"] = matrix.shape
        self.memory_info[name]["dtype"] = matrix.dtype
        # Creating shared memory objects
        memory = SharedMemory(memory_name, create=True, size=matrix.nbytes)
        # Creating numpy array from buffer
        buf = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=memory.buf)
        # Copying content
        buf[:, :] = matrix[:, :]
        self._memory_list.append(memory)
        self.logger.info(f"Embedding ({name}) in memory: {matrix.nbytes / 1024 / 1024:.2f} Mbytes "
                         f"({matrix.nbytes} bytes)")

    def get(self, name: str):
        info = self.memory_info[name]
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        return copy.copy(matrix)

    def get_dimension(self, name: str, d: int):
        info = self.memory_info[name]
        memory = SharedMemory(info["name"])
        matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        return copy.copy(matrix[:, d])

    def free(self):
        for mem in self._memory_list:
            mem.unlink()
