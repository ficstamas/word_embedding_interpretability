import numpy as np
from scipy.sparse import load_npz
import scipy.sparse as sp
import os
from src.modules.utilities.logging import Logger
from src.modules.utilities.memory import construct_shared_memory_name
from multiprocessing.shared_memory import SharedMemory
from src.modules.transform.pipeline import Pipeline
import copy
import functools


class Embedding:
    """
    Handles the loading of embedding spaces from Numpy arrays
    """
    def __init__(self, path: str):
        self.path = path
        self._memory_list = []
        self.memory_info = {"train": {"name": None, "size": None, "shape": (), "dtype": None, "sparse": False,
                                      "sparse_data": {
                                          "data": {"dtype": None, "shape": None, "size": None},
                                          "indices": {"dtype": None, "shape": None, "size": None},
                                          "indptr": {"dtype": None, "shape": None, "size": None}
                                      }},
                            "test": {"name": None, "size": None, "shape": (), "dtype": None, "sparse": False,
                                     "sparse_data": {
                                          "data": {"dtype": None, "shape": None, "size": None},
                                          "indices": {"dtype": None, "shape": None, "size": None},
                                          "indptr": {"dtype": None, "shape": None, "size": None}
                                      }}}
        self.logger = Logger().logger
        # Only set when 'keep_in_memory' is true
        self.train = None
        self.test = None

    def load(self, train_path: str, test_path: str, keep_in_memory=False, transform=True):
        """
        Loads embedding spaces
        :param train_path: Path to embedding space which represents the train set
        :param test_path: Path to embedding space which represents the test set
        :param keep_in_memory: If False then the representations are moved to the SharedMemory, otherwise it can
        be found in `self.train` or `self.test`
        :param transform: Whether to apply the transformation from `<project_path>/transforms/` folder
        :return:
        """
        self.load_train(train_path, keep_in_memory, transform)
        self.load_test(test_path, keep_in_memory, transform)

    def load_train(self, train_path: str, keep_in_memory=False, transform=True):
        """
        Loads Train set only
        :param train_path: Path to embedding space which represents the train set
        :param keep_in_memory: If False then the representations are moved to the SharedMemory, otherwise it can
        be found in `self.train` or `self.test`
        :param transform: Whether to apply the transformation from `<project_path>/transforms/` folder
        :return:
        """
        sparse = False
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Path not exists: {train_path}")
        if train_path.endswith(".npy"):
            T = np.load(train_path)
        elif train_path.endswith(".npz"):
            T = load_npz(train_path).tocsc()
            sparse = True
        else:
            raise FileExistsError("Unsupported file format!")

        log = Logger().logger
        config_path = os.path.join(self.path, "transforms/config.json")
        if transform and os.path.exists(config_path):
            pl = Pipeline(config_path)
            T = pl.apply(T.toarray() if sparse else T, self.path)
        else:
            log.info(f"No transformation was applied on the embedding. (config_found: {os.path.exists(config_path)},"
                     f" transformation_requested: {transform})")

        if not keep_in_memory:
            if type(T) is sp.csc.csc_matrix:
                self._allocate_memory_sparse(T, "train")
            elif type(T) is np.ndarray:
                self._allocate_memory(T, "train")
            else:
                raise TypeError(f"Encountered unsupported matrix format: {type(T)}")
            del T
        else:
            self.train = T.toarray() if sparse else T

    def load_test(self, test_path: str, keep_in_memory=False, transform=True):
        """
        Loads Test set only
        :param test_path: Path to embedding space which represents the train set
        :param keep_in_memory: If False then the representations are moved to the SharedMemory, otherwise it can
        be found in `self.train` or `self.test`
        :param transform: Whether to apply the transformation from `<project_path>/transforms/` folder
        :return:
        """
        sparse = False
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Path not exists: {test_path}")
        if test_path.endswith(".npy"):
            E = np.load(test_path)
        elif test_path.endswith(".npz"):
            E = load_npz(test_path).tocsc()
            sparse = True
        else:
            raise FileExistsError("Unsupported file format!")

        log = Logger().logger
        config_path = os.path.join(self.path, "transforms/config.json")
        if transform and os.path.exists(config_path):
            pl = Pipeline(config_path)
            E = pl.apply(E.toarray() if sparse else E, self.path)
        else:
            log.info(f"No transformation was applied on the embedding. (config_found: {os.path.exists(config_path)},"
                     f" transformation_requested: {transform})")

        if not keep_in_memory:
            if type(E) is sp.csc.csc_matrix:
                self._allocate_memory_sparse(E, "test")
            elif type(E) is np.ndarray:
                self._allocate_memory(E, "test")
            else:
                raise TypeError(f"Encountered unsupported matrix format: {type(T)}")
            del E
        else:
            self.test = E.toarray() if sparse else E

    def _allocate_memory(self, matrix: np.ndarray, name: str):
        """
        Allocates embeddings in SharedMemory
        :param matrix: Dense ndarray type matrix
        :param name: Name of the matrix (e.g. train, test...)
        :return:
        """
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

    def _allocate_memory_sparse(self, matrix: sp.csc_matrix, name: str):
        """
        Allocates sparse embeddings in SharedMemory
        :param matrix: Sparse csc type matrix
        :param name: Name of the matrix (e.g. train, test...)
        :return:
        """
        # Constructing memory name to avoid collision
        memory_name = construct_shared_memory_name(name)
        self.memory_info[name]["name"] = memory_name
        self.memory_info[name]["shape"] = matrix.shape
        self.memory_info[name]["sparse"] = True
        # Creating shared memory objects
        memory_data = SharedMemory(memory_name + "_data", create=True, size=matrix.data.nbytes)

        self.memory_info[name]["sparse_data"]["data"] = {
            "dtype": matrix.data.dtype,
            "shape": matrix.data.shape,
            "size": matrix.data.nbytes
        }

        memory_indices = SharedMemory(memory_name + "_indices", create=True, size=matrix.indices.nbytes)

        self.memory_info[name]["sparse_data"]["indices"] = {
            "dtype": matrix.indices.dtype,
            "shape": matrix.indices.shape,
            "size": matrix.indices.nbytes
        }

        memory_indptr = SharedMemory(memory_name + "_indptr", create=True, size=matrix.indptr.nbytes)

        self.memory_info[name]["sparse_data"]["indptr"] = {
            "dtype": matrix.indptr.dtype,
            "shape": matrix.indptr.shape,
            "size": matrix.indptr.nbytes
        }

        # Creating numpy array from buffer
        buf_data = np.ndarray(matrix.data.shape, dtype=matrix.data.dtype, buffer=memory_data.buf)
        buf_indices = np.ndarray(matrix.indices.shape, dtype=matrix.indices.dtype, buffer=memory_indices.buf)
        buf_indptr = np.ndarray(matrix.indptr.shape, dtype=matrix.indptr.dtype, buffer=memory_indptr.buf)

        # Copying content
        buf_data[:] = matrix.data[:]
        buf_indices[:] = matrix.indices[:]
        buf_indptr[:] = matrix.indptr[:]

        self._memory_list.append(memory_data)
        self._memory_list.append(memory_indices)
        self._memory_list.append(memory_indptr)

        self.logger.info("Allocating sparse matrix in shared memory")
        self.logger.info(f"Embedding ({name+'_data'}) in memory: {matrix.data.nbytes / 1024 / 1024:.2f} Mbytes "
                         f"({matrix.data.nbytes} bytes)")
        self.logger.info(f"Embedding ({name + '_indices'}) in memory: {matrix.indices.nbytes / 1024 / 1024:.2f} Mbytes "
                         f"({matrix.indices.nbytes} bytes)")
        self.logger.info(f"Embedding ({name + '_indptr'}) in memory: {matrix.indptr.nbytes / 1024 / 1024:.2f} Mbytes "
                         f"({matrix.indptr.nbytes} bytes)")
        overall_nbytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        self.logger.info(f"Embedding overall in memory: {overall_nbytes / 1024 / 1024:.2f} Mbytes "
                         f"({overall_nbytes} bytes)")

    def get(self, name: str) -> np.ndarray:
        """
        Returns the copy of the embedding space from SharedMemory
        :param name: `train` or `test`
        :return:
        """
        info = self.memory_info[name]
        if not info["sparse"]:
            memory = SharedMemory(info["name"])
            matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        else:
            memory_data = SharedMemory(str(info["name"]) + "_data")
            memory_indices = SharedMemory(str(info["name"]) + "_indices")
            memory_indptr = SharedMemory(str(info["name"]) + "_indptr")

            info_sparse = info["sparse_data"]

            matrix_data = np.ndarray(shape=info_sparse["data"]["shape"], dtype=info_sparse["data"]["dtype"],
                                     buffer=memory_data.buf)
            matrix_indices = np.ndarray(shape=info_sparse["indices"]["shape"], dtype=info_sparse["indices"]["dtype"],
                                        buffer=memory_indices.buf)
            matrix_indptr = np.ndarray(shape=info_sparse["indptr"]["shape"], dtype=info_sparse["indptr"]["dtype"],
                                       buffer=memory_indptr.buf)

            matrix = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), shape=info["shape"])

        return copy.copy(matrix)

    @functools.lru_cache(maxsize=128)
    def get_dimension(self, name: str, d: int) -> np.ndarray:
        """
        Returns the copy of a dimension of the embedding space from SharedMemory
        :param name: `train` or `test`
        :param d: Dimension
        :return:
        """
        info = self.memory_info[name]
        if not info["sparse"]:
            memory = SharedMemory(info["name"])
            matrix = np.ndarray(shape=info["shape"], dtype=info["dtype"], buffer=memory.buf)
        else:
            memory_data = SharedMemory(str(info["name"]) + "_data")
            memory_indices = SharedMemory(str(info["name"]) + "_indices")
            memory_indptr = SharedMemory(str(info["name"]) + "_indptr")

            info_sparse = info["sparse_data"]

            matrix_data = np.ndarray(shape=info_sparse["data"]["shape"], dtype=info_sparse["data"]["dtype"],
                                     buffer=memory_data.buf)
            matrix_indices = np.ndarray(shape=info_sparse["indices"]["shape"], dtype=info_sparse["indices"]["dtype"],
                                        buffer=memory_indices.buf)
            matrix_indptr = np.ndarray(shape=info_sparse["indptr"]["shape"], dtype=info_sparse["indptr"]["dtype"],
                                       buffer=memory_indptr.buf)

            matrix = sp.csc_matrix((matrix_data, matrix_indices, matrix_indptr), shape=info["shape"])
        return copy.copy(matrix[:, d])

    def free(self):
        """
        Free all SharedMemory
        :return:
        """
        for mem in self._memory_list:
            mem.unlink()
