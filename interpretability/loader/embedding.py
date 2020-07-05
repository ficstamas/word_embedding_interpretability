import re
import numpy as np
import scipy.sparse as sp

import gzip
from zipfile import ZipFile
from multiprocessing.shared_memory import SharedMemory
import json
import array
from interpretability.core.config import Config


__all__ = ["Embedding"]


class Embedding(object):
    """
    This class provides utils for efficient storage and manipulation of sparse (embedding) matrices.
    Objects are assumed to be located in the rows.
    """

    def __init__(self, config: Config):
        self.config = config
        self._memories = []
        # Shared memory object
        self.embedding_memory_name = config.memory_prefix+"embedding"
        # Type and shape of the array in shared memory
        self.embedding_memory_dtype = None
        self.embedding_memory_size = None
        self.embedding_memory_shape = None

        # Shared memory objects (dict is dumped into as json string and converted to bytes[utf8])
        self.i2w_memory_name = config.memory_prefix+"i2w"
        self.i2w_memory_size = 0
        self.w2i_memory_name = config.memory_prefix+"w2i"
        self.w2i_memory_size = 0

        if config.embedding.numpy:
            self.load_numpy(config.embedding.path)
        else:
            if config.embedding.dense:
                self.load_dense_embeddings(config.embedding.path, max_words=config.embedding.lines_to_read)
            else:
                self.load_sparse_embeddings(config.embedding.path, max_words=config.embedding.lines_to_read)

    def load_dense_embeddings(self, path: str, max_words=-1):
        """
        Reads in the dense embedding file.

        Parameters
        ----------
        path : str
            Location of the gzipped dense embedding file
            If None, no filtering takes place.
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        """
        if path.endswith('.gz'):
            lines = gzip.open(path, 'rt')
        elif path.endswith('.zip'):
            myzip = ZipFile(path)  # we assume only one embedding file to be included in a zip file
            lines = myzip.open(myzip.namelist()[0])
        else:
            lines = open(path, mode='r', encoding='utf8')
        data, words = [], []
        for counter, line in enumerate(lines):
            if len(words) % 5000 == 0:
                self.config.logger.info("{} lines read in from a dense embedding file".format(len(words)))

            if len(words) == max_words:
                break
            tokens = line.rstrip().split(' ')
            if len(words) == 0 and len(tokens) == 2 and re.match('[1-9][0-9]*', tokens[0]):
                # the first line might contain the number of embeddings and dimensionality of the vectors
                continue
            try:
                values = [float(i) for i in tokens[1:]]
                if sum([v ** 2 for v in values]) > 0:  # only embeddings with non-zero norm are kept
                    data.append(values)
                    words.append(tokens[0])
            except Exception:
                self.config.logger.error('Error while parsing input line #{}: {}'.format(counter, line))

        W = np.array(data)

        i2w = dict(enumerate(words))

        self.memory_allocation(W, i2w)

    def load_sparse_embeddings(self, path, max_words=-1):
        """
        Reads in the sparse embedding file.

        Parameters
        ----------
        path : str
            Location of the gzipped sparse embedding file
            If None, no filtering takes place.
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        """

        i2w = {}
        data, indices, indptr = [], [], [0]

        if path.endswith('.gz'):
            lines = gzip.open(path, 'rt', encoding='utf8')
        else:
            lines = open(path, mode='r', encoding='utf8')

        for line_number, line in enumerate(lines):

            if len(i2w) % 5000 == 0:
                self.config.logger.info("{} lines read in from a sparse embedding file".format(len(i2w)))

            if line_number == max_words:
                break
            parts = line.rstrip().split(' ')
            i2w[len(i2w)] = parts[0]
            for i, value in enumerate(parts[1:]):
                value = float(value)
                if value != 0:
                    data.append(float(value))
                    indices.append(i)
            indptr.append(len(indices))

        sparse = sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, i + 1)).toarray()
        self.memory_allocation(sparse, i2w)

    def load_numpy(self, path: str):
        W = None
        if self.config.embedding.dense:
            W = np.load(path)
        else:
            W = sp.load_npz(path)
            W = W.toarray()
        self._allocate_embedding(W[:self.config.embedding.lines_to_read, :]
                                 if self.config.embedding.lines_to_read != -1 else W)

    def memory_allocation(self, matrix, i2w):
        """
        Allocates shared memory objects
        :param matrix:
        :param i2w:
        :return:
        """
        self._allocate_embedding(matrix)
        self._allocate_labels(i2w)

    def _allocate_embedding(self, matrix):
        # Creating shared memory objects
        memory = SharedMemory(self.embedding_memory_name, create=True, size=matrix.nbytes)

        buf = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=memory.buf)
        self.embedding_memory_dtype = matrix.dtype
        self.embedding_memory_shape = matrix.shape
        self.embedding_memory_size = matrix.nbytes
        buf[:, :] = matrix[:, :]
        del matrix
        self._memories.append(memory)
        self.config.logger.info(
            f"Embedding in memory: {self.embedding_memory_size / 1024 / 1024:.2f} Mbytes ({self.embedding_memory_size} bytes)")

    def _allocate_labels(self, i2w):
        w2i = {w: i for i, w in i2w.items()}
        i2w_byte = array.array('B')
        i2w_byte.frombytes(json.dumps(i2w).encode("utf8"))
        w2i_byte = array.array('B')
        w2i_byte.frombytes(json.dumps(w2i).encode("utf8"))

        i2w_memory = SharedMemory(self.i2w_memory_name, create=True, size=i2w_byte.__len__())
        i2w_memory.buf[:] = i2w_byte[:]
        self.i2w_memory_size = i2w_byte.__len__()

        w2i_memory = SharedMemory(self.w2i_memory_name, create=True, size=w2i_byte.__len__())
        w2i_memory.buf[:] = w2i_byte[:]
        self.w2i_memory_size = w2i_byte.__len__()

        # Adding share ables to list to preserve one view and prevent them them from garbage collection
        self._memories.append(i2w_memory)
        self._memories.append(w2i_memory)

    @classmethod
    def buff_to_dict(cls, shr: SharedMemory, size: int) -> dict:
        json_string_bytes = array.array('B', shr.buf)[:size]
        json_string = json_string_bytes.tobytes().decode("utf8")
        dictionary = json.loads(json_string)
        return dictionary

    def free(self):
        for mem in self._memories:
            mem: SharedMemory
            mem.unlink()
