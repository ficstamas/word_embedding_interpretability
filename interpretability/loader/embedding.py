import re
import numpy as np
import scipy.sparse as sp

import gzip
from zipfile import ZipFile
from multiprocessing.shared_memory import SharedMemory
import json
import array

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


__all__ = ["Embedding", "embedding_reader"]


class Embedding(object):
    """
    This class provides utils for efficient storage and manipulation of sparse (embedding) matrices.
    Objects are assumed to be located in the rows.
    """

    def __init__(self, embedding_path, dense_input, max_words=-1):
        """
        Parameters
        ----------
        embedding_path : str
            Location of the embedding
        dense_input : bool
            Marks if it is a dense embedding
        words_to_keep : list, optional
            list of words to keep
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        """
        self._memories = []
        # Shared memory object
        self.embedding_memory_name = "embedding"
        # Type and shape of the array in shared memory
        self.embedding_memory_dtype = None
        self.embedding_memory_size = None
        self.embedding_memory_shape = None

        # Shared memory objects (dict is dumped into as json string and converted to bytes[utf8])
        self.i2w_memory_name = "i2w"
        self.i2w_memory_size = 0
        self.w2i_memory_name = "w2i"
        self.w2i_memory_size = 0

        if dense_input:
            self.load_dense_embeddings(embedding_path, max_words=max_words)
        else:
            self.load_sparse_embeddings(embedding_path, max_words=max_words)

    def load_dense_embeddings(self, path: str, max_words=-1):
        """
        Reads in the dense embedding file.

        Parameters
        ----------
        path : str
            Location of the gzipped dense embedding file
            If None, no filtering takes place.
        words_to_keep : list, optional
            list of words to keep
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        Returns
        -------
        tuple:
            w2i:
                Wordform to identifier dictionary (Can be accessed as SharedMemory: w2i),
            i2w:
                Identifier to wordform dictionary (Can be accessed as SharedMemory: i2w),
            W:
                The dense embedding matrix (Can be accessed as SharedMemory: embedding)
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
                logging.info("{} lines read in from a dense embedding file".format(len(words)))

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
                print('Error while parsing input line #{}: {}'.format(counter, line))

        W = np.array(data)

        memory = SharedMemory(self.embedding_memory_name, create=True, size=W.nbytes)

        buf = np.ndarray(W.shape, dtype=W.dtype, buffer=memory.buf)
        self.embedding_memory_dtype = W.dtype
        self.embedding_memory_shape = W.shape
        self.embedding_memory_size = W.nbytes
        buf[:, :] = W[:, :]
        del W

        i2w = dict(enumerate(words))
        w2i = {v: k for k, v in i2w.items()}

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

        # Adding shareables to list to preserve one view and prevent them them from garbage collection
        self._memories.append(i2w_memory)
        self._memories.append(w2i_memory)
        self._memories.append(memory)

    def load_sparse_embeddings(self, path, max_words=-1):
        """
        Reads in the sparse embedding file.

        Parameters
        ----------
        path : str
            Location of the gzipped sparse embedding file
            If None, no filtering takes place.
        words_to_keep : list, optional
            list of words to keep
        max_words : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        Returns
        -------
        tuple:
            w2i:
                Wordform to identifier dictionary,
            i2w:
                Identifier to wordform dictionary,
            W:
                The sparse embedding matrix
        """

        i2w = {}
        data, indices, indptr = [], [], [0]

        if path.endswith('.gz'):
            lines = gzip.open(path, 'rt', encoding='utf8')
        else:
            lines = open(path, mode='r', encoding='utf8')

        for line_number, line in enumerate(lines):

            if len(i2w) % 5000 == 0:
                logging.info("{} lines read in from a sparse embedding file".format(len(i2w)))

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
        w2i = {w: i for i, w in i2w.items()}

        sparse = sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, i + 1)).toarray()

        # Creating shared memory objects
        memory = SharedMemory(self.embedding_memory_name, create=True, size=sparse.nbytes)

        buf = np.ndarray(sparse.shape, dtype=sparse.dtype, buffer=memory.buf)
        self.embedding_memory_dtype = sparse.dtype
        self.embedding_memory_shape = sparse.shape
        self.embedding_memory_size = sparse.nbytes
        buf[:, :] = sparse[:, :]
        del sparse

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

        # Adding shareables to list to preserve one view and prevent them them from garbage collection
        self._memories.append(i2w_memory)
        self._memories.append(w2i_memory)
        self._memories.append(memory)

    @classmethod
    def buff_to_dict(cls, shr: SharedMemory, size: int) -> dict:
        json_string_bytes = array.array('B', shr.buf)[:size]
        json_string = json_string_bytes.tobytes().decode("utf8")
        dictionary = json.loads(json_string)
        return dictionary


def embedding_reader(input_file: str, dense_file: bool, lines_to_read=-1):
    """
    Reads embedding file
    Parameters
    ----------
        input_file : str
            Path to the input file.
        dense_file : bool
            True if it marks a dense embedding, false otherwise.
        lines_to_read : int, optional
            Indicates the number of lines to read in.
            If negative, the entire file gets processed.
        mcrae_dir : str, optional
            Path to the McRae file
        mcrae_words_only : bool
            Use McRae words only
    Returns
    -------
    Embedding:
        The Embedding object
    """
    path_to_embedding = input_file

    emb = Embedding(path_to_embedding, dense_file, max_words=lines_to_read)

    return emb
