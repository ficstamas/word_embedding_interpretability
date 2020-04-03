import re
import numpy as np
import scipy.sparse as sp

import gzip
from zipfile import ZipFile
from multiprocessing.shared_memory import SharedMemory
import json
import io
import sys
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

    def __init__(self, embedding_path, dense_input, words_to_keep=None, max_words=-1):
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
        # Shared memory object
        self.memory = None
        # Type and shape of the array in shared memory
        self.dtype = None
        self.shape = None

        # Shared memory objects (dict is dumped into json string and converted to bytes[utf8])
        self.i2w_memory = None
        self.i2w_memory_size = 0
        self.w2i_memory = None
        self.w2i_memory_size = 0

        if dense_input:
            self.w2i, self.i2w = self.load_dense_embeddings(embedding_path, words_to_keep=words_to_keep,
                                                                    max_words=max_words)
        else:
            self.w2i, self.i2w = self.load_sparse_embeddings(embedding_path, words_to_keep=words_to_keep,
                                                                     max_words=max_words)

    def load_dense_embeddings(self, path: str, words_to_keep=None, max_words=-1):
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
            if words_to_keep is not None and not tokens[0] in words_to_keep:
                continue
            try:
                values = [float(i) for i in tokens[1:]]
                if sum([v ** 2 for v in values]) > 0:  # only embeddings with non-zero norm are kept
                    data.append(values)
                    words.append(tokens[0])
            except Exception:
                print('Error while parsing input line #{}: {}'.format(counter, line))

        # Adding unknown vector

        W = np.array(data)

        self.memory = SharedMemory('embedding', create=True, size=W.nbytes)

        buf = np.ndarray(W.shape, dtype=W.dtype, buffer=self.memory.buf)
        self.dtype = W.dtype
        self.shape = W.shape
        buf[:, :] = W[:, :]
        del W

        i2w = dict(enumerate(words))
        w2i = {v: k for k, v in i2w.items()}

        i2w_byte = array.array('B')
        i2w_byte.frombytes(json.dumps(i2w).encode("utf8"))
        w2i_byte = array.array('B')
        w2i_byte.frombytes(json.dumps(w2i).encode("utf8"))

        self.i2w_memory = SharedMemory("i2w", create=True, size=i2w_byte.__len__())
        self.i2w_memory.buf[:] = i2w_byte[:]
        self.i2w_memory_size = i2w_byte.__len__()

        self.w2i_memory = SharedMemory("w2i", create=True, size=w2i_byte.__len__())
        self.w2i_memory.buf[:] = w2i_byte[:]
        self.w2i_memory_size = w2i_byte.__len__()

        return w2i, i2w

    def load_sparse_embeddings(self, path, words_to_keep=None, max_words=-1):
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

            if words_to_keep is not None and parts[0] not in words_to_keep:
                continue

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
        self.memory = SharedMemory('embedding', create=True, size=sparse.nbytes)

        buf = np.ndarray(sparse.shape, dtype=sparse.dtype, buffer=self.memory.buf)
        self.dtype = sparse.dtype
        self.shape = sparse.shape
        buf[:, :] = sparse[:, :]
        del sparse

        i2w_byte = array.array('B')
        i2w_byte.frombytes(json.dumps(i2w).encode("utf8"))
        w2i_byte = array.array('B')
        w2i_byte.frombytes(json.dumps(w2i).encode("utf8"))

        self.i2w_memory = SharedMemory("i2w", create=True, size=i2w_byte.__len__())
        self.i2w_memory.buf[:] = i2w_byte[:]

        self.w2i_memory = SharedMemory("w2i", create=True, size=w2i_byte.__len__())
        self.w2i_memory.buf[:] = w2i_byte[:]

        return w2i, i2w

    def query_by_index(self, idx, top_words=25000, top_k=10):
        assert type(self.W) == sp.csr_matrix  # this method only works for sparse matrices at the moment
        relative_scores = []
        word_ids = []
        for wid, we in enumerate(self.W):
            if wid == top_words:
                break
            if idx in we.indices:
                s = np.sum(we.data)
                for i, d in zip(we.indices, we.data):
                    if i == idx:
                        relative_scores.append(d / s)
                        word_ids.append(wid)
                        break
        order = np.argsort(relative_scores)
        if top_k > 0: order = order[-top_k:]
        return [(self.i2w[word_ids[j]], relative_scores[j], word_ids[j]) for j in order]


def _constructing_mcrae_features(mcrae_dir: str):
    """
    Creating feature dictionary

    Parameters
    ----------
    mcrae_dir: str
        Path to the McRae file

    Returns
    -------
    dict:
        Concept-Index dictionary
    """
    c2i = {}

    # Iterating over the lines of the file
    for i, l in enumerate(open('{}/McRae-BRM-InPress/CONCS_FEATS_concstats_brm.txt'.format(mcrae_dir))):
        # Skipping column labels
        if i == 0:
            continue
        # Split it into columns
        parts = l.split()
        # Creating Concept-Index pairs
        if parts[0] not in c2i:
            c2i[parts[0]] = len(c2i)
    return c2i


def embedding_reader(input_file: str, dense_file: bool, lines_to_read=-1, mcrae_dir=None, mcrae_words_only=None):
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

    c2i = {}
    if mcrae_dir:
        c2i = _constructing_mcrae_features(mcrae_dir)

    if mcrae_dir is not None and mcrae_words_only is not None:
        emb = Embedding(path_to_embedding, dense_file, max_words=lines_to_read, words_to_keep=c2i.keys())
    else:
        emb = Embedding(path_to_embedding, dense_file, max_words=lines_to_read, words_to_keep=None)

    return emb
