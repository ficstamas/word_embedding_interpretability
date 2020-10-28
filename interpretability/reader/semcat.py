import os
import logging
import numpy as np
import random
import math
from .embedding import Embedding
from interpretability.core.config import Config
from multiprocessing.shared_memory import SharedMemory
import json
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


__all__ = ["SemCat"]


class SemCat:
    """
    Wraps vocab and converter dictionaries
    """
    def __init__(self, vocab, c2i, i2c, dw):
        self._vocab = vocab
        self._c2i = c2i
        self._i2c = i2c
        self._dropped_words = dw

    @property
    def vocab(self) -> dict:
        """
        Returns a dictionary where keys are the categories and the values are the lists of the words related to them
        Returns
        -------
        dict:
            key -> [List]
        """
        return self._vocab

    @property
    def c2i(self):
        """
        Category name to index dictionary
        Returns
        -------
        dict:
            key -> value
        """
        return self._c2i

    @property
    def i2c(self):
        """
        Index to category name dictionary
        Returns
        -------
        dict:
            key -> value
        """
        return self._i2c

    @property
    def dropped_words(self) -> dict:
        """
        Randomly dropped words
        Returns
        -------
            key -> [list]
        """
        return self._dropped_words


def random_drop(vocab, embedding: Embedding, percent, vocab_size):
    weights_mem = SharedMemory(embedding.embedding_memory_name)
    W = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                   buffer=weights_mem.buf)

    w2i_mem = SharedMemory(embedding.w2i_memory_name)
    w2i = embedding.buff_to_dict(w2i_mem, embedding.w2i_memory_size)

    dropped_words = {}
    for c in vocab:
        if c not in dropped_words:
            dropped_words[c] = []

        existing_words = []
        for word in vocab[c]:  # iterating over words
            try:
                _ = W[w2i[word]]
                existing_words.append(word)
            except KeyError:
                continue

        size = existing_words.__len__()
        rm_num = math.floor(size * percent)
        vocab_size -= rm_num
        for i in range(rm_num):
            rng = random.randint(0, size - i - 1)
            dropped_words[c].append(existing_words[rng])
            del existing_words[rng]
        vocab[c] = existing_words

    return dropped_words


def category_center(vocab, embedding: Embedding, percent, vocab_size):
    weights_mem = SharedMemory(embedding.embedding_memory_name)
    W = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                   buffer=weights_mem.buf)

    w2i_mem = SharedMemory(embedding.w2i_memory_name)
    w2i = embedding.buff_to_dict(w2i_mem, embedding.w2i_memory_size)

    dropped_words = {}
    for c in vocab:  # c is the category
        if c not in dropped_words:  # populating dropped_words with empty lists
            dropped_words[c] = []
        mean = None
        for word in vocab[c]:  # iterating over words
            try:
                # getting coefficients of the words of c
                if mean is None:
                    mean = np.array([W[w2i[word]]])
                else:
                    vector = [W[w2i[word]]]
                    mean = np.append(mean, vector, axis=0)
            except KeyError:
                continue
        # getting the category center
        mean = np.mean(mean)
        ranks = None
        # calculating distances from the category centers and sorting by that afterwards
        for i, word in enumerate(vocab[c]):
            try:
                if ranks is None:
                    ranks = np.array([[i, _distance(mean, W[w2i[word]])]])
                else:
                    ranks = np.append(ranks, [[i, _distance(mean, W[w2i[word]])]], axis=0)
            except KeyError:
                continue
        ranks = ranks[ranks[:, 1].argsort(), :]
        drop = math.floor(ranks.shape[0] * percent)
        vocab_size -= drop

        word_list = []
        for i in range(drop):
            index = int(ranks[-i - 1, 0])
            dropped_words[c].append(vocab[c][index])
            word_list.append(vocab[c][index])
        for w in word_list:
            vocab[c].remove(w)

    return dropped_words


def semcat_reader(config: Config) -> SemCat:
    """
    Reads in SEMCAT categories and words

    Parameters
    ----------
    config: Config
        reference
    Returns
    -------
    SemCat:
        Wrapper
    """
    vocab = {}
    vocab_size = 0

    w2i, i2w = {}, {}

    id = 0
    # Read categories
    if config.semantic_categories.file_format == "json":
        vocab = json.load(open(config.semantic_categories.path, mode="r", encoding="utf8"))
        for key in vocab:
            w2i[key] = id
            id += 1
    else:
        for file in os.listdir(config.semantic_categories.path):
            if file.endswith(".txt"):
                category_name = file.rstrip('.txt').split('-')[0]
                with open(os.path.join(config.semantic_categories.path, file), mode='r', encoding='utf8') as f:
                    words = f.read().splitlines()
                    vocab_size += words.__len__()
                    w2i[category_name] = id
                    vocab[category_name] = words
                    id += 1

    random.seed(config.semantic_categories.seed)
    percent = config.semantic_categories.drop_rate

    dropped_words = None

    if config.semantic_categories.drop_method == 'random':
        dropped_words = random_drop(vocab, config.embedding.embedding, percent, vocab_size)
    elif config.semantic_categories.drop_method == 'category_center':
        dropped_words = category_center(vocab, config.embedding.embedding, percent, vocab_size)

    i2w = {v: k for k, v in w2i.items()}

    config.logger.info(
        f"{vocab.__len__()} categories are read from SEMCAT files, which contain overall {vocab_size} words.")
    return SemCat(vocab, w2i, i2w, dropped_words)


def _distance(x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.sum(np.power(x-y, 2)))
