import os
import logging
import numpy as np
import random
import math
from .embedding import Embedding

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


__all__ = ["SemCat", "semcat_reader"]


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


def semcat_reader(input_dir: str, embedding: Embedding, random=False, seed=None, percent=0, center=False) -> SemCat:
    """
    Reads in SEMCAT categories and words

    Parameters
    ----------
    input_dir: str
        The path to the directory where the categories are found.
    embedding: Embedding
        The object of the embedding to ensure the match with the vocab
    random: bool
        Flag to drop words randomly from categories
    seed: int
        Seed for the random number generation
    percent:
        The percentage of the words to be dropped
    center:
        Dropping words from categories based on the category centers
    Returns
    -------
    SemCat:
        Wrapper
    """
    if random and center:
        raise Exception("Use 'random' or 'center' not both!")

    vocab = {}
    vocab_size = 0

    w2i, i2w = {}, {}

    id = 0

    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            category_name = file.rstrip('.txt').split('-')[0]
            with open(os.path.join(input_dir, file), mode='r', encoding='utf8') as f:
                words = f.read().splitlines()
                vocab_size += words.__len__()
                w2i[category_name] = id
                vocab[category_name] = words
                id += 1

    random.seed(params["seed"])
    percent = params["percent"]
    rng_dropout = params["random"]

    dropped_words = {}

    if rng_dropout:
        for c in vocab:
            if c not in dropped_words:
                dropped_words[c] = []

            existing_words = []
            for word in vocab[c]: # iterating over words
                try:
                    _ = embedding.W[embedding.w2i[word]]
                    existing_words.append(word)
                except KeyError:
                    continue

            size = existing_words.__len__()
            rm_num = math.floor(size*percent)
            vocab_size -= rm_num
            for i in range(rm_num):
                rng = random.randint(0, size-i-1)
                dropped_words[c].append(existing_words[rng])
                del existing_words[rng]
            vocab[c] = existing_words

    if params["center"]:
        for c in vocab: # c is the category
            if c not in dropped_words: # populating dropped_words with empty lists
                dropped_words[c] = []
            mean = None
            for word in vocab[c]: # iterating over words
                try:
                    # getting coefficients of the words of c
                    if mean is None:
                        mean = np.array([embedding.W[embedding.w2i[word]]])
                    else:
                        vector = [embedding.W[embedding.w2i[word]]]
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
                        ranks = np.array([[i, _distance(mean, embedding.W[embedding.w2i[word]])]])
                    else:
                        ranks = np.append(ranks, [[i, _distance(mean, embedding.W[embedding.w2i[word]])]], axis=0)
                except KeyError:
                    continue
            ranks = ranks[ranks[:, 1].argsort(), :]
            drop = math.floor(ranks.shape[0]*percent)
            vocab_size -= drop

            word_list = []
            for i in range(drop):
                index = int(ranks[-i-1, 0])
                dropped_words[c].append(vocab[c][index])
                word_list.append(vocab[c][index])
            for w in word_list:
                vocab[c].remove(w)

    i2w = {v: k for k, v in w2i.items()}

    logging.info(
        f"{vocab.__len__()} categories are read from SEMCAT files, which contain overall {vocab_size} words.")
    return SemCat(vocab, w2i, i2w, dropped_words)


def _distance(x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.sum(np.power(x-y, 2)))
