import os
import logging
import numpy as np
import random
import math
from .embedding import Embedding
from .semcat import SemCat, _distance

from interpretability.core.config import Config
from multiprocessing.shared_memory import SharedMemory

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def read(input_dir: str, config: Config, embedding: Embedding, params=None):
    """
    Reads in SEMCAT categories and words
    Dont use this function, use the one in semcat.py
    This function does not sync category words and embedding vocab
    Parameters
    ----------
    input_dir: str
        The path to the directory where the categories are found.
    config: Config
        Config object
    embedding: Embedding
        embedding
    params:
        params for method
    Returns
    -------
    SemCat:
        Wrapper
    """
    weights_mem = SharedMemory(embedding.embedding_memory_name)
    W = np.ndarray(shape=embedding.embedding_memory_shape, dtype=embedding.embedding_memory_dtype,
                   buffer=weights_mem.buf)

    w2i_mem = SharedMemory(embedding.w2i_memory_name)
    w2i = embedding.buff_to_dict(w2i_mem, embedding.w2i_memory_size)

    if params is None:
        params = {"random": False,
                  "seed": None,
                  "percent": 0,
                  "center": False}

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
            size = vocab[c].__len__()
            rm_num = math.ceil(size*percent)
            vocab_size -= rm_num
            for i in range(rm_num):
                rng = random.randint(0, size-i-1)
                dropped_words[c].append(vocab[c][rng])
                del vocab[c][rng]

    if params["center"]:
        for c in vocab:
            if c not in dropped_words:
                dropped_words[c] = []
            mean = None
            for word in vocab[c]:
                try:
                    if mean is None:
                        mean = np.array([W[w2i[word]]])
                    else:
                        vector = [W[w2i[word]]]
                        mean = np.append(mean, vector, axis=0)
                except KeyError:
                    continue
            mean = np.mean(mean)
            ranks = None
            for i, word in enumerate(vocab[c]):
                try:
                    if ranks is None:
                        ranks = np.array([[i, _distance(mean, W[w2i[word]])]])
                    else:
                        ranks = np.append(ranks, [[i, _distance(mean, W[w2i[word]])]], axis=0)
                except KeyError:
                    continue
            ranks = ranks[ranks[:, 1].argsort(), :]
            drop = math.ceil(vocab[c].__len__()*percent)
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