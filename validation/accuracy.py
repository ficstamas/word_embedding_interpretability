from interpretability.core.config import Config
from interpretability.loader.semcor import Semcor
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import tqdm
import os


def accuracy(config: Config, relaxation=1):
    embedding = config.embedding.embedding
    weights_mem = SharedMemory(config.data.transformed_space)
    w = np.ndarray(shape=config.data.transformed_space_shape,
                   buffer=weights_mem.buf)

    # index to lexname
    i2w_mem = SharedMemory(embedding.i2w_memory_name)
    i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)

    semcor: Semcor
    semcor = config.semantic_categories.categories
    # semcor.word_vector_tokens contains index to token

    score = []

    for lexname in tqdm.tqdm(semcor.dropped_words):
        for token in semcor.dropped_words[lexname]:
            try:
                for i in semcor.word_vector_tokens:
                    word = semcor.word_vector_tokens[i]
                    if word == token:
                        word_vector = np.array([w[i, :].T])
                        lex_indexes = np.array([np.arange(word_vector.shape[1])])
                        pairs = np.append(word_vector, lex_indexes, axis=0)
                        sorted_word_vector = pairs[:, pairs[0, :].argsort()]
                        for j in range(1, relaxation + 1):
                            element = sorted_word_vector[:, -j]
                            if i2w[str(i)] != '<unknown>' and int(element[1]) == semcor.c2i[i2w[str(i)]]:
                                score.append(1 / j)
                            else:
                                score.append(0)
            except IndexError:
                continue

    config.logger.info(np.average(score))
    fd = open(os.path.join(config.project.results, f"accuracy@{relaxation}.txt"), mode='w')
    fd.write(str(np.average(score)))
    fd.close()
