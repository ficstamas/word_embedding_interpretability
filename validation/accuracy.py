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

    score = []

    validation_vocab = list()
    validation_label = list()
    for lexname in semcor.dropped_words:
        for token in semcor.dropped_words[lexname]:
            validation_vocab.append(token)
            validation_label.append(semcor.c2i[lexname])

    test_weights = None
    test_labels = []
    argmax_matrix = None

    if not os.path.exists(os.path.join(config.project.results, "accuracy.npz")):
        config.logger.info("Calculating ordered accuracy matrix...")
        j = 0
        for i in tqdm.trange(w.shape[0]):
            if i2w[str(i)] == '<unknown>':
                continue
            if test_weights is None and semcor.word_vector_tokens[i] in validation_vocab:
                k = validation_vocab.index(semcor.word_vector_tokens[i])
                sw = w[i, :]
                test_weights = sw[np.newaxis, :]
                # test_labels[j] = [validation_label[k], validation_vocab[k]]
                test_labels.append(validation_label[k])
                j += 1
            else:
                if semcor.word_vector_tokens[i] in validation_vocab:
                    k = validation_vocab.index(semcor.word_vector_tokens[i])
                    test_weights = np.concatenate([test_weights, w[np.newaxis, i, :]], axis=0)
                    test_labels.append(validation_label[k])
                    j += 1

        for i in tqdm.trange(test_weights.shape[0]):
            word_vector = np.array([w[i, :].T])
            lex_indexes = np.array([np.arange(word_vector.shape[1])])
            pairs = np.append(word_vector, lex_indexes, axis=0)
            sorted_word_vector = pairs[:, pairs[0, :].argsort()]
            if argmax_matrix is None:
                argmax_matrix = sorted_word_vector[np.newaxis, 1]
            else:
                argmax_matrix = np.concatenate([argmax_matrix, sorted_word_vector[np.newaxis, 1]], axis=0)
        test_labels = np.array(test_labels)
        np.savez(os.path.join(config.project.results, "accuracy.npz"), labels=test_labels.astype(np.int8), values=argmax_matrix.astype(np.int8))
    else:
        config.logger.info("Loading ordered accuracy matrix...")
        arrs = np.load(os.path.join(config.project.results, "accuracy.npz"))
        test_labels = arrs['labels']
        argmax_matrix = arrs['values']

    score = []

    for i in range(argmax_matrix.shape[0]):
        appended = False
        for j in range(1, relaxation + 1):
            value = argmax_matrix[i, -j]
            if value == test_labels[i]:
                score.append(1/j)
                appended = True
        if not appended:
            score.append(0)

    config.logger.info(f"Accuracy is {np.mean(score)}")
    fd = open(os.path.join(config.project.results, f"accuracy.txt"), mode='a')
    fd.write(f"{np.mean(score)}@{relaxation}\n")
    fd.close()

    # for lexname in tqdm.tqdm(semcor.dropped_words):
    #     for token in semcor.dropped_words[lexname]:
    #         try:
    #             for i in semcor.word_vector_tokens:
    #                 word = semcor.word_vector_tokens[i]
    #                 if word == token:
    #                     word_vector = np.array([w[i, :].T])
    #                     lex_indexes = np.array([np.arange(word_vector.shape[1])])
    #                     pairs = np.append(word_vector, lex_indexes, axis=0)
    #                     sorted_word_vector = pairs[:, pairs[0, :].argsort()]
    #                     appended = False
    #                     for j in range(1, relaxation + 1):
    #                         element = sorted_word_vector[:, -j]
    #                         if i2w[str(i)] != '<unknown>' and int(element[1]) == semcor.c2i[i2w[str(i)]]:
    #                             score.append(1 / j)
    #                             appended = True
    #                     if not appended:
    #                         score.append(0)
    #         except IndexError:
    #             continue

    # config.logger.info(np.average(score))
    # fd = open(os.path.join(config.project.results, f"accuracy@{relaxation}.txt"), mode='w')
    # fd.write(str(np.average(score)))
    # fd.close()
