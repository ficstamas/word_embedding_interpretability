import numpy as np
import tqdm
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Queue, Manager
from interpretability.core.config import Config
from interpretability.score.wrappers.descriptors import MemoryInfo
from queue import Empty


def V_p(V, n_j, lamb, config: Config):
    embedding = config.embedding.embedding
    # i2w
    i2w_mem = SharedMemory(embedding.i2w_memory_name)
    i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)
    return set([i2w[str(int(o))] for o in V[1, -lamb * n_j:]])


def V_n(V, n_j, lamb, config: Config):
    embedding = config.embedding.embedding
    # i2w
    i2w_mem = SharedMemory(embedding.i2w_memory_name)
    i2w = embedding.buff_to_dict(i2w_mem, embedding.i2w_memory_size)
    return set([i2w[str(int(o))] for o in V[1, :lamb * n_j]])


def is_p(i, j, config: Config, embedding_memory: MemoryInfo, lamb: int):
    lambdas = {l+1: 0 for l in range(lamb)}

    # Embedding
    weights_mem = SharedMemory(embedding_memory.name)
    w = np.ndarray(shape=embedding_memory.shape, buffer=weights_mem.buf)

    V_1 = np.array([w[:, j]])
    V_2 = np.array([np.arange(V_1.shape[1])])
    V = np.append(V_1, V_2, axis=0)
    V_sorted = V[:, V[0, :].argsort()]

    S = set(config.semantic_categories.categories.vocab[config.semantic_categories.categories.i2c[i]])
    n_i = config.semantic_categories.categories.vocab[config.semantic_categories.categories.i2c[i]].__len__()

    for l in range(lamb):
        v_p = V_p(V_sorted, n_i, l+1, config)
        v_n = V_n(V_sorted, n_i, l+1, config)

        IS_p = S.intersection(v_p).__len__() / n_i * 100
        IS_n = S.intersection(v_n).__len__() / n_i * 100

        lambdas[l+1] = max(IS_p, IS_n)
    return lambdas


def j_star(i: int, distance_matrix: np.ndarray):
    return int(np.argmax(distance_matrix[:, i, 0]).astype(dtype=np.int))


def is_i(task: int, config: Config, embedding_memory: MemoryInfo, distance_memory: MemoryInfo, lamb):
    # Distance space
    dist_mem = SharedMemory(distance_memory.name)
    distance_matrix = np.ndarray(shape=distance_memory.shape, buffer=dist_mem.buf)

    return is_p(task, j_star(task, distance_matrix), config, embedding_memory, lamb)


def score_dist(config: Config, embedding_memory: MemoryInfo, distance_memory: MemoryInfo, task_queue: Queue,
               relaxation_memory: MemoryInfo, lamb):

    relaxation_mem = SharedMemory(relaxation_memory.name)
    relaxation_matrix = np.ndarray(shape=relaxation_memory.shape, buffer=relaxation_mem.buf)

    if type(multiprocessing.current_process()) == multiprocessing.Process:
        print("poggers I am the main")
    while True:
        try:
            task = task_queue.get(True, 0.5)
        except Empty:
            config.logger.info(f"Task Queue is empty")
            break

        IS_i = is_i(task, config, embedding_memory, distance_memory, lamb)
        for i in range(lamb):
            relaxation_matrix[i, task] = IS_i[i+1]
        config.logger.info(f"Task Queue: {task_queue.qsize()}")


def score(config: Config, embedding_memory: MemoryInfo, distance_memory: MemoryInfo, lamb=5):
    """
    Calculating interpretability scores
    :param config: Config object
    :param embedding_memory: Memory info about the embedding
    :param distance_memory: Memory info about the distance matrix
    :param proc: Number of processes to use
    :param lamb: the number of lambda to compute from 1
    :return:
    """
    IS_i = []
    number_of_processes = config.project.processes
    pool = multiprocessing.Pool(processes=number_of_processes)

    # Results
    r = np.zeros([lamb, embedding_memory.shape[-1]], dtype=np.float)
    results_name = f"{config.memory_prefix}_lambdas_per_dim"
    results_mem = SharedMemory(name=results_name, create=True, size=r.nbytes)
    buf = np.ndarray(r.shape, dtype=r.dtype, buffer=results_mem.buf)
    buf[:, :] = r[:, :]
    del r

    relaxation_memory = MemoryInfo()
    relaxation_memory.name = results_name
    relaxation_memory.shape = buf.shape

    task_manager = Manager()
    task_queue = task_manager.Queue()

    for i in range(config.semantic_categories.categories.i2c.__len__()):
        task_queue.put(i)

    inputs = []
    for i in range(number_of_processes):
        inputs.append([config, embedding_memory, distance_memory, task_queue, relaxation_memory, lamb])

    with pool as p:
        _ = p.starmap(score_dist, inputs)

    res = np.mean(buf, axis=1)
    IS_i = [res[i] for i in range(res.shape[0])]

    return IS_i
