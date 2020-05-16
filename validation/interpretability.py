from interpretability.score.categorical_interpretability import score
from interpretability.core.config import Config
from interpretability.score.wrappers.descriptors import MemoryInfo


def interpretability(config: Config, raw=False):
    """
    Calculates interpretability scores
    :param config:
    :param raw: If True calculates the interpretability of the raw embedding
    :return:
    """
    embedding_info = MemoryInfo()
    distance_info = MemoryInfo()
    if raw:
        embedding_info.name = config.embedding.embedding.embedding_memory_name
        embedding_info.shape = config.embedding.embedding.embedding_memory_shape

        distance_info.name = config.data.distance_matrix
        distance_info.shape = config.data.distance_matrix_shape
    else:
        embedding_info.name = config.data.transformed_space
        embedding_info.shape = config.data.transformed_space_shape

        distance_info.name = config.data.transformed_space_distance_matrix
        distance_info.shape = config.data.transformed_space_distance_matrix_shape

    score(config, embedding_info, distance_info)
