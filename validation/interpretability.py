from interpretability.score.categorical_interpretability import score
from interpretability.core.config import Config
from interpretability.score.wrappers.descriptors import MemoryInfo
import os


def interpretability(config: Config, raw=False, lamb=10):
    """
    Calculates interpretability scores
    :param config:
    :param raw: If True calculates the interpretability of the raw embedding
    :param lamb: Relaxation parameter
    :return:
    """
    config.logger.info("Calculating interpretability...")
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
    config.logger.info(embedding_info.__str__())
    config.logger.info(distance_info.__str__())
    results = score(config, embedding_info, distance_info, lamb=lamb)
    config.logger.info(f"Results: {results}")
    with open(os.path.join(config.project.results, "interpretability.txt"), mode="w", encoding="utf8") as f:
        for i, res in enumerate(results):
            f.write(f"{i+1} {res}\n")
    config.logger.info(f"Results are saved to: {os.path.join(config.project.results, 'interpretability.txt')}")
