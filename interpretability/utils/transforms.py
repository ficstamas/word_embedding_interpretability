import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from interpretability.core.config import Config


def transform_embedding(embedding, distance_matrix):
    config = Config()
    distances = distance_matrix[:, :, 0]
    signs = distance_matrix[:, :, 1]

    # performing the usual operations
    config.logger.info("Performing L1 normalization...")

    normalized = Normalizer('l1').transform(distances.T).T

    config.logger.info("Performing sign correction...")

    sign_corrected = normalized * signs

    config.logger.info("Performing standard scaling...")

    scaled = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(embedding)

    transformed_space = scaled.dot(sign_corrected)

    config.logger.info("Transformed space calculated!")
    return transformed_space
