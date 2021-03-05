import numpy as np
import os
from src.modules.utilities.logging import Logger


def apply_transformation(embedding: np.ndarray, path: str):
    log = Logger().logger
    dpath = os.path.join(path, "model/distance_matrix.npy")
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"Distance matrix can not be found at: {dpath}")

    log.info("Loading distance matrix...")
    matrix = np.load(dpath)
    D = matrix[:, :, 0]
    S = matrix[:, :, 1]

    log.info(f"Applying distance transformation...")
    ND = D / np.linalg.norm(D, 1, axis=0)
    NSD = ND * S
    return embedding @ NSD

