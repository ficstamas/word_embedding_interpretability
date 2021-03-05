from argparse import ArgumentParser
from src.modules.utilities.file_system import init_filesystem
from src.modules.utilities.logging import Logger
from src.modules.load.numpy import Embedding
from src.modules.load.semcor import lexname_as_label_eval
import sys
from src.modules.transform import distance_matrix
from src.modules.evaluation import EVALUATION_MAP
import json
import os
import numpy as np


LABEL_PROCESSORS = {
    "semcor-lexname": lexname_as_label_eval
}


if __name__ == '__main__':
    # Processing Arguments
    parser = ArgumentParser(description='Calculates Transformation Matrix for preprocession')

    # Embedding
    parser.add_argument("--test", type=str, required=True,
                        help="Path to the embedding space which serves as the train set. (It can be a .npy for dense "
                             "and .npz for sparse representations)")
    # Labels
    parser.add_argument("--test_labels", type=str, required=True,
                        help="Path to labels for train set.")
    # Processing for Labels
    parser.add_argument("--label_processor", type=str, required=True, choices=["semcor-lexname"],
                        help="The way the labels from the input files are going to be processed. (semcor)")
    # Project
    parser.add_argument("--path", type=str, required=True,
                        help="Path for the project")
    parser.add_argument("-save", action="store_true", default=False,
                        help="Whether to save the transformed space")
    # Weights
    parser.add_argument("-label_frequency", action="store_true", default=False,
                        help="Applying label frequency based weighting on the output embedding.")
    # Evaluation method
    parser.add_argument("--evaluation_method", type=str, default="argmax", choices=['argmax'],
                        help="Evaluation method")


    args = parser.parse_args()

    init_filesystem(args.path, ("logs", "results", "model", "transforms"))
    logger_object = Logger()
    logger_object.setup(args.path)
    embeddings = Embedding()
    weights = {
        "label_frequency": args.label_frequency
    }

    try:
        embeddings.load_test(args.test, keep_in_memory=True)
        test_labels = LABEL_PROCESSORS[args.label_processor](args.test_labels)
        transformed_space = distance_matrix.apply_transformation(embeddings.test, args.path)
        if weights["label_frequency"]:
            normed_freq = np.load(os.path.join(args.path, "model/label_frequency.npy"))
            normed_freq = normed_freq / np.linalg.norm(normed_freq, 2)
            transformed_space = transformed_space * normed_freq
        if args.save:
            np.save(os.path.join(args.path, "model/evaluation_space.npy"), transformed_space)
        params = {
            "save": True,
            "path": args.path
        }
        results = EVALUATION_MAP[args.evaluation_method](transformed_space, test_labels, **params)
    except Exception:
        # Handling and logging unexpected Exceptions
        logger_object.logger.exception("Exited Program because an unexpected event occurred! Checkout the error logs to"
                                       " earn more: `logs/error.log`")
        embeddings.free()
        sys.exit(1)

    embeddings.free()

    # breakpoint()
