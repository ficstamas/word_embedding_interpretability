from argparse import ArgumentParser
from src.modules.utilities.file_system import init_filesystem
from src.modules.utilities.logging import Logger
from src.modules.load.numpy import Embedding
from src.modules.load.semcor import lexname_as_label_eval
import sys
from src.modules.transform import distance_matrix
from src.modules.evaluation import argmax
import json
import os


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

    args = parser.parse_args()

    init_filesystem(args.path, ("logs", "results", "model", "transforms"))
    logger_object = Logger()
    logger_object.setup(args.path)
    embeddings = Embedding()

    try:
        embeddings.load_test(args.test, keep_in_memory=True)
        test_labels = LABEL_PROCESSORS[args.label_processor](args.test_labels)
        transformed_space = distance_matrix.apply_transformation(embeddings.test, args.path)
        results = argmax.test(transformed_space, test_labels)
        out_path = os.path.join(args.path, "results/argmax.json")
        with open(out_path, mode="w") as f:
            json.dump(results, f, indent=4)
        logger_object.logger.info(f"Results are dumped at: {out_path}")
    except Exception:
        # Handling and logging unexpected Exceptions
        logger_object.logger.exception("Exited Program because an unexpected event occurred! Checkout the error logs to"
                                       "learn more: `logs/error.log`")
        embeddings.free()
        sys.exit(1)

    embeddings.free()

    # breakpoint()
