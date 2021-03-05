from multiprocessing import freeze_support
from argparse import ArgumentParser
from src.modules.utilities.file_system import init_filesystem
from src.modules.utilities.logging import Logger
from src.modules.load.numpy import Embedding
from src.modules.load.semcor import lexname_as_label
import sys
from src.modules.multiprocessing.distance_matrix import Distance
import numpy as np
import os

LABEL_PROCESSORS = {
    "semcor-lexname": lexname_as_label
}


if __name__ == '__main__':
    freeze_support()  # for Windows support
    # Processing Arguments
    parser = ArgumentParser(description='Calculates Distance matrix.')

    # Embedding
    parser.add_argument("--train", type=str, required=True,
                        help="Path to the embedding space which serves as the train set. (It can be a .npy for dense "
                             "and .npz for sparse representations)")
    # Labels
    parser.add_argument("--train_labels", type=str, required=True,
                        help="Path to labels for train set.")
    # Processing for Labels
    parser.add_argument("--label_processor", type=str, required=True, choices=["semcor-lexname"],
                        help="The way the labels from the input files are going to be processed. (semcor)")
    # Project
    parser.add_argument("--path", type=str, required=True,
                        help="Path for the project")
    # Distance
    parser.add_argument("--distance", type=str, required=True,
                        choices=['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal',
                                 'bhattacharyya_exponential', 'hellinger_exponential'],
                        help="Applied distance.")
    parser.add_argument("--kde_kernel", type=str, required=False, default='gaussian',
                        choices=['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'],
                        help="The kernel for kernel density estimation. Only applied when using the distances with "
                             "their continuous form (Default: gaussian)")
    parser.add_argument("--kde_bandwidth", type=float, required=False, default=0.2,
                        help="The bandwidth for kernel density estimation. (Default: 0.2)")
    # Jobs
    parser.add_argument("--jobs", type=int, default=2,
                        help="Number of jobs.")

    args = parser.parse_args()

    init_filesystem(args.path, ("logs", "results", "model", "transforms"))
    logger_object = Logger()
    logger_object.setup(args.path)
    embeddings = Embedding()

    distance_params = {
        'kernel': args.kde_kernel,
        'bandwidth': args.kde_bandwidth
    }
    # TODO apply transformations
    try:
        embeddings.load(args.train)
        train_labels, test_labels = LABEL_PROCESSORS[args.label_processor](args.train_labels)
        distance_process = Distance((embeddings.memory_info["train"]["shape"][1], len(train_labels.l2i), 2),
                                    args.distance, distance_params, train_labels, embeddings)
        distance_process.run(args.jobs)
        distance_process.save(args.path)
        distance_process.free()
        np.save(os.path.join(args.path, "model/label_frequency.npy"), train_labels.label_frequency())
        # breakpoint()
    except Exception:
        # Handling and logging unexpected Exceptions
        logger_object.logger.exception("Exited Program because an unexpected event occurred!")
        embeddings.free()
        sys.exit(1)

    embeddings.free()

    # breakpoint()
