from multiprocessing import freeze_support
from argparse import ArgumentParser
from src.modules.utilities.file_system import init_filesystem
from src.modules.utilities.logging import Logger
from src.modules.load.numpy import Embedding
from src.modules.load.semcor import lexname_as_label
import sys
from src.modules.transform.pipeline import Pipeline

LABEL_PROCESSORS = {
    "semcor-lexname": lexname_as_label
}


if __name__ == '__main__':
    freeze_support()  # for Windows support
    # Processing Arguments
    parser = ArgumentParser(description='Calculates Transformation Matrices for preprocessing.')

    # Embedding
    parser.add_argument("--train", type=str, required=True,
                        help="Path to the embedding space which serves as the train set. (It can be a .npy for dense "
                             "and .npz for sparse representations)")
    # Project
    parser.add_argument("--path", type=str, required=True,
                        help="Path for the project")
    # Configuration
    parser.add_argument("--configuration", type=str, required=True,
                        help="Path for the configuration file which determines the preprocessing procedure")
    # Jobs
    parser.add_argument("--jobs", type=int, default=2,
                        help="Number of jobs.")

    args = parser.parse_args()

    init_filesystem(args.path, ("logs", "results", "model", "transforms"))
    logger_object = Logger()
    logger_object.setup(args.path)
    embeddings = Embedding(args.path)

    try:
        embeddings.load_train(args.train, keep_in_memory=True, transform=False)
        pipeline = Pipeline(args.configuration)
        pipeline.fit(embeddings.train)
        pipeline.save(args.path)
    except Exception:
        # Handling and logging unexpected Exceptions
        logger_object.logger.exception("Exited Program because an unexpected event occurred!")
        embeddings.free()
        sys.exit(1)

    embeddings.free()

    # breakpoint()
