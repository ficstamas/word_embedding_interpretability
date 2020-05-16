from multiprocessing import freeze_support
from argparse import ArgumentParser
from interpretability.core.config import Config
from models import *
from interpretability.loader.embedding import Embedding as EmbeddingObject
from interpretability.loader.semcat import semcat_reader
from validation import interpretability, accuracy
import json
import os


if __name__ == '__main__':
    freeze_support()  # for Windows support
    parser = ArgumentParser(description='Glove interpretibility')

    # Embedding
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to the embedding file")
    parser.add_argument("-dense", action='store_true', required=False,
                        help="Mark dense embeddings")
    parser.add_argument("--lines_to_read", type=int, required=False,
                        help="Lines to read from the embedding, to read the whole embedding set it to -1. (Default: -1)")
    # Semantic categories
    parser.add_argument("--smc_path", type=str,
                        help="Path to the Semantic Categories directory")
    parser.add_argument("--smc_method", type=str, required=False,
                        help="Method to drop words from semantic categories. Available: ['random', 'category_center']")
    parser.add_argument("--smc_rate", type=float, required=False,
                        help="The percent to drop. (Default: 0.0)")
    parser.add_argument("--seed", type=int, required=False,
                        help="Seed for random number generation (Default: None)")

    # KDE parameters
    parser.add_argument("--kde_kernel", type=str, required=False,
                        help="The kernel for kernel density estimation. (Default: gaussian)")
    parser.add_argument("--kde_bandwidth", type=float, required=False,
                        help="The bandwidth for kernel density estimation. (Default: 0.2)")

    # Distance
    parser.add_argument("--distance", type=str, required=True,
                        help="Method to measure distance. Available: "
                             "['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal']")

    # Project
    parser.add_argument("--workspace", type=str, required=True,
                        help="Workspace where projects are going to be saved under the given --name")
    parser.add_argument("--name", type=str, required=False,
                        help="Name of the project. (Default: 'default')")
    parser.add_argument("--processes", type=int, required=False,
                        help="Number of processes to use. (Default: 2)")

    # Used Model and Validation
    parser.add_argument("--model", type=str, required=True,
                        help="The used model for calculation")
    parser.add_argument("-load", type=bool, required=False,
                        help="Calculate interpretability scores")
    parser.add_argument("-save", type=bool, required=False,
                        help="Calculate interpretability scores")
    parser.add_argument("-interpretability", type=bool, required=False,
                        help="Calculate interpretability scores")
    parser.add_argument("-accuracy", type=bool, required=False,
                        help="Calculate accuracy of the model")

    parser.set_defaults(lines_to_read=-1, dense=False, smc_method='random', seed=None,
                        smc_rate=0.0, kde_kernel="gaussian", kde_bandwidth=0.2, name='default', processes=2,
                        model="default", interpretability=False, accuracy=False, load=False, save=False)

    args = parser.parse_args()

    config = Config()

    # Setting every parameter
    config.set_project_path(args.workspace, args.name)
    config.set_embedding(args.embedding_path, args.dense, args.lines_to_read)
    config.set_semantic_categories(args.smc_path, args.smc_method, args.smc_rate, args.seed)
    config.set_distance(args.distance)
    config.set_kde(args.kde_kernel, args.kde_bandwidth)
    config.project.processes = args.processes

    config.log_config()

    # Loading
    embedding = EmbeddingObject(config)
    config.embedding.embedding = embedding
    semcat = semcat_reader(config)
    config.semantic_categories.categories = semcat

    # Models
    if args.model == "default":
        model = DefaultModel()
        if args.load:
            model.load()
        else:
            model.run()

        if args.save():
            model.save()

        if args.interpretability:
            interpretability.interpretability(config)

        if args.accuracy:
            accuracy.accuracy(config)

    with open(os.path.join(config.project.project, "params.config"), mode="w", encoding="utf8") as f:
        json.dump(config.__repr__(), f, indent=4)