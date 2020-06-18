from multiprocessing import freeze_support
from argparse import ArgumentParser
from interpretability.core.config import Config
from models import *
from interpretability.loader.embedding import Embedding as EmbeddingObject
from interpretability.loader.semcat import semcat_reader
from interpretability.loader.old_semcat import read as old_semcat_reader
from interpretability.loader.semcor import read as semcor_reader
from validation import interpretability, accuracy
import json
import os
import time
import platform
import sys

if __name__ == '__main__':
    freeze_support()  # for Windows support
    parser = ArgumentParser(description='Glove interpretibility')

    # Embedding
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to the embedding file")
    parser.add_argument("-dense", action='store_true', required=False,
                        help="Mark dense embeddings")
    parser.add_argument("-numpy", action='store_true', required=False,
                        help="Use if the embedding is stored in npy or npz format")
    parser.add_argument("--lines_to_read", type=int, required=False,
                        help="Lines to read from the embedding, to read the whole embedding set it to -1. (Default: -1)")
    # Semantic categories
    parser.add_argument("--smc_path", type=str,
                        help="Path to the Semantic Categories directory")
    parser.add_argument("--smc_loader", type=str, required=False,
                        help="The way of the semantic categories are going to be loaded. Available: "
                             "['semcat', 'old_semcat', 'semcor']. Default: 'semcat'")
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
    parser.add_argument("--model", type=str, required=False,
                        help="The used model for calculation")
    parser.add_argument("-load",action='store_true', required=False,
                        help="Calculate interpretability scores")
    parser.add_argument("-save", action='store_true', required=False,
                        help="Calculate interpretability scores")
    parser.add_argument("-interpretability", action='store_true', required=False,
                        help="Calculate interpretability scores")
    parser.add_argument("-accuracy", action='store_true', required=False,
                        help="Calculate accuracy of the model")
    parser.add_argument("--relaxation", type=int, required=False,
                        help="Lambda parameter for interpretability calculation. The higher value means more "
                             "relaxation (Default: 10)")

    # MCMC model params
    parser.add_argument("--mcmc_acceptance", type=int, required=False,
                        help="The minimum number of accepted estimation during Metropolis–Hastings algorithm "
                             "(Default: 200)")
    parser.add_argument("--mcmc_noise", type=float, required=False,
                        help="The percent of noise to every semantic category. Calculated based on every semantic "
                             "categories size (Default: 0.2)")

    parser.set_defaults(lines_to_read=-1, dense=False, smc_method='random', seed=None,
                        smc_rate=0.0, kde_kernel="gaussian", kde_bandwidth=0.2, name='default', processes=2,
                        model="default", interpretability=False, accuracy=False, load=False, save=False, relaxation=10,
                        mcmc_acceptance=200, mcmc_noise=0.2, smc_loader='semcat', numpy=False)

    args = parser.parse_args()

    # Setting up shared memory object prefixes in a platform dependent way
    if platform.system() == 'Linux':
        memory_prefix = f"{os.getuid()}_{os.getpid()}_{int(round(time.time()*1000))}_"
    elif platform.system() == 'Windows':
        memory_prefix = f"{int(round(time.time()*1000))}"
    else:
        print("The OS is not supported")
        sys.exit(0)

    config = Config(memory_prefix)

    # Setting every parameter
    config.set_project_path(args.workspace, args.name)
    config.set_embedding(args.embedding_path, args.dense, args.lines_to_read)
    config.embedding.numpy = args.numpy
    config.set_semantic_categories(args.smc_path, args.smc_loader, args.smc_method, args.smc_rate, args.seed)
    config.set_distance(args.distance)
    config.set_kde(args.kde_kernel, args.kde_bandwidth)
    config.project.processes = args.processes
    config.model.mcmc_acceptance = args.mcmc_acceptance
    config.model.mcmc_noise = args.mcmc_noise

    config.log_config()

    # Loading
    embedding = EmbeddingObject(config)
    config.embedding.embedding = embedding

    semcat = None
    if config.semantic_categories.load_method == "semcat":
        semcat = semcat_reader(config)
    elif config.semantic_categories.load_method == "old_semcat":
        params = {
            "random": config.semantic_categories.drop_method == "random",
            "seed": config.semantic_categories.seed,
            "percent": config.semantic_categories.drop_rate,
            "center": config.semantic_categories.drop_method == "category_center"
        }
        semcat = old_semcat_reader(config.semantic_categories.path, config, embedding, params)
    elif config.semantic_categories.load_method == "semcor":
        semcat, word_vector_labels = semcor_reader(config.semantic_categories.path, config)
        config.embedding.embedding._allocate_labels(word_vector_labels)
    else:
        config.logger.error("Invalid loader type for semantic categories")
        sys.exit(0)
    config.semantic_categories.categories = semcat

    # Models
    model = None
    if args.model == "default":
        model = DefaultModel()
    elif args.model == "mcmc":
        model = MCMCModel()

    # Init values
    if args.load:
        model.load()
    else:
        model.run()
    # Save
    if args.save:
        model.save()
    # Interpretability
    if args.interpretability:
        interpretability.interpretability(config, lamb=args.relaxation)
    # Accuracy
    if args.accuracy:
        accuracy.accuracy(config)

    with open(os.path.join(config.project.project, "params.config"), mode="w", encoding="utf8") as f:
        json.dump(config.__repr__(), f, indent=4)

    config.free()
