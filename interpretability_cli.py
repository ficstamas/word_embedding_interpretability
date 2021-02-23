from multiprocessing import freeze_support
from argparse import ArgumentParser
from interpretability.core.config import Config
from models import *
from interpretability.reader.embedding import Embedding as EmbeddingObject
from interpretability.reader.semcat import semcat_reader
from interpretability.reader.old_semcat import read as old_semcat_reader
from interpretability.reader.semcor import read as semcor_reader
from validation import interpretability, word_retrieval_test, accuracy
import json
import os
import time
import platform
import sys
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.INFO)


if __name__ == '__main__':
    freeze_support()  # for Windows support
    parser = ArgumentParser(description='Interpretable Embedding generation')

    # Embedding
    parser.add_argument("--embedding_path", type=str, required=True,
                        help="Path to the embedding file")
    parser.add_argument("-dense", action='store_true', required=False,
                        help="Use it if it is a dense embedding space")
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
                        help="The percentage of words to drop. (Default: 0.0)")
    parser.add_argument("--seed", type=int, required=False,
                        help="Seed for random number generation (Default: None)")

    # KDE parameters
    parser.add_argument("--kde_kernel", type=str, required=False,
                        help="The kernel for kernel density estimation. Only applied when using the distances with "
                             "their continuous form (Default: gaussian)")
    parser.add_argument("--kde_bandwidth", type=float, required=False,
                        help="The bandwidth for kernel density estimation. (Default: 0.2)")

    # Distance
    parser.add_argument("--distance", type=str, required=True,
                        help="The applied distance. Available: "
                             "['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal']")
    parser.add_argument("--distance_weight", type=str, required=False,
                        choices=["none", "relative_frequency"],
                        help="Weight of distances")

    # Project
    parser.add_argument("--workspace", type=str, required=True,
                        help="Workspace where projects are going to be saved under the given --name")
    parser.add_argument("--name", type=str, required=False,
                        help="Name of the project. (Default: 'default')")
    parser.add_argument("--processes", type=int, required=False,
                        help="Number of processes to use. (Default: 2)")

    # Used Model and Validation
    parser.add_argument("--model", type=str, required=False,
                        help="The used model for calculation. Available: 'default', 'contextual', "
                             "'mcmc' (Default: 'default')")
    parser.add_argument("-load", action='store_true', required=False,
                        help="Loads existing model")
    parser.add_argument("-save", action='store_true', required=False,
                        help="Saves current model")
    parser.add_argument("-overwrite", action='store_true', required=False,
                        help="Whether to overwrite existing projects")
    parser.add_argument("-interpretability", action='store_true', required=False,
                        help="Calculates interpretability scores")
    parser.add_argument("--accuracy", type=str, required=False,
                        help="Calculates accuracy of the model. ['word_retrieval_test', 'accuracy']")
    parser.add_argument("-accuracy_preprocessed", action='store_true', required=False,
                        help="Whether to apply preprocessing (such as Standardization) on the embedding space")
    parser.add_argument("-accuracy_recalculate", action='store_true', required=False,
                        help="Whether to recalculate the cached matrix")
    parser.add_argument("--relaxation", type=int, required=False,
                        help="Relaxation parameter for interpretability calculation. The higher value means more "
                             "relaxation (Default: 10)")

    parser.add_argument("--test", type=str, required=False,
                        help="Path to the test words. (only used by contextual models atm.)")
    parser.add_argument("--test_weights", type=str, required=False,
                        help="Path to NumPy array which contains the weights for the test words. "
                             "(only used by contextual models atm.)")

    # MCMC model params
    parser.add_argument("--mcmc_acceptance", type=int, required=False,
                        help="The minimum number of accepted estimation during Metropolis–Hastings algorithm "
                             "(Default: 200)")
    parser.add_argument("--mcmc_noise", type=float, required=False,
                        help="The percentage of noise appliead to every semantic category. "
                             "Calculated based on every semantic "
                             "category's size (Default: 0.2)")

    parser.set_defaults(lines_to_read=-1, dense=False, smc_method='random', seed=None,
                        smc_rate=0.0, kde_kernel="gaussian", kde_bandwidth=0.2, name='default', processes=2,
                        model="default", interpretability=False, accuracy=None, load=False, save=False,
                        relaxation=10,
                        mcmc_acceptance=200, mcmc_noise=0.2, smc_loader='semcat', numpy=False, test_word_weights=None,
                        distance_weight=None, overwrite=False, accuracy_preprocessed=False, accuracy_recalculate=False)

    args = parser.parse_args()

    config = Config(memory_prefix=None)

    # Setting every parameter
    config.project.save = args.save
    config.set_project_path(args.workspace, args.name, args.overwrite)
    config.set_embedding(args.embedding_path, args.dense, args.lines_to_read)
    config.embedding.numpy = args.numpy
    config.set_semantic_categories(args.smc_path, args.smc_loader, args.smc_method, args.smc_rate, args.seed)
    config.set_distance(args.distance)
    config.set_kde(args.kde_kernel, args.kde_bandwidth)
    config.project.processes = args.processes
    config.model.mcmc_acceptance = args.mcmc_acceptance
    config.model.mcmc_noise = args.mcmc_noise
    config.data.test_word_weights_path = args.test_weights
    config.semantic_categories.test_words_path = args.test

    config.log_config()

    # Loading
    embedding = EmbeddingObject(config)
    config.embedding.embedding = embedding

    semcat = None
    eval_vector_labels = None
    word_vector_labels = None
    border = None
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
        semcat, word_vector_labels, eval_vector_labels, border = semcor_reader(config.semantic_categories.path, config)
        config.embedding.embedding._allocate_labels(word_vector_labels)
    else:
        config.logger.error("Invalid reader type for semantic categories")
        sys.exit(0)
    config.semantic_categories.categories = semcat

    # Models
    model = None
    if args.model == "default":
        model = DefaultModel()
    elif args.model == "mcmc":
        model = MCMCModel()
    elif args.model == "contextual":
        model = ContextualModel()

    # Init values
    if args.load:
        model.load()
    else:
        model.run()

    # weight of distances
    weight = None
    if args.distance_weight:
        model.relative_frequency()
        weight = model.relative_frequency_matrix

    # Save
    if args.save:
        model.save()
    # Interpretability
    if args.interpretability:
        interpretability.interpretability(config, lamb=args.relaxation)
    # Accuracy
    if args.accuracy is not None:
        if args.accuracy == 'word_retrieval_test':
            word_retrieval_test.accuracy(config)
        elif args.accuracy.startswith('accuracy'):
            relax = int(args.accuracy.split('@')[-1])
            accuracy.accuracy(eval_vector_labels, relaxation=relax, weight=weight,
                              preprocessed=args.accuracy_preprocessed,
                              wt=None if word_vector_labels is None else word_vector_labels,
                              recalculate=args.accuracy_recalculate,
                              border=border)

    with open(os.path.join(config.project.project, "params.config"), mode="w", encoding="utf8") as f:
        json.dump(config.__repr__(), f, indent=4)

    config.free()
