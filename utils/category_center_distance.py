import os
import json
import sys
from argparse import ArgumentParser
from .gather_accuracy import get_random_string
import numpy as np
from interpretability.loader.semcor import read
from interpretability.core.config import Config
from interpretability.loader.embedding import Embedding as EmbeddingObject
import tqdm
from multiprocessing.shared_memory import SharedMemory
from interpretability.utils.transforms import transform_embedding


def gather(workspace):
    results = {}

    if not os.path.exists(workspace):
        print(f"Path {workspace} does not exists")
        sys.exit(-1)

    for root, dirs, files in os.walk(workspace):
        config = Config(memory_prefix=None)
        if "params.config" not in files:
            continue
        # restoring config
        config.restore_from_json(os.path.join(root, "params.config"))

        # loading files
        embedding = EmbeddingObject(config)
        config.embedding.embedding = embedding

        semcor, word_vector_labels, eval_vector_labels = read(config.semantic_categories.path, config)
        config.semantic_categories.categories = semcor
        config.data.load_test_word_weights()

        config.logger.info("Loading W_D")
        distance_matrix_path = os.path.join(config.project.models, "distance_matrix.npy")
        distance_matrix = np.load(distance_matrix_path)

        eval_vector_space = transform_embedding(config.data.test_word_weights, distance_matrix)

        config.logger.info("Loading I")
        transformed_space_path = os.path.join(config.project.models, "transformed_space.npy")
        transformed_space = np.load(transformed_space_path)

        mask = np.zeros([word_vector_labels.__len__()], dtype=np.int32)

        config.logger.info("Creating masks for lexnames...")
        for i in tqdm.trange(mask.shape[0]):
            if word_vector_labels[i] == '<unknown>':
                mask[i] = -1
                continue
            lexname = word_vector_labels[i]
            lexname_id = semcor.c2i[lexname]
            mask[i] = lexname_id
        config.logger.info("Masks are done!")
        config.logger.info("Calculating distances from category centers ...")

        category_centers = None

        for i in tqdm.trange(semcor.c2i.__len__()):
            if semcor.i2c[i] == 'adj.ppl':
                continue
            indexes = np.where(mask == i)
            category_center = np.mean(transformed_space[indexes, :][0], axis=0)
            # print(category_center.shape)
            if category_centers is None:
                category_centers = np.array([category_center])
            else:
                category_centers = np.concatenate([category_centers, [category_center]], axis=0)

        # print(category_centers.shape)
        config.logger.info("Calculating dot product")
        # return
        category_distance = np.dot(eval_vector_space, category_centers.T)  # eval_vector_space.dot(category_centers.T)
        config.logger.info("Distances are calculated!")
        p = os.path.join(config.project.results, "category_distance.npy")
        np.save(p, category_distance)

        config.logger.info(f"Extracting true labels")
        true_labels = np.zeros([eval_vector_labels.__len__()], dtype=np.int64)
        for i in tqdm.trange(true_labels.shape[0]):
            if eval_vector_labels[i] == '<unknown>':
                true_labels[i] = -1
                continue
            true_labels[i] = semcor.c2i[eval_vector_labels[i]]

        res = true_labels[true_labels == np.argmax(category_distance, axis=1)].shape[0]/eval_vector_labels.__len__()
        config.logger.info(f"Accuracy: {res}")
        results[get_random_string(8)] = {'accuracy_path': p,
                                         'workspace': workspace,
                                         'name': config.project.name,
                                         'scores': res}
        config.free()
        del distance_matrix, semcor, word_vector_labels, eval_vector_labels, transformed_space, config, category_distance

    # save
    fp = open(os.path.join(workspace, "gathered_category_distance.json"), mode='w', encoding='utf8')
    json.dump(results, fp, indent=4)
    fp.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Gather results in workspace')

    # Embedding
    parser.add_argument("--workspace", type=str, required=True,
                        help="Path to the workspace")

    args = parser.parse_args()
    gather(args.workspace)
