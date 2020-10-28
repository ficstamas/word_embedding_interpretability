import os
import json
import sys
from argparse import ArgumentParser
from .gather_accuracy import get_random_string
import numpy as np
from interpretability.reader.semcor import read
from interpretability.core.config import Config
from interpretability.reader.embedding import Embedding as EmbeddingObject
import tqdm
from interpretability.utils.transforms import transform_embedding
from sklearn.preprocessing import StandardScaler, Normalizer


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

        # loading train and test words from semcor
        semcor, word_vector_labels, eval_vector_labels = read(config.semantic_categories.path, config)
        config.semantic_categories.categories = semcor
        config.data.load_test_word_weights()

        # lading distance matrix
        config.logger.info("Loading W_D")
        distance_matrix_path = os.path.join(config.project.models, "distance_matrix.npy")
        distance_matrix = np.load(distance_matrix_path)

        # transforming test space
        eval_vector_space = transform_embedding(config.data.test_word_weights, distance_matrix)

        # loading trained transformed space
        config.logger.info("Loading I")
        transformed_space_path = os.path.join(config.project.models, "transformed_space.npy")
        transformed_space = np.load(transformed_space_path)

        # creating masks for the semantic categories
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

        # calculating the category centers
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

        # calculating the distances
        config.logger.info("Calculating dot product")
        # return
        category_centers = Normalizer('l2').transform(category_centers)
        # eval_vector_space = StandardScaler().fit_transform(eval_vector_space)
        category_distance = np.dot(eval_vector_space, category_centers.T)  # eval_vector_space.dot(category_centers.T)
        # category_distance = Normalizer('l1').transform(category_distance.T).T # normalizing vectors

        config.logger.info("Distances are calculated!")
        p = os.path.join(config.project.results, "category_distance.npy")
        np.save(p, category_distance)

        # checking the labels
        config.logger.info(f"Extracting true labels")
        true_labels = np.zeros([eval_vector_labels.__len__()], dtype=np.int64)
        for i in tqdm.trange(true_labels.shape[0]):
            if eval_vector_labels[i] == '<unknown>':
                true_labels[i] = -1
                continue
            true_labels[i] = semcor.c2i[eval_vector_labels[i]]
        
        mask = np.zeros(true_labels.shape[0], dtype=np.bool)

        mask[np.where(true_labels != -1)] = True
        # mask = ~mask        
        config.logger.info(f"True labels: {true_labels.shape}, prediction: {np.argmax(category_distance, axis=1).shape}\n mask shape: {mask.shape}")
        true_labels_masked = true_labels[mask]
        predicted_labels_masked = np.argmax(category_distance[mask, :], axis=1)
        config.logger.info(f"plm: {predicted_labels_masked.shape}, tlm: {true_labels_masked.shape}")
        res = true_labels_masked[true_labels_masked == predicted_labels_masked].shape[0]/true_labels_masked.shape[0]
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
