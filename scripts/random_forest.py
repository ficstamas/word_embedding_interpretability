from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from interpretability.loader.semcor import read_with_path
import numpy as np
import tqdm


def train(model: RandomForestClassifier, x, y):
    model.fit(x, y)


def test(model: RandomForestClassifier, x, y):
    return model.score(x, y)


if __name__ == '__main__':
    parser = ArgumentParser(description='')

    # Embedding
    parser.add_argument("--train_vectors", type=str,
                        help="Training weights")
    parser.add_argument("--test_vectors", type=str,
                        help="Testing weights")
    parser.add_argument("--semcor_train", type=str,
                        help="Semcor train path")
    parser.add_argument("--semcor_test", type=str,
                        help="Semcor test path")
    parser.add_argument("--name", type=str,
                        help="")
    parser.add_argument("--output", type=str,
                        help="")

    args = parser.parse_args()

    lr = RandomForestClassifier(random_state=0)

    print("Loading labels...")

    semcor, train_label_dict, test_label_dict = read_with_path(args.semcor_train, args.semcor_test)

    print("Creating mask...")

    y_train = np.zeros(train_label_dict.__len__())
    mask = np.zeros(train_label_dict.__len__(), dtype=bool)

    id = 0
    for key in tqdm.tqdm(train_label_dict):
        if '<unknown>' != train_label_dict[key]:
            y_train[id] = semcor.c2i[train_label_dict[key]]
            mask[id] = True
        id += 1

    y_train = y_train[mask]

    print("Loading training features...")

    x_train = np.load(args.train_vectors)[mask, :]

    print("Creating mask...")

    y_test = np.zeros(test_label_dict.__len__())
    mask = np.zeros(test_label_dict.__len__(), dtype=bool)

    id = 0
    for key in tqdm.tqdm(test_label_dict):
        if '<unknown>' != test_label_dict[key]:
            y_test[id] = semcor.c2i[test_label_dict[key]]
            mask[id] = True
        id += 1

    y_test = y_test[mask]

    print("Loading test features...")

    x_test = np.load(args.test_vectors)[mask, :]

    print("Fitting...")

    train(lr, x_train, y_train)

    print("Testing...")
    score = test(lr, x_test, y_test)

    print(f"Score for {args.name} is {score}")

    with open(args.output, mode="a", encoding="utf8") as f:
        f.write(f"\n{args.name}:{score}")
